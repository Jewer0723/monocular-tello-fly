"""
Tello 三階段任務控制系統（修正版）
狀態機: MIDAS → FORWARD → CIRCLE → MIDAS（永不自動降落）
"""

import torch
import cv2
import numpy as np
from djitellopy import Tello
import time
import pygame
from collections import deque
from ultralytics import YOLO
from pyzbar import pyzbar
import csv
import os
from datetime import datetime

# ===================== 全局配置 =====================
FRAME_W, FRAME_H = 640, 480
CONTROL_INTERVAL = 0.1

# ===================== 狀態定義 =====================
class DroneState:
    MIDAS = "MIDAS"        # 巡航避障模式
    FORWARD = "FORWARD"    # 前進接近目標模式
    CIRCLE = "CIRCLE"      # 環繞掃描模式
    # 移除 LANDING 狀態，降落完全手動

# ===================== MidAS 巡航參數 =====================
MIDAS_CONFIG = {
    "BASE_FORWARD": 20,
    "TURN_SPEED": 40,
    "OBSTACLE_THRESHOLD": 0.35,
    "CLEAR_THRESHOLD": 0.25,
    "TURN_DURATION": 1.5,
    "SMOOTHING_WINDOW": 5,
    "TARGET_FOUND_AREA": 30000,      # 找到目標的最小面積
    "TARGET_LOST_TIMEOUT": 5,        # MIDAS模式下目標丟失多久才清除目標標記
}

# ===================== 前進追蹤參數 =====================
FORWARD_CONFIG = {
    "TARGET_AREA": 120000,
    "AREA_TOLERANCE": 15000,
    "KP_YAW": 0.25,
    "KP_UPDOWN": 0.25,
    "KP_FORWARD": 0.0006,
    "MAX_SPEED": 20,
    "DEADZONE": 20,
    "MIN_AREA": 30000,              # 最小有效面積（低於此視為丟失目標）
    "TARGET_LOST_TIMEOUT": 3,       # 目標丟失3秒就放棄追蹤
    "MAX_EXECUTION_TIME": 30,       # 最長追蹤30秒（超過返回巡航，不降落）
}

# ===================== 環繞掃描參數 =====================
CIRCLE_CONFIG = {
    "ORBIT_SPEED": 9,
    "MIN_SCAN_TIME": 10,           # 最少掃描10秒
    "MAX_SCAN_TIME": 50,           # 最多掃描50秒
    "TARGET_LOST_TIMEOUT": 3,      # 目標丟失3秒就放棄環繞
    "ALTITUDE_OFFSET": 30,         # 切換模式時的高度調整
    "CSV_FILE": "scanned_codes.csv"
}

# ===================== MidAS 巡航避障控制器 =====================
class MidASCruiser:
    """純巡航避障，不參與任何目標追蹤邏輯"""
    def __init__(self):
        # 初始化 MiDaS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("MidAS using device:", self.device)

        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

        # 平滑佇列
        self.center_queue = deque(maxlen=MIDAS_CONFIG["SMOOTHING_WINDOW"])
        self.left_queue = deque(maxlen=MIDAS_CONFIG["SMOOTHING_WINDOW"])
        self.right_queue = deque(maxlen=MIDAS_CONFIG["SMOOTHING_WINDOW"])

        # 狀態
        self.state = "FORWARD"
        self.turn_start_time = 0
        self.obstacle_count = 0

    def process_frame(self, frame):
        """處理深度圖"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

        # 獲取區域深度
        center_val, left_val, right_val = self._get_depth_regions(depth_norm)

        # 平滑處理
        self.center_queue.append(center_val)
        self.left_queue.append(left_val)
        self.right_queue.append(right_val)

        center_avg = np.mean(self.center_queue)

        return depth_norm, center_avg

    def _get_depth_regions(self, depth_map):
        """獲取各區域深度值"""
        h, w = depth_map.shape
        center_height = h // 3
        center_width = w // 3
        center_top = h // 2 - center_height // 2
        center_left = w // 2 - center_width // 2

        center = depth_map[center_top:center_top + center_height,
                 center_left:center_left + center_width]
        left = depth_map[center_top:center_top + center_height, :w // 4]
        right = depth_map[center_top:center_top + center_height, 3 * w // 4:]

        center_val = np.median(center) if center.size > 0 else 0.5
        left_val = np.median(left) if left.size > 0 else 0.5
        right_val = np.median(right) if right.size > 0 else 0.5

        return center_val, left_val, right_val

    def get_control(self, center_depth, current_time):
        """根據深度計算控制指令"""
        if self.state == "FORWARD":
            if center_depth > MIDAS_CONFIG["OBSTACLE_THRESHOLD"]:
                self.state = "TURNING"
                self.turn_start_time = current_time
                self.obstacle_count += 1
                print(f"🚨 MidAS避障: 深度={center_depth:.3f}, 開始右轉")
        else:  # TURNING
            turn_elapsed = current_time - self.turn_start_time
            if turn_elapsed >= MIDAS_CONFIG["TURN_DURATION"]:
                if center_depth < MIDAS_CONFIG["CLEAR_THRESHOLD"]:
                    self.state = "FORWARD"
                    print(f"✅ MidAS: 前方安全，繼續前進")
                else:
                    self.turn_start_time = current_time

        if self.state == "FORWARD":
            return MIDAS_CONFIG["BASE_FORWARD"], 0
        else:
            return 0, MIDAS_CONFIG["TURN_SPEED"]

    def draw_overlay(self, frame, center_depth, fbv, yv):
        """繪製巡航模式畫面"""
        h, w = frame.shape[:2]

        if self.state == "TURNING":
            color = (0, 165, 255)
            status = "TURNING RIGHT"
        else:
            if center_depth > MIDAS_CONFIG["OBSTACLE_THRESHOLD"]:
                color = (0, 0, 255)
                status = "OBSTACLE!"
            elif center_depth > MIDAS_CONFIG["CLEAR_THRESHOLD"]:
                color = (0, 255, 255)
                status = "CAUTION"
            else:
                color = (0, 255, 0)
                status = "CLEAR"

        cv2.rectangle(frame, (w//3, h//3), (2*w//3, 2*h//3), color, 2)

        cv2.putText(frame, "MODE: MIDAS CRUISE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Status: {status}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Depth: {center_depth:.3f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Obstacles: {self.obstacle_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        if self.state == "TURNING":
            cv2.arrowedLine(frame, (w//2-50, h//2), (w//2+50, h//2),
                          (0, 165, 255), 3, tipLength=0.3)

        return frame

# ===================== 前進追蹤控制器 =====================
class ForwardTracker:
    """前進接近目標，不進行避障"""
    def __init__(self):
        self.model = YOLO("../model/box2.pt")
        self.has_target = False
        self.target_lost_time = None
        self.start_time = None
        self.last_bbox_area = 0

    def start(self):
        """啟動追蹤模式"""
        self.start_time = time.time()
        self.has_target = False
        self.target_lost_time = None
        print("🎯 開始前進追蹤模式")

    def process_frame(self, frame):
        """處理YOLO偵測，回傳控制指令"""
        results = self.model(frame, conf=0.6, verbose=False)

        yaw = 0
        up_down = 0
        forward = 0
        left_right = 0
        bbox = None
        bbox_area = 0
        target_reached = False

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_box = max(boxes, key=lambda b:
                (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2
            bbox_area = (x2 - x1) * (y2 - y1)

            self.last_bbox_area = bbox_area
            bbox = (x1, y1, x2, y2)

            # 檢查是否到達目標
            if bbox_area >= FORWARD_CONFIG["TARGET_AREA"]:
                target_reached = True

            # 計算控制
            error_x = bbox_cx - FRAME_W // 2
            error_y = bbox_cy - FRAME_H // 2
            error_area = FORWARD_CONFIG["TARGET_AREA"] - bbox_area

            # 偏航控制
            if abs(error_x) > FORWARD_CONFIG["DEADZONE"]:
                if abs(error_x) > 120:
                    left_right = self._clamp(int(FORWARD_CONFIG["KP_YAW"] * error_x),
                                 -FORWARD_CONFIG["MAX_SPEED"],
                                 FORWARD_CONFIG["MAX_SPEED"])
                else:
                    yaw = self._clamp(int(FORWARD_CONFIG["KP_YAW"] * error_x),
                                      -FORWARD_CONFIG["MAX_SPEED"],
                                      FORWARD_CONFIG["MAX_SPEED"])

            # 高度控制
            if abs(error_y) > FORWARD_CONFIG["DEADZONE"]:
                up_down = self._clamp(int(-FORWARD_CONFIG["KP_UPDOWN"] * error_y),
                                     -FORWARD_CONFIG["MAX_SPEED"],
                                     FORWARD_CONFIG["MAX_SPEED"])

            # 前進控制
            if abs(error_area) > FORWARD_CONFIG["AREA_TOLERANCE"]:
                forward = self._clamp(int(FORWARD_CONFIG["KP_FORWARD"] * error_area),
                                     -FORWARD_CONFIG["MAX_SPEED"],
                                     FORWARD_CONFIG["MAX_SPEED"])

            self.has_target = True
            self.target_lost_time = None

        else:
            # 目標丟失
            if self.has_target:
                if self.target_lost_time is None:
                    self.target_lost_time = time.time()
                    print(f"⚠️ 目標丟失，等待恢復...")
                self.has_target = False

        return left_right, forward, up_down, yaw, bbox, bbox_area, target_reached

    def should_abort(self):
        """檢查是否應該放棄追蹤（返回巡航，不降落）"""
        if not self.has_target and self.target_lost_time is not None:
            lost_duration = time.time() - self.target_lost_time
            if lost_duration > FORWARD_CONFIG["TARGET_LOST_TIMEOUT"]:
                print(f"⚠️ 目標丟失超過{FORWARD_CONFIG['TARGET_LOST_TIMEOUT']}秒，放棄追蹤")
                return True
        return False

    def is_timeout(self):
        """檢查是否超時（返回巡航，不降落）"""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > FORWARD_CONFIG["MAX_EXECUTION_TIME"]:
                print(f"⏰ 追蹤模式執行超過{elapsed:.0f}秒，返回巡航")
                return True
        return False

    def _clamp(self, val, minv, maxv):
        return max(minv, min(maxv, val))

# ===================== 環繞掃描控制器 =====================
class CircleScanner:
    """環繞目標並掃描條碼"""
    def __init__(self):
        self.model = YOLO("../model/box2.pt")
        self.start_time = None
        self.has_target = False
        self.target_lost_time = None
        self.scanned_set = set()
        self.scan_count = 0

        # 初始化CSV
        self.csv_file = CIRCLE_CONFIG["CSV_FILE"]
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Data"])

    def start(self):
        """啟動環繞模式"""
        self.start_time = time.time()
        self.has_target = False
        self.target_lost_time = None
        self.scan_count = 0
        print("🔄 開始環繞掃描模式")

    def process_frame(self, frame):
        """處理YOLO偵測和QR掃描"""
        results = self.model(frame, conf=0.6, verbose=False)

        yaw = 0
        up_down = 0
        forward = 0
        left_right = CIRCLE_CONFIG["ORBIT_SPEED"]
        bbox = None
        qr_detected = False

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_box = max(boxes, key=lambda b:
                (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2
            bbox_area = (x2 - x1) * (y2 - y1)
            bbox = (x1, y1, x2, y2)

            # 環繞時輕微修正位置
            error_x = bbox_cx - FRAME_W // 2
            error_y = bbox_cy - FRAME_H // 2
            error_area = FORWARD_CONFIG["TARGET_AREA"] - bbox_area

            if abs(error_x) > FORWARD_CONFIG["DEADZONE"]:
                yaw = self._clamp(int(FORWARD_CONFIG["KP_YAW"] * error_x),
                                 -FORWARD_CONFIG["MAX_SPEED"],
                                 FORWARD_CONFIG["MAX_SPEED"])

            if abs(error_y) > FORWARD_CONFIG["DEADZONE"]:
                up_down = self._clamp(int(-FORWARD_CONFIG["KP_UPDOWN"] * error_y),
                                     -FORWARD_CONFIG["MAX_SPEED"],
                                     FORWARD_CONFIG["MAX_SPEED"])

            if abs(error_area) > FORWARD_CONFIG["AREA_TOLERANCE"]:
                forward = self._clamp(int(FORWARD_CONFIG["KP_FORWARD"] * error_area),
                                     -FORWARD_CONFIG["MAX_SPEED"],
                                     FORWARD_CONFIG["MAX_SPEED"])

            self.has_target = True
            self.target_lost_time = None

            # QR Code掃描
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            barcodes = pyzbar.decode(gray)

            for barcode in barcodes:
                data = barcode.data.decode("utf-8")
                if data not in self.scanned_set:
                    self.scanned_set.add(data)
                    self.scan_count += 1

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(self.csv_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, data])

                    print(f"✅ 掃描到條碼: {data}")
                    qr_detected = True
        else:
            if self.has_target:
                if self.target_lost_time is None:
                    self.target_lost_time = time.time()
                    print(f"⚠️ 環繞中目標丟失，等待恢復...")
                self.has_target = False

        return left_right, forward, up_down, yaw, bbox, qr_detected

    def should_abort(self):
        """檢查是否應該放棄環繞（返回巡航，不降落）"""
        if not self.has_target and self.target_lost_time is not None:
            lost_duration = time.time() - self.target_lost_time
            if lost_duration > CIRCLE_CONFIG["TARGET_LOST_TIMEOUT"]:
                print(f"⚠️ 目標丟失超過{CIRCLE_CONFIG['TARGET_LOST_TIMEOUT']}秒，放棄環繞")
                return True
        return False

    def is_complete(self):
        """檢查環繞任務是否完成"""
        elapsed = time.time() - self.start_time
        # 至少掃描到1個條碼，且執行超過最小時間，或超過最大時間
        if self.scan_count > 0 and elapsed >= CIRCLE_CONFIG["MIN_SCAN_TIME"]:
            print(f"✅ 環繞掃描完成！掃描到{self.scan_count}個條碼")
            return True
        if elapsed >= CIRCLE_CONFIG["MAX_SCAN_TIME"]:
            print(f"⏰ 環繞超時，掃描到{self.scan_count}個條碼")
            return True
        return False

    def _clamp(self, val, minv, maxv):
        return max(minv, min(maxv, val))

# ===================== 主控制器 =====================
class TelloMissionController:
    """三階段任務控制器 - 永不自動降落"""
    def __init__(self):
        # 初始化Tello
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.tello.set_speed(50)

        print(f"電池電量: {self.tello.get_battery()}%")

        # 初始化各控制器
        self.midas = MidASCruiser()
        self.forward = ForwardTracker()
        self.circle = CircleScanner()

        # 狀態管理
        self.current_state = DroneState.MIDAS
        self.state_start_time = time.time()
        self.manual_mode = False
        self.running = True

        # 目標狀態
        self.target_found = False
        self.target_lost_time = None

        # 初始化pygame
        pygame.init()
        pygame.display.set_mode((300, 200))
        pygame.display.set_caption("Tello Mission Control")

    def get_keyboard_control(self):
        """讀取鍵盤控制"""
        lr = fb = ud = yv = 0
        manual_active = False
        quit_flag = False
        force_state_change = None
        takeoff_command = False
        land_command = False

        pygame.event.pump()
        keys = pygame.key.get_pressed()

        SPEED = YAW_SPEED = UD_SPEED = 50

        # 上升/下降
        if keys[pygame.K_w]:
            ud = UD_SPEED
            manual_active = True
        if keys[pygame.K_s]:
            ud = -UD_SPEED
            manual_active = True

        # 左轉/右轉
        if keys[pygame.K_a]:
            yv = -YAW_SPEED
            manual_active = True
        if keys[pygame.K_d]:
            yv = YAW_SPEED
            manual_active = True

        # 前進/後退
        if keys[pygame.K_UP]:
            fb = SPEED
            manual_active = True
        if keys[pygame.K_DOWN]:
            fb = -SPEED
            manual_active = True

        # 左右平移
        if keys[pygame.K_LEFT]:
            lr = -SPEED
            manual_active = True
        if keys[pygame.K_RIGHT]:
            lr = SPEED
            manual_active = True

        # 懸停
        if keys[pygame.K_SPACE]:
            lr = fb = ud = yv = 0
            manual_active = True

        # 起飛
        if keys[pygame.K_t]:
            takeoff_command = True

        # 降落
        if keys[pygame.K_l]:
            land_command = True

        # 強制狀態切換 (數字鍵)
        if keys[pygame.K_1]:
            force_state_change = DroneState.MIDAS
        if keys[pygame.K_2]:
            force_state_change = DroneState.FORWARD
        if keys[pygame.K_3]:
            force_state_change = DroneState.CIRCLE

        # ESC退出
        if keys[pygame.K_ESCAPE]:
            quit_flag = True

        return manual_active, lr, fb, ud, yv, quit_flag, force_state_change, takeoff_command, land_command

    def change_state(self, new_state):
        """切換狀態"""
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()

        # 狀態初始化
        if new_state == DroneState.FORWARD:
            self.forward.start()
        elif new_state == DroneState.CIRCLE:
            self.circle.start()

        print(f"\n🔄 狀態切換: {old_state} → {new_state}")

    def run(self):
        """主執行迴圈"""
        print("\n" + "="*50)
        print("Tello 三階段任務控制器啟動")
        print("狀態流程: MIDAS → FORWARD → CIRCLE → MIDAS")
        print("="*50)
        print("\n[控制鍵]")
        print("  T: 起飛")
        print("  L: 降落")
        print("  W/S: 上升/下降")
        print("  A/D: 左轉/右轉")
        print("  方向鍵: 前進/後退/左移/右移")
        print("  數字鍵1-3: 強制切換狀態")
        print("  ESC: 緊急停止")
        print("="*50)
        print("\n⚠️  注意：永不自動降落，請手動按 L 降落")
        print("="*50)

        frame_reader = self.tello.get_frame_read()
        last_control_time = time.time()
        frame_count = 0

        # 起飛提示
        print("\n🛸 請按 T 起飛")

        try:
            while self.running:
                # 讀取畫面
                frame = frame_reader.frame
                if frame is None:
                    time.sleep(0.05)
                    continue

                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                frame_count += 1

                # 處理鍵盤控制
                manual_active, lr, fb, ud, yv, quit_flag, force_state, takeoff_cmd, land_cmd = \
                    self.get_keyboard_control()

                if quit_flag:
                    print("使用者中斷程式")
                    break

                # 處理起飛/降落指令
                if takeoff_cmd:
                    print("🛸 手動起飛")
                    self.tello.takeoff()
                    time.sleep(1)

                if land_cmd:
                    print("🛬 手動降落")
                    self.tello.land()
                    time.sleep(1)

                if force_state:
                    self.change_state(force_state)

                # ============ 自動控制模式 ============
                if not manual_active:
                    control_cmd = [0, 0, 0, 0]

                    if self.current_state == DroneState.MIDAS:
                        # MidAS巡航避障
                        depth_norm, center_depth = self.midas.process_frame(frame)
                        fbv, yv = self.midas.get_control(center_depth, time.time())
                        control_cmd = [0, fbv, 0, yv]

                        # 繪製深度圖
                        depth_display = cv2.applyColorMap(
                            (depth_norm * 255).astype(np.uint8),
                            cv2.COLORMAP_JET
                        )
                        cv2.imshow("Depth Map", depth_display)

                        # YOLO偵測目標（不參與控制，只做切換判斷）
                        results = self.forward.model(frame, conf=0.6, verbose=False)
                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            best_box = max(boxes, key=lambda b:
                                (b.xyxy[0][2] - b.xyxy[0][0]) *
                                (b.xyxy[0][3] - b.xyxy[0][1]))
                            area = (best_box.xyxy[0][2] - best_box.xyxy[0][0]) * \
                                   (best_box.xyxy[0][3] - best_box.xyxy[0][1])

                            # 目標面積夠大才切換
                            if area > MIDAS_CONFIG["TARGET_FOUND_AREA"]:
                                print(f"🎯 巡航中找到目標! 面積={area:.0f}")
                                self.change_state(DroneState.FORWARD)

                        # 繪製巡航畫面
                        frame = self.midas.draw_overlay(frame, center_depth, fbv, yv)

                    elif self.current_state == DroneState.FORWARD:
                        # 前進追蹤目標
                        lr, fb, ud, yv, bbox, area, reached = \
                            self.forward.process_frame(frame)
                        control_cmd = [lr, fb, ud, yv]

                        # 繪製
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Area: {area}", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        cv2.putText(frame, "MODE: FORWARD TRACK", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # 檢查任務完成（到達目標）
                        if reached:
                            print(f"🎉 到達目標! 面積={area}")
                            # 稍微停頓一下
                            self.tello.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            self.change_state(DroneState.CIRCLE)

                        # 檢查是否放棄追蹤（目標丟失太久或超時）
                        elif self.forward.should_abort() or self.forward.is_timeout():
                            print("↩️ 返回巡航模式")
                            self.change_state(DroneState.MIDAS)

                    elif self.current_state == DroneState.CIRCLE:
                        # 環繞掃描
                        lr, fb, ud, yv, bbox, qr_found = \
                            self.circle.process_frame(frame)
                        control_cmd = [lr, fb, ud, yv]

                        # 繪製
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        cv2.putText(frame, "MODE: CIRCLE SCAN", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"Scanned: {self.circle.scan_count}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        # 檢查任務完成（掃描足夠或超時）
                        if self.circle.is_complete():
                            print("⬆️ 環繞完成，拉高高度返回巡航")
                            # 拉高30cm確保安全
                            self.tello.send_rc_control(0, 0, 30, 0)
                            time.sleep(1)
                            self.tello.send_rc_control(0, 0, 0, 0)
                            self.change_state(DroneState.MIDAS)

                        # 檢查是否放棄環繞（目標丟失太久）
                        elif self.circle.should_abort():
                            print("↩️ 目標丟失，返回巡航模式")
                            self.change_state(DroneState.MIDAS)

                    # 發送控制指令
                    current_time = time.time()
                    if current_time - last_control_time >= CONTROL_INTERVAL:
                        self.tello.send_rc_control(*control_cmd)
                        last_control_time = current_time

                # ============ 手動模式 ============
                else:
                    self.tello.send_rc_control(lr, fb, ud, yv)
                    cv2.putText(frame, "MANUAL MODE", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 顯示狀態資訊
                cv2.putText(frame, f"State: {self.current_state}", (10, FRAME_H-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Battery: {self.tello.get_battery()}%",
                           (FRAME_W-150, FRAME_H-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 顯示控制提示
                cv2.putText(frame, "T:Takeoff L:Land", (10, FRAME_H-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Tello Mission Control", frame)

                if cv2.waitKey(1) == 27:  # ESC
                    break

        except Exception as e:
            print(f"錯誤: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        """清理資源"""
        print("\n🧹 清理資源中...")
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        print("⚠️  請記得手動降落")
        self.tello.streamoff()
        pygame.quit()
        cv2.destroyAllWindows()
        print("✅ 程式結束")

# ===================== 程式入口 =====================
if __name__ == "__main__":
    controller = TelloMissionController()
    controller.run()