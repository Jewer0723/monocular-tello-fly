"""
main_fly10.py — Tello 四階段任務控制系統
狀態機: MIDAS → FORWARD → CIRCLE → QR_SCAN → MIDAS

架構：
  - 移除 ORB-SLAM3 / RViz / FlightTracker / PointCloudVisualizer
  - 新增 pan+tilt 雲台追蹤系統（ESP32 串口控制）
  - 返航流程：ReturnHomeController（DR 飛回）
             → GimbalTracker（角度重播 + YOLO 搜尋）
             → 視覺引導對中降落
"""
import csv
import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import pygame
import torch
from djitellopy import Tello
from pyzbar import pyzbar
from ultralytics import YOLO

# ===================== 全域設定 =====================
FRAME_W, FRAME_H = 640, 480
DRONE_FRAME_W, DRONE_FRAME_H = 640, 320
CONTROL_INTERVAL = 0.05

# ── 模型路徑 ──────────────────────────────────────────────────────
MODEL_DIR          = "../model"
box_model_path     = f"{MODEL_DIR}/box2.pt"
barcode_model_path = f"{MODEL_DIR}/barcode1.pt"
DRONE_MODEL_PATH   = f"{MODEL_DIR}/drone1.pt"  # 無人機偵測模型（雲台 + webcam 降落用）
box_conf = 0.7
qr_conf  = 0.7
drone_conf = 0.4

# ===================== 飛行狀態 =====================
class DroneState:
    MIDAS          = "MIDAS"           # 巡航避障模式
    FORWARD        = "FORWARD"         # 前進接近目標模式
    CIRCLE         = "CIRCLE"          # 環繞掃描模式（偵測條碼）
    QR_SCAN        = "QR_SCAN"         # QR Code掃描模式（鎖定靠近）
    RETURN_HOME    = "RETURN_HOME"     # 低電量直線返航降落

# ===================== MidAS 巡航參數 =====================
MIDAS_CONFIG = {
    # 隨機高度變化（MIDAS 巡航時使用）
    "ALT_CHANGE_INTERVAL": 3.0,
    "ALT_SPEED":            12,
    "ALT_MIN_CM":           50,
    "ALT_MAX_CM":          180,
    "BASE_FORWARD":      20,
    "TURN_SPEED":        40,
    "OBSTACLE_THRESHOLD": 0.35,
    "CLEAR_THRESHOLD":   0.25,
    "TURN_DURATION":     1.5,
    "SMOOTHING_WINDOW":  5,
    "TARGET_FOUND_AREA": 10000,
}

# ===================== 前進追蹤參數 =====================
FORWARD_CONFIG = {
    "TARGET_AREA":        100000,
    "AREA_TOLERANCE":     15000,
    "KP_YAW":             0.3,
    "KP_UPDOWN":          0.3,
    "KP_FORWARD":         0.0006,
    "MAX_SPEED":          20,
    "DEADZONE":           20,
    "MIN_AREA":           10000,
    "TARGET_LOST_TIMEOUT": 1,
    "MAX_EXECUTION_TIME":  30,
}

# ===================== 環繞掃描參數 =====================
CIRCLE_CONFIG = {
    "ORBIT_SPEED":           7,   # 原 7 → 5，降低環繞速度
    "YAW_CORRECTION_SPEED": 25,   # 原 15 → 25，提高 yaw 修正上限
    "HEIGHT_CORRECTION_SPEED": 15,
    "MIN_CIRCLE_TIME":        5,
    "MAX_CIRCLE_TIME":       30,
    "TARGET_LOST_TIMEOUT":    1,  # 原 1 → 2，容許短暫丟失
    "TARGET_AREA":       120000,
    "AREA_TOLERANCE":      5000,
    "KP_FORWARD":         0.0006,
    "MAX_EXECUTION_TIME":    30,
}

# ===================== QR掃描參數 =====================
QR_SCAN_CONFIG = {
    "TARGET_AREA":          60000,  # 原 20000 → 60000，確保靠得夠近
    "AREA_TOLERANCE":        5000,
    "KP_YAW":               0.25,
    "KP_UPDOWN":            0.25,
    "KP_FORWARD":          0.0006,
    "MAX_SPEED":             15,
    "DEADZONE":              15,
    "TARGET_LOST_TIMEOUT":    3,
    "MAX_EXECUTION_TIME":    30,
    "QR_SCAN_INTERVAL":      0.3,
    "FORWARD_WHEN_NO_DECODE": True,
    "MIN_AREA_BEFORE_DECODE": 40000,  # 原 1000 → 40000
    "CSV_FILE":              "scanned_codes.csv"
}

# ===================== USB 鏡頭參數 =====================
WEBCAM_CONFIG = {
    "HANDOFF_DIST_CM"  : 200,    # DR 估算距起飛點 < 200cm 時啟用
    "LAND_AREA_THRESH" : 100000  # 降落面積閾值，實測後調整
}

# ===================== 雲台參數 =====================
GIMBAL_CONFIG = {
    "SERIAL_PORT":       "COM3",
    "SERIAL_BAUD":       115200,
    "PAN_CENTER":        90,
    "TILT_CENTER":       90,
    "PAN_MIN":           0,
    "PAN_MAX":           180,
    "TILT_MIN":          50,
    "TILT_MAX":          130,
    "KP_PAN":            0.04,
    "KP_TILT":           0.03,
    "DEADZONE_PX":       30,
    "MAX_HISTORY":       3000,  # 最多記錄幀數
    "RECORD_INTERVAL":   0.10,  # 每幾秒記錄一次
    "REPLAY_SPEED":      1.5,    # 重播倍率（> 1 超前預測）
    "REPLAY_INTERVAL":   0.08,   # 重播每幀間隔秒數
    "DETECT_MIN_AREA":   400,
    "LAND_AREA_THRESH":  100000,  # 降落面積閾值，實測後調整
    "FB_APPROACH":       18,     # RTH 接近速度（由 ReturnHomeController 使用）
    "KP_LR":             0.06,   # 視覺引導水平增益
    "KP_UD":             0.05,   # 視覺引導垂直增益
    "MAX_LR":            30,     # 左右最大速度
    "MAX_UD":            20,     # 上下最大速度
    "LOST_HOVER_SEC":    8.0,    # 目標遺失後懸停等待秒數
}

# ===================== 返航參數 =====================
LOW_BATTERY_CONFIG = {
    "THRESHOLD":          70,   # 低於此電量(%)觸發回航
    "CHECK_INTERVAL":   5,     # 每幾秒查一次電量
    "RETURN_SPEED":       20,   # 回航飛行速度
    "YAW_KP":            0.8,   # 偏航修正係數
    "WAYPOINT_RADIUS":    60,   # 中繼航點到達容忍半徑(cm)
    "LAND_RADIUS":        5,   # 起飛點降落容忍半徑(cm)
    "ARRIVE_CM":          5,     # 距起飛點此距離(cm)視為到達
    "HOVER_SEC":         2.0,    # 到達後懸停秒數
    "SPEED":             20,     # 返航飛行速度 (cm/s RC 值)
    "DESCEND_SPD":       -8,     # 接近地面時下降速度
    "TARGET_H_CM":        60,     # 開始緩降的高度門檻
}


# ===================== MIDAS 巡航避障控制器 =====================
class MidASCruiser:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("MidAS using device:", self.device)

        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

        self.center_queue = deque(maxlen=MIDAS_CONFIG["SMOOTHING_WINDOW"])
        self.left_queue   = deque(maxlen=MIDAS_CONFIG["SMOOTHING_WINDOW"])
        self.right_queue  = deque(maxlen=MIDAS_CONFIG["SMOOTHING_WINDOW"])

        self.state           = "FORWARD"
        self.turn_start_time = 0
        self.obstacle_count  = 0

    def process_frame(self, frame):
        img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth      = prediction.cpu().numpy()
        depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

        center_val, left_val, right_val = self._get_depth_regions(depth_norm)
        self.center_queue.append(center_val)
        self.left_queue.append(left_val)
        self.right_queue.append(right_val)

        center_avg = np.mean(self.center_queue)
        return depth_norm, center_avg

    def _get_depth_regions(self, depth_map):
        h, w          = depth_map.shape
        center_height = h // 3
        center_width  = w // 3
        center_top    = h // 2 - center_height // 2
        center_left   = w // 2 - center_width  // 2

        center = depth_map[center_top:center_top + center_height,
                           center_left:center_left + center_width]
        left   = depth_map[center_top:center_top + center_height, :w // 4]
        right  = depth_map[center_top:center_top + center_height, 3 * w // 4:]

        center_val = np.median(center) if center.size > 0 else 0.5
        left_val   = np.median(left)   if left.size   > 0 else 0.5
        right_val  = np.median(right)  if right.size  > 0 else 0.5

        return center_val, left_val, right_val

    def get_control(self, center_depth, current_time):
        if self.state == "FORWARD":
            if center_depth > MIDAS_CONFIG["OBSTACLE_THRESHOLD"]:
                self.state           = "TURNING"
                self.turn_start_time = current_time
                self.obstacle_count += 1
                print(f"🚨 MidAS避障: 深度={center_depth:.3f}, 開始右轉")
        else:
            turn_elapsed = current_time - self.turn_start_time
            if turn_elapsed >= MIDAS_CONFIG["TURN_DURATION"]:
                if center_depth < MIDAS_CONFIG["CLEAR_THRESHOLD"]:
                    self.state = "FORWARD"
                    print("✅ MidAS: 前方安全，繼續前進")
                else:
                    self.turn_start_time = current_time

        if self.state == "FORWARD":
            return MIDAS_CONFIG["BASE_FORWARD"], 0
        else:
            return 0, MIDAS_CONFIG["TURN_SPEED"]

    def draw_overlay(self, frame, center_depth, fbv, yv):
        h, w = frame.shape[:2]

        if self.state == "TURNING":
            color  = (0, 165, 255)
            status = "TURNING RIGHT"
        else:
            if center_depth > MIDAS_CONFIG["OBSTACLE_THRESHOLD"]:
                color  = (0, 0, 255)
                status = "OBSTACLE!"
            elif center_depth > MIDAS_CONFIG["CLEAR_THRESHOLD"]:
                color  = (0, 255, 255)
                status = "CAUTION"
            else:
                color  = (0, 255, 0)
                status = "CLEAR"

        cv2.rectangle(frame, (w//3, h//3), (2*w//3, 2*h//3), color, 2)
        cv2.putText(frame, "MODE: MIDAS CRUISE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Status: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.state == "TURNING":
            cv2.arrowedLine(frame, (w//2-50, h//2), (w//2+50, h//2),
                            (0, 165, 255), 3, tipLength=0.3)

        return frame

# ===================== 目標追蹤基底類別 =====================
class TargetTracker:
    def __init__(self, model_path, config):
        self.model  = YOLO(model_path)
        self.config = config

        self.has_target           = False
        self.target_lost_time     = None
        self.start_time           = None
        self.last_bbox            = None
        self.last_bbox_area       = 0
        self.target_center_history = deque(maxlen=5)

    def start(self):
        self.start_time       = time.time()
        self.has_target       = False
        # Bug Fix: 初始化時就記錄「丟失時間」為 start，
        # 否則從未偵測到目標時 target_lost_time 永遠是 None，
        # should_abort() 的條件永遠不成立，超時無法觸發。
        self.target_lost_time = time.time()
        self.target_center_history.clear()

    # ------------------------------------------------------------------
    # 加入後處理過濾：長寬比、畫面佔比、最小面積
    # ------------------------------------------------------------------
    def _is_valid_box(self, x1, y1, x2, y2):
        """過濾掉牆壁、櫃子等非紙箱物體"""
        w   = x2 - x1
        h   = y2 - y1
        area       = w * h
        aspect     = w / (h + 1e-5)
        area_ratio = area / (FRAME_W * FRAME_H)

        # 長寬比需在 0.3~3.0 之間（排除極扁長條）
        if not (0.3 < aspect < 3.0):
            return False
        # 不能佔畫面 70% 以上（排除牆壁/背景）
        if area_ratio > 0.70:
            return False
        # 不能太小
        if area < 8000:
            return False
        return True

    def detect_target(self, frame, conf=box_conf):  # conf 預設提高至 0.75
        results = self.model(frame, conf=conf, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # 先過濾無效框
            valid_boxes = []
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                if self._is_valid_box(x1, y1, x2, y2):
                    valid_boxes.append(b)

            if not valid_boxes:
                # Bug Fix: 同主 else 路徑，移除 has_target 前置判斷
                if self.target_lost_time is None:
                    self.target_lost_time = time.time()
                    print("⚠️ 目標丟失（過濾後無有效框），等待恢復...")
                self.has_target = False
                return False, 0, 0, 0, None

            best_box = max(valid_boxes, key=lambda b:
                (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            bbox_cx   = (x1 + x2) // 2
            bbox_cy   = (y1 + y2) // 2
            bbox_area = (x2 - x1) * (y2 - y1)

            self.target_center_history.append((bbox_cx, bbox_cy))
            avg_cx = int(np.mean([c[0] for c in self.target_center_history]))
            avg_cy = int(np.mean([c[1] for c in self.target_center_history]))

            self.last_bbox      = (x1, y1, x2, y2)
            self.last_bbox_area = bbox_area
            self.has_target     = True
            self.target_lost_time = None

            return True, avg_cx, avg_cy, bbox_area, (x1, y1, x2, y2)
        else:
            # Bug Fix: 移除 has_target 前置判斷。
            # 原本「第一次就偵測不到目標」時 has_target=False，
            # 所以 target_lost_time 永遠不會被寫入，should_abort() 永遠 False。
            if self.target_lost_time is None:
                self.target_lost_time = time.time()
                print("⚠️ 目標丟失，等待恢復...")
            self.has_target = False
            return False, 0, 0, 0, None

    def calculate_control(self, target_cx, target_cy, target_area, target_area_goal):
        error_x    = target_cx   - FRAME_W // 2
        error_y    = target_cy   - FRAME_H // 2
        error_area = target_area_goal - target_area

        yaw        = 0
        up_down    = 0
        forward    = 0
        left_right = 0

        if abs(error_x) > self.config["DEADZONE"]:
            if abs(error_x) > 120:
                left_right = self._clamp(
                    int(self.config["KP_YAW"] * error_x),
                    -self.config["MAX_SPEED"],
                    self.config["MAX_SPEED"]
                )
            else:
                yaw = self._clamp(
                    int(self.config["KP_YAW"] * error_x),
                    -self.config["MAX_SPEED"],
                    self.config["MAX_SPEED"]
                )

        if abs(error_y) > self.config["DEADZONE"]:
            up_down = self._clamp(
                int(-self.config["KP_UPDOWN"] * error_y),
                -self.config["MAX_SPEED"],
                self.config["MAX_SPEED"]
            )

        if abs(error_area) > self.config["AREA_TOLERANCE"]:
            forward = self._clamp(
                int(self.config["KP_FORWARD"] * error_area),
                -self.config["MAX_SPEED"],
                self.config["MAX_SPEED"]
            )

        return left_right, forward, up_down, yaw

    def should_abort(self):
        if not self.has_target and self.target_lost_time is not None:
            lost_duration = time.time() - self.target_lost_time
            if lost_duration > self.config["TARGET_LOST_TIMEOUT"]:
                return True
        return False

    def is_timeout(self):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.config["MAX_EXECUTION_TIME"]:
                return True
        return False

    def _clamp(self, val, minv, maxv):
        return max(minv, min(maxv, val))

# ===================== 前進追蹤控制器 =====================
class ForwardTracker(TargetTracker):
    def __init__(self):
        super().__init__(box_model_path, FORWARD_CONFIG)

    def process_frame(self, frame):
        detected, cx, cy, area, bbox = self.detect_target(frame)

        if detected:
            lr, fb, ud, yaw = self.calculate_control(cx, cy, area, self.config["TARGET_AREA"])
            reached = area >= self.config["TARGET_AREA"]
            return lr, fb, ud, yaw, bbox, area, reached
        else:
            return 0, 0, 0, 0, None, 0, False

# ===================== 環繞掃描控制器 =====================
class CircleScanner(TargetTracker):
    """環繞目標，同時偵測條碼，並保持固定距離"""

    def __init__(self):
        super().__init__(box_model_path, CIRCLE_CONFIG)
        self.qr_model        = YOLO(barcode_model_path)
        self.scanned_set     = set()
        self.orbit_direction = 1
        self.smooth_center   = deque(maxlen=3)

    def start(self):
        super().start()
        self.smooth_center.clear()
        print("🔄 開始環繞掃描模式")

    def process_frame(self, frame):
        """處理環繞控制和QR偵測"""
        detected, cx, cy, area, bbox = self.detect_target(frame)
        qr_detected = False
        qr_bbox     = None

        # 預設控制值
        left_right = CIRCLE_CONFIG["ORBIT_SPEED"]
        forward    = 0
        up_down    = 0
        yaw        = 0

        if detected:
            self.smooth_center.append((cx, cy))
            avg_cx = int(np.mean([c[0] for c in self.smooth_center]))
            avg_cy = int(np.mean([c[1] for c in self.smooth_center]))

            error_x    = avg_cx - FRAME_W // 2
            error_y    = avg_cy - FRAME_H // 2
            error_area = CIRCLE_CONFIG["TARGET_AREA"] - area

            # ----------------------------------------------------------
            # yaw 修正：移除縮小係數，直接全力修正
            #         誤差大時暫停環繞讓 yaw 先追上目標
            # ----------------------------------------------------------
            if abs(error_x) > 120:
                # 誤差太大 → 暫停側移，集中修正 yaw
                left_right = 0
                yaw = self._clamp(
                    int(FORWARD_CONFIG["KP_YAW"] * error_x),
                    -CIRCLE_CONFIG["YAW_CORRECTION_SPEED"],
                    CIRCLE_CONFIG["YAW_CORRECTION_SPEED"]
                )
            else:
                # 誤差在可接受範圍 → 繼續環繞，同時小幅修正 yaw
                left_right = CIRCLE_CONFIG["ORBIT_SPEED"]
                if abs(error_x) > FORWARD_CONFIG["DEADZONE"]:
                    yaw = self._clamp(
                        int(FORWARD_CONFIG["KP_YAW"] * error_x),  # 移除 *0.3
                        -CIRCLE_CONFIG["YAW_CORRECTION_SPEED"],
                        CIRCLE_CONFIG["YAW_CORRECTION_SPEED"]
                    )

            # 高度修正
            if abs(error_y) > FORWARD_CONFIG["DEADZONE"]:
                up_down = self._clamp(
                    int(-FORWARD_CONFIG["KP_UPDOWN"] * error_y * 0.5),
                    -CIRCLE_CONFIG["HEIGHT_CORRECTION_SPEED"],
                    CIRCLE_CONFIG["HEIGHT_CORRECTION_SPEED"]
                )

            # 保持固定距離
            if abs(error_area) > CIRCLE_CONFIG["AREA_TOLERANCE"]:
                forward = self._clamp(
                    int(CIRCLE_CONFIG["KP_FORWARD"] * error_area),
                    -FORWARD_CONFIG["MAX_SPEED"],
                    FORWARD_CONFIG["MAX_SPEED"]
                )

            # 偵測 QR Code
            qr_detected, qr_bbox = self.detect_qr_code(frame, bbox)

        return left_right, forward, up_down, yaw, bbox, qr_detected, qr_bbox

    def detect_qr_code(self, frame, target_bbox):
        """偵測 QR Code 位置"""
        if target_bbox is None:
            return False, None

        x1, y1, x2, y2 = target_bbox

        roi_x1 = max(0, x1 - 50)
        roi_y1 = max(0, y1 - 50)
        roi_x2 = min(FRAME_W, x2 + 50)
        roi_y2 = min(FRAME_H, y2 + 50)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return False, None

        results = self.qr_model(roi, conf=qr_conf, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes    = results[0].boxes
            best_box = max(boxes, key=lambda b:
                (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))

            qx1, qy1, qx2, qy2 = map(int, best_box.xyxy[0])

            # 轉換回原始座標
            qx1 += roi_x1
            qy1 += roi_y1
            qx2 += roi_x1
            qy2 += roi_y1

            print("🔍 偵測到QR Code位置")
            return True, (qx1, qy1, qx2, qy2)

        return False, None

    def is_complete(self):
        """檢查環繞是否完成（最少時間）"""
        elapsed = time.time() - self.start_time
        return elapsed >= CIRCLE_CONFIG["MIN_CIRCLE_TIME"]

    def should_abort(self):
        # Bug Fix: 直接用基類邏輯，避免與基類的 target_lost_time 不同步。
        return super().should_abort()

# ===================== QR 掃描控制器 =====================
class QRScanner(TargetTracker):
    """專門鎖定並掃描QR Code，無法解碼時持續前進"""

    def __init__(self):
        super().__init__(barcode_model_path, QR_SCAN_CONFIG)
        self.scanned_set          = set()
        self.scan_count           = 0
        self.last_scan_time       = 0
        self.scan_complete        = False
        self.qr_lost_time         = None
        self.scanned_data         = None
        self.consecutive_failures = 0
        self.csv_file             = QR_SCAN_CONFIG["CSV_FILE"]

        self.load_scanned_data()

    def load_scanned_data(self):
        """載入CSV中已有的掃描資料"""
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            self.scanned_set.add(row[1])
                print(f"📚 已載入 {len(self.scanned_set)} 筆歷史掃描資料")
            except Exception as e:
                print(f"⚠️ 載入歷史資料時出錯: {e}")

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["時間", "資料"])

    def start(self, qr_bbox=None):
        """啟動QR掃描模式，可指定初始QR位置"""
        super().start()   # 已在基類 start() 設定 target_lost_time = now
        self.scan_complete        = False
        self.scanned_data         = None
        self.qr_lost_time         = time.time()  # Bug Fix: 與基類對齊，從啟動就開始計時
        self.last_scan_time       = 0
        self.consecutive_failures = 0

        if qr_bbox:
            cx   = (qr_bbox[0] + qr_bbox[2]) // 2
            cy   = (qr_bbox[1] + qr_bbox[3]) // 2
            area = (qr_bbox[2] - qr_bbox[0]) * (qr_bbox[3] - qr_bbox[1])
            self.target_center_history.append((cx, cy))
            self.last_bbox      = qr_bbox
            self.last_bbox_area = area
            self.has_target     = True

        print("📸 開始QR Code掃描模式")

    def process_frame(self, frame):
        """處理QR Code追蹤和掃描"""
        detected, cx, cy, area, bbox = self.detect_target(frame, conf=box_conf)

        qr_decoded   = False
        decoded_data = None

        if detected:
            self.qr_lost_time = None
            lr, fb, ud, yaw   = self.calculate_control(cx, cy, area, self.config["TARGET_AREA"])

            if self.config["FORWARD_WHEN_NO_DECODE"] and not self.scan_complete:
                if area < self.config["MIN_AREA_BEFORE_DECODE"]:
                    fb = self.config["MAX_SPEED"]
                    print(f"📏 持續前進中... 目前面積={area:.0f}, 目標={self.config['MIN_AREA_BEFORE_DECODE']}")

            reached      = area >= self.config["TARGET_AREA"]
            current_time = time.time()

            if current_time - self.last_scan_time > self.config["QR_SCAN_INTERVAL"]:
                decoded, data = self.decode_qr_code(frame, bbox)

                if decoded:
                    if data in self.scanned_set:
                        print(f"⚠️ 條碼已掃描過: {data}，立即返回巡航")
                        self.scan_count += 1
                        self.scanned_data = data
                        self.scan_complete = True
                        qr_decoded = True
                        decoded_data = data
                        self.consecutive_failures = 0
                    else:
                        self.scanned_set.add(data)
                        self.scan_count  += 1
                        self.scanned_data = data
                        self.scan_complete = True
                        qr_decoded        = True
                        decoded_data      = data
                        self.consecutive_failures = 0

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(self.csv_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, data])

                        print(f"✅ 成功掃描新條碼: {data}")
                else:
                    self.consecutive_failures += 1
                    if self.consecutive_failures % 5 == 0:
                        print(f"📸 嘗試解碼中... (第{self.consecutive_failures}次失敗)")

                self.last_scan_time = current_time

            return lr, fb, ud, yaw, bbox, area, reached, qr_decoded, decoded_data
        else:
            # Bug Fix: target_lost_time 由基類 detect_target 統一管理，
            # 這裡同步更新 qr_lost_time 供 log 用即可，不重複設邏輯。
            if self.qr_lost_time is None:
                self.qr_lost_time = time.time()
                print("⚠️ QR目標丟失，等待恢復...")
            return 0, 0, 0, 0, None, 0, False, False, None

    # ------------------------------------------------------------------
    # 強化解碼流程：ROI 放大 + 多種預處理 + 全帧兜底
    # ------------------------------------------------------------------
    def decode_qr_code(self, frame, qr_bbox):
        """在QR Code區域內解碼"""
        if qr_bbox is None:
            return False, None

        x1, y1, x2, y2 = qr_bbox

        pad    = 40                              # 擴大 padding（原 20）
        roi_x1 = max(0, x1 - pad)
        roi_y1 = max(0, y1 - pad)
        roi_x2 = min(FRAME_W, x2 + pad)
        roi_y2 = min(FRAME_H, y2 + pad)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return False, None

        # 強制放大 ROI：pyzbar 對小圖解碼率極差
        roi_h, roi_w = roi.shape[:2]
        min_dim      = min(roi_h, roi_w)
        if min_dim < 150:
            scale = max(2, int(300 / (min_dim + 1e-5)))
            roi   = cv2.resize(roi, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 擴充預處理方法
        sharpening_kernel = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        methods = [
            gray,                                                               # 原始灰階
            cv2.GaussianBlur(gray, (3, 3), 0),                                  # 高斯模糊
            cv2.equalizeHist(gray),                                             # 直方圖均衡化
            cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2),                   # 自適應二值化
            cv2.filter2D(gray, -1, sharpening_kernel),                          # 銳化
            cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],             # OTSU
            cv2.bitwise_not(gray),                                              # 反色
            cv2.bitwise_not(
                cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),        # 反色OTSU
        ]

        for method in methods:
            barcodes = pyzbar.decode(method)
            if barcodes:
                return True, barcodes[0].data.decode("utf-8")

        # 最後兜底：對整張 frame 解碼（不限 ROI）
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        barcodes  = pyzbar.decode(gray_full)
        if barcodes:
            return True, barcodes[0].data.decode("utf-8")

        return False, None

    def is_complete(self):
        if self.scan_complete:
            return True
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.config["MAX_EXECUTION_TIME"]:
                print(f"⏰ QR掃描超時 ({elapsed:.0f}秒)")
                return True
        return False

    def should_abort(self):
        # Bug Fix: 原本 scanned_data 也會短路 should_abort，
        # 導致即使目標丟失也無法退出。現在只有真正掃描完成才短路。
        if self.scan_complete:
            return False
        # 走基類的 target_lost_time 判斷
        return super().should_abort()




# ===================== 返航控制器（DR 飛回起飛點）=====================




# ===================== 飛行位置追蹤器（Dead Reckoning）=====================
class FlightTracker:
    """
    速度積分位置追蹤。
    只保留位置計算，供 ReturnHomeController 和 GimbalTracker 使用。
    座標系：x=左右, y=高度, z=前後（起飛點為原點）
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = self.y = self.z = 0.0
        self.yaw  = 0.0
        self.home = (0.0, 0.0, 0.0)
        self.path: list = [(0.0, 0.0, 0.0, False)]
        self.last_time = time.time()

    def reset_pose(self):
        """起飛後呼叫：清位移/軌跡，保留 home"""
        self.x = self.y = self.z = 0.0
        self.yaw = 0.0
        self.path = [(0.0, 0.0, 0.0, False)]
        self.last_time = time.time()

    def update(self, tello, is_manual: bool = False):
        now = time.time()
        dt  = now - self.last_time
        self.last_time = now
        if dt <= 0 or dt > 1.0:
            return
        try:
            vx  = float(tello.get_speed_x())
            vy  = float(tello.get_speed_y())
            vz  = float(tello.get_speed_z())
            yaw = float(tello.get_yaw())
        except Exception:
            return
        self.yaw  = yaw
        self.z   += (-vx) * dt
        self.x   += (-vy) * dt
        self.y   +=   vz  * dt
        self.path.append((self.x, self.y, self.z, is_manual))


# ===================== Webcam 靜態視覺降落（固定鏡頭備援）=====================
class WebcamLanding:
    """
    固定 webcam（正對起飛點）靜態視覺引導降落。
    作為 GimbalTracker 的備援：當雲台無法使用時啟用。

    流程：
      APPROACH : YOLO 偵測 Tello，框框面積 < LAND_AREA_THRESH → 前進 + 對中
      LAND     : 框框面積 >= LAND_AREA_THRESH → 降落
    """
    def __init__(self, tello, tracker, cam_index: int = 1):
        self.cfg = WEBCAM_CONFIG
        self.tello    = tello
        self.tracker  = tracker
        self._cam_idx = cam_index
        self._cap     = None
        self._active  = False
        self._phase   = "idle"
        self._lost_t  = 0.0
        self._model   = YOLO(DRONE_MODEL_PATH)
        self.last_frame = None
        self.last_bbox  = None
        self.last_area  = 0

    def should_handoff(self) -> bool:
        dx   = self.tracker.home[0] - self.tracker.x
        dz   = self.tracker.home[2] - self.tracker.z
        return math.sqrt(dx**2 + dz**2) < self.cfg["HANDOFF_DIST_CM"]

    def start(self):
        if self._active:
            return
        self._cap = cv2.VideoCapture(self._cam_idx)
        if not self._cap.isOpened():
            print(f"[WebcamLanding] ❌ webcam index={self._cam_idx} 無法開啟")
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DRONE_FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  DRONE_FRAME_H)
        self._active = True
        self._phase  = "approach"
        self._lost_t = 0.0
        print("[WebcamLanding] ✅ 啟動")

    def stop(self):
        self._active = False
        self._phase  = "idle"
        if self._cap:
            self._cap.release()
            self._cap = None

    def is_active(self) -> bool:
        return self._active

    def is_landing(self) -> bool:
        return self._phase == "land"

    def get_rc(self) -> list:
        if not self._active or self._cap is None:
            return [0, 0, 0, 0]
        ret, frame = self._cap.read()
        if not ret:
            return [0, 0, 0, 0]
        self.last_frame = frame.copy()
        h, w = frame.shape[:2]
        bbox = self._detect(frame)
        self.last_bbox = bbox
        if bbox is None:
            self.last_area = 0
            if self._lost_t == 0.0:
                self._lost_t = time.time()
            elif time.time() - self._lost_t > 2.0:
                self.stop()
            return [0, 0, 0, 0]
        self._lost_t = 0.0
        x1, y1, x2, y2 = bbox
        area = (x2-x1) * (y2-y1)
        self.last_area = area
        if area >= self.cfg["LAND_AREA_THRESH"]:
            self._phase = "land"
            self.tello.land()
            self.stop()
            return [0, 0, 0, 0]
        ex = (x1+x2)//2 - w//2
        ey = (y1+y2)//2 - h//2
        lr = int(0.06 * ex) if abs(ex) > 40 else 0
        ud = int(-0.05 * ey) if abs(ey) > 30 else 0
        lr = max(-30, min(30, lr))
        ud = max(-20, min(20, ud))
        return [lr, 18, ud, 0]

    def draw_hud(self, main_frame):
        if self.last_frame is None:
            return main_frame
        thumb = cv2.resize(self.last_frame, (320, 180))
        if self.last_bbox is not None:
            x1, y1, x2, y2 = self.last_bbox
            sx = 320 / self.last_frame.shape[1]
            sy = 180 / self.last_frame.shape[0]
            cv2.rectangle(thumb, (int(x1*sx), int(y1*sy)),
                          (int(x2*sx), int(y2*sy)), (0,255,0), 2)
            cv2.putText(thumb, f"area:{self.last_area}",
                (5, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        fh, fw = main_frame.shape[:2]
        main_frame[fh-190:fh-10, 10:330] = thumb
        cv2.rectangle(main_frame, (10, fh-190), (330, fh-10), (0,200,0), 1)
        cv2.putText(main_frame, "WEBCAM LAND", (10, fh-195),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,100), 2)
        return main_frame


    def _detect(self, frame) -> Optional[tuple]:
        if self._model is None:
            return None
        try:
            results = self._model(frame, verbose=False, device=0)[0]
            best_box, best_area = None, 400
            for box in results.boxes:
                if float(box.conf) < drone_conf:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                area = (x2-x1)*(y2-y1)
                if area > best_area:
                    best_area, best_box = area, (x1,y1,x2,y2)
            return best_box
        except Exception as e:
            print(f"[WebcamLanding] detect: {e}")
            return None


class GimbalTracker:
    """
    pan+tilt 雲台追蹤控制器（ESP32 串口）。

    生命週期：
      程式啟動 → __init__() 載入模型 + 開 webcam + 開預覽視窗
      每幀     → update_tracking() 偵測無人機、顯示 UI、起飛後送角度給雲台
      起飛後   → start_tracking() 開串口、設 _tracking=True
      進入返航 → start_return() 反向重播角度歷史
      回航中每幀 → update_tracking() 同時做偵測，偵測到後 get_return_rc() 直接介入控制
    """
    def __init__(self, tello, tracker,
                 cam_index: int = 1,
                 port: str = GIMBAL_CONFIG["SERIAL_PORT"]):
        self.cfg = GIMBAL_CONFIG
        self.tello   = tello
        self.tracker = tracker
        self._port   = port

        # 串口（起飛後才開）
        self._serial      = None
        self._serial_lock = threading.Lock()

        # 狀態
        self._tracking     = False   # 起飛後才為 True
        self._returning    = False   # 進入返航後
        self._visual_guide = False   # 已偵測到無人機，純視覺引導中
        self._landed       = False

        # 雲台角度
        self._cur_pan  = self.cfg["PAN_CENTER"]
        self._cur_tilt = self.cfg["TILT_CENTER"]

        # 角度歷史
        self._history: deque = deque(maxlen=self.cfg["MAX_HISTORY"])
        self._last_record_t  = 0.0

        # 返航重播
        self._replay_list = []
        self._replay_idx  = 0
        self._last_replay_t = 0.0

        # 視覺引導遺失計時
        self._lost_t  = 0.0

        # 最新偵測結果（update_tracking 更新，get_return_rc 直接用）
        self._last_bbox  = None
        self._last_area  = 0
        self._last_frame = None   # update_tracking 讀到的 frame

        # webcam + 模型（程式啟動就初始化）
        self._cap   = None
        self._preview_win = "Gimbal Webcam"
        self._cam_idx = cam_index

        # 直接載入模型（和 box2/barcode1 相同方式）
        self._model = YOLO(DRONE_MODEL_PATH)
        print(f"[Gimbal] ✅ 載入 {DRONE_MODEL_PATH}")
        self._open_camera()
        if self._cap is not None:
            cv2.namedWindow(self._preview_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._preview_win, DRONE_FRAME_W, DRONE_FRAME_H)
            print("[Gimbal] 預覽視窗開啟，等待起飛")

    # ═══════════════════════════════════════════════════════════════
    # 公開 API
    # ═══════════════════════════════════════════════════════════════

    def start_tracking(self):
        """起飛後呼叫：開串口，開始讓雲台跟隨並記錄角度歷史"""
        self._open_serial()
        self._send_angle(self.cfg["PAN_CENTER"], self.cfg["TILT_CENTER"])
        self._tracking = True
        self._history.clear()
        print("[Gimbal] ✅ 雲台追蹤啟動")

    def update_tracking(self):
        """
        每幀必須呼叫（起飛前後都要呼叫）。

        做三件事：
          1. 讀 webcam frame + YOLO 偵測
          2. 顯示預覽視窗（含偵測框 / 準線 / 狀態列）
          3. 起飛後（_tracking=True）：送角度給雲台、記錄歷史
        偵測結果存在 self._last_bbox / self._last_frame，
        供 get_return_rc() 在同幀直接使用，不重複讀 webcam。
        """
        if self._cap is None:
            return

        ret, frame = self._cap.read()
        if not ret:
            return
        self._last_frame = frame.copy()

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        bbox = self._detect(frame)
        self._last_bbox = bbox
        self._last_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) if bbox else 0

        # ── 起飛後送角度給雲台 ───────────────────────────────────
        if self._tracking and not self._returning and bbox is not None:
            x1, y1, x2, y2 = bbox
            ex = (x1+x2)//2 - cx
            ey = (y1+y2)//2 - cy
            p = self._cur_pan  + self.cfg["KP_PAN"]  * ex if abs(ex) > self.cfg["DEADZONE_PX"] else self._cur_pan
            t = self._cur_tilt + self.cfg["KP_TILT"] * ey if abs(ey) > self.cfg["DEADZONE_PX"] else self._cur_tilt
            p = max(self.cfg["PAN_MIN"],  min(self.cfg["PAN_MAX"],  p))
            t = max(self.cfg["TILT_MIN"], min(self.cfg["TILT_MAX"], t))
            if p != self._cur_pan or t != self._cur_tilt:
                self._cur_pan, self._cur_tilt = p, t
                self._send_angle(p, t)
            # 記錄角度歷史
            now = time.time()
            if now - self._last_record_t >= self.cfg["RECORD_INTERVAL"]:
                self._history.append((self._cur_pan, self._cur_tilt))
                self._last_record_t = now

        # ── 返航重播：在 update 裡推進雲台角度 ──────────────────
        if self._returning and not self._visual_guide:
            now = time.time()
            if (self._replay_list and
                    self._replay_idx < len(self._replay_list) and
                    now - self._last_replay_t >= self.cfg["REPLAY_INTERVAL"] / self.cfg["REPLAY_SPEED"]):
                pan, tilt = self._replay_list[self._replay_idx]
                self._send_angle(pan, tilt)
                self._cur_pan, self._cur_tilt = pan, tilt
                self._replay_idx += 1
                self._last_replay_t = now

        # ── 繪製 UI ──────────────────────────────────────────────
        display = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bx, by = (x1+x2)//2, (y1+y2)//2
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(display, (bx, by), 5, (0,255,0), -1)
            cv2.putText(display, f"DRONE  area:{self._last_area}",
                (x1, max(y1-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.arrowedLine(display, (cx,cy), (bx,by), (255,100,0), 2, tipLength=0.2)

        # 準線
        cross_col = (0,255,0) if bbox else (0,0,200)
        cv2.line(display, (cx-20,cy),(cx+20,cy), cross_col, 1)
        cv2.line(display, (cx,cy-20),(cx,cy+20), cross_col, 1)

        # 狀態列
        if self._visual_guide:
            status = f"VISUAL GUIDE  area:{self._last_area}  pan:{self._cur_pan:.0f}  tilt:{self._cur_tilt:.0f}"
            scol   = (0, 255, 100)
        elif self._returning:
            pct    = int(self._replay_idx / max(len(self._replay_list),1) * 100)
            status = f"RETURN REPLAY {pct}%  pan:{self._cur_pan:.0f}  tilt:{self._cur_tilt:.0f}"
            scol   = (0, 200, 255)
        elif self._tracking:
            status = f"TRACKING  pan:{self._cur_pan:.0f}  tilt:{self._cur_tilt:.0f}  hist:{len(self._history)}"
            scol   = (0, 255, 180)
        else:
            status = "PREVIEW  (waiting for takeoff)"
            scol   = (180, 180, 180)

        cv2.putText(display, status, (10,24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, scol, 2)
        cv2.imshow(self._preview_win, display)
        cv2.waitKey(1)

    def start_return(self):
        """進入返航：反向重播角度歷史，準備迎接無人機"""
        self._returning    = True
        self._visual_guide = False
        self._landed       = False
        self._lost_t       = 0.0
        self._replay_list  = list(reversed(self._history))
        self._replay_idx   = 0
        self._last_replay_t = time.time()
        print(f"[Gimbal] 返航模式，歷史幀={len(self._replay_list)}")
        if not self._replay_list:
            self._send_angle(self.cfg["PAN_CENTER"], self.cfg["TILT_CENTER"])

    def get_return_rc(self) -> Optional[list]:
        """
        返航模式每幀呼叫（在 update_tracking 之後）。
        直接用 update_tracking 存好的 _last_bbox，不重複讀 webcam。

        回傳值：
          None  → 尚未偵測到，讓 ReturnHomeController 繼續飛
          list  → 視覺引導的 RC 指令 [lr, fb, ud, yaw]
        """
        if not self._returning:
            return None

        bbox = self._last_bbox

        if bbox is not None:
            self._last_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            self._lost_t = 0.0
            if not self._visual_guide:
                print(f"[Gimbal] ✅ 偵測到無人機（area={self._last_area}），介入控制")
                self._visual_guide = True
            return self._calc_visual_rc(bbox)
        else:
            if self._visual_guide:
                if self._lost_t == 0.0:
                    self._lost_t = time.time()
                elif time.time() - self._lost_t > self.cfg["LOST_HOVER_SEC"]:
                    print("[Gimbal] ⚠️ 遺失超過 2s，交回 RTH")
                    self._visual_guide = False
                    self._lost_t = 0.0
                return [0, 0, 0, 0]   # 遺失時懸停
            return None   # 重播中未找到，讓 RTH 繼續

    def is_active(self) -> bool:
        return self._tracking or self._returning

    def is_landing(self) -> bool:
        return self._landed

    def stop(self):
        self._tracking  = False
        self._returning = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._serial:
            try: self._serial.close()
            except: pass
        try:
            cv2.destroyWindow(self._preview_win)
        except Exception:
            pass

    def draw_hud(self, main_frame):
        """在 Tello 主畫面左下角疊加 webcam 縮圖"""
        f = self._last_frame
        if f is None:
            return main_frame
        thumb = cv2.resize(f, (320, 180))
        if self._last_bbox is not None:
            x1,y1,x2,y2 = self._last_bbox
            sx = 320 / f.shape[1]; sy = 180 / f.shape[0]
            cv2.rectangle(thumb, (int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),(0,255,0),2)
        fh, fw = main_frame.shape[:2]
        main_frame[fh-190:fh-10, 10:330] = thumb
        cv2.rectangle(main_frame,(10,fh-190),(330,fh-10),(0,220,0),1)
        lbl = "VISUAL GUIDE" if self._visual_guide else ("RETURN" if self._returning else "GIMBAL CAM")
        cv2.putText(main_frame, lbl,(10,fh-195),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,100),2)
        return main_frame

    # ═══════════════════════════════════════════════════════════════
    # 內部方法
    # ═══════════════════════════════════════════════════════════════

    def _calc_visual_rc(self, bbox) -> list:
        """
        視覺引導 RC。只負責 lr/ud 對中，fb=0 由呼叫端合併 RTH 飛近指令。
        框框達到 LAND_AREA_THRESH → 直接降落。
        """
        if self._last_frame is None:
            return [0, 0, 0, 0]
        h, w = self._last_frame.shape[:2]
        x1,y1,x2,y2 = bbox
        bx, by = (x1+x2)//2, (y1+y2)//2
        area   = (x2-x1)*(y2-y1)

        if area >= self.cfg["LAND_AREA_THRESH"]:
            print(f"[Gimbal] 🛬 降落（area={area}）")
            self.tello.land()
            self._landed    = True
            self._returning = False
            return [0, 0, 0, 0]

        ex = bx - w//2
        ey = by - h//2
        lr = int(self.cfg["KP_LR"] * ex) if abs(ex) > 30 else 0
        ud = int(-self.cfg["KP_UD"] * ey) if abs(ey) > 25 else 0
        lr = max(-self.cfg["MAX_LR"], min(self.cfg["MAX_LR"], lr))
        ud = max(-self.cfg["MAX_UD"], min(self.cfg["MAX_UD"], ud))

        # 雲台同步跟隨
        p = self._cur_pan  + self.cfg["KP_PAN"]  * ex if abs(ex) > self.cfg["DEADZONE_PX"] else self._cur_pan
        t = self._cur_tilt + self.cfg["KP_TILT"] * ey if abs(ey) > self.cfg["DEADZONE_PX"] else self._cur_tilt
        p = max(self.cfg["PAN_MIN"], min(self.cfg["PAN_MAX"], p))
        t = max(self.cfg["TILT_MIN"],min(self.cfg["TILT_MAX"],t))
        if p != self._cur_pan or t != self._cur_tilt:
            self._cur_pan, self._cur_tilt = p, t
            self._send_angle(p, t)

        # fb=0：前進由主迴圈合併 RTH 的 fb 指令
        return [lr, 0, ud, 0]

    def _send_angle(self, pan: float, tilt: float):
        cmd = f"P{int(pan)},T{int(tilt)}\n"
        with self._serial_lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.write(cmd.encode())
                except Exception as e:
                    print(f"[Gimbal] serial: {e}")

    def _detect(self, frame) -> Optional[tuple]:
        if self._model is None:
            return None
        try:
            results = self._model(frame, verbose=False, device=0)[0]
            best, best_area = None, self.cfg["DETECT_MIN_AREA"]
            for box in results.boxes:
                if float(box.conf) < drone_conf:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                area = (x2-x1)*(y2-y1)
                if area > best_area:
                    best_area, best = area, (x1,y1,x2,y2)
            return best
        except Exception as e:
            print(f"[Gimbal] detect: {e}")
            return None


    def _open_camera(self):
        self._cap = cv2.VideoCapture(self._cam_idx)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DRONE_FRAME_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  DRONE_FRAME_H)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"[Gimbal] webcam index={self._cam_idx} 開啟成功")
        else:
            print(f"[Gimbal] ❌ webcam index={self._cam_idx} 無法開啟")
            self._cap = None

    def _open_serial(self):
        try:
            import serial as _serial
            import serial.tools.list_ports as _ports
            self._serial = _serial.Serial(self._port, self.cfg["SERIAL_BAUD"], timeout=1)
            time.sleep(2)
            self._serial.readline()  # 讀掉 READY
            print(f"[Gimbal] ESP32 串口 {self._port} 連線成功")
        except Exception as e:
            print(f"[Gimbal] ❌ 串口失敗: {e}")
            self._serial = None


class ReturnHomeController:
    """
    低電量自動返航降落控制器。
    三階段：fly（直線飛回起飛點）→ hover（懸停 2 秒穩定）→ land（降落）

    RC 座標推導（FlightTracker 軸映射確認）：
      tracker.z += (-vgx)*dt, tracker.x += (-vgy)*dt
      fb>0 => vgy 增 => tracker.x 減 => 往 -x 方向
      lr>0 => vgx 增 => tracker.z 減 => 往 -z 方向
      yaw=0 時到達 (dx,dz): fb = -(ndx*sin+ndz*cos), lr = -(ndx*cos-ndz*sin)

    無超時：fly 階段飛到真正到達起飛點 ARRIVE_CM 以內才切換 hover。
    """
    def __init__(self, tello, tracker):
        self.cfg = LOW_BATTERY_CONFIG
        self.tello   = tello
        self.tracker = tracker
        self._phase  = "idle"
        self._t      = 0.0

    def start(self):
        """觸發返航"""
        dx   = self.tracker.home[0] - self.tracker.x
        dz   = self.tracker.home[2] - self.tracker.z
        dist = math.sqrt(dx**2 + dz**2)
        print(f"[ReturnHome] 啟動：估計距離={dist:.0f}cm")
        self._phase = "fly"
        self._t     = time.time()

    def get_rc(self) -> list:
        """每幀呼叫，回傳 [lr, fb, ud, yaw]。無超時，飛到到達才切換。"""
        if self._phase == "idle":
            return [0, 0, 0, 0]

        elapsed = time.time() - self._t

        if self._phase == "fly":
            dx      = self.tracker.home[0] - self.tracker.x
            dz      = self.tracker.home[2] - self.tracker.z
            dist_cm = math.sqrt(dx**2 + dz**2)

            if dist_cm > self.cfg["ARRIVE_CM"]:
                total = max(dist_cm, 1.0)
                ndx   = dx / total
                ndz   = dz / total

                # 世界座標 (ndx, ndz) → 機體 RC 指令
                # 推導自 FlightTracker 軸映射（見 class docstring）
                yaw_r = math.radians(self.tracker.yaw)
                fb_v  = -int(self.cfg["SPEED"] * (
                    ndx * math.sin(yaw_r) + ndz * math.cos(yaw_r)))
                lr_v  = -int(self.cfg["SPEED"] * (
                    ndx * math.cos(yaw_r) - ndz * math.sin(yaw_r)))

                try:    cur_h = self.tello.get_height()
                except: cur_h = 80
                ud_v = self.cfg["DESCEND_SPD"] if cur_h > self.cfg["TARGET_H_CM"] else 0
                return [lr_v, fb_v, ud_v, 0]
            else:
                print(f"[ReturnHome] 到達起飛點（dist={dist_cm:.0f}cm），懸停...")
                self._phase = "hover"
                self._t     = time.time()
                return [0, 0, 0, 0]

        elif self._phase == "hover":
            if elapsed >= self.cfg["HOVER_SEC"]:
                print("[ReturnHome] 降落")
                self._phase = "land"
                self.tello.land()
            return [0, 0, 0, 0]

        else:  # land
            return [0, 0, 0, 0]

    def is_active(self) -> bool:
        return self._phase != "idle"

    def is_landing(self) -> bool:
        return self._phase == "land"

    @property
    def phase(self) -> str:
        return self._phase


# ===================== 主任務控制器 =====================
class TelloMissionController:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.tello.set_speed(50)

        print(f"電池電量: {self.tello.get_battery()}%")

        self.midas      = MidASCruiser()
        self.forward    = ForwardTracker()
        self.circle     = CircleScanner()
        self.qr_scanner = QRScanner()

        # 飛行位置追蹤（Dead Reckoning）
        self.tracker = FlightTracker()

        # 返航控制器 + 雲台 + webcam 視覺降落
        self.return_home  = ReturnHomeController(self.tello, self.tracker)
        self.webcam_land  = WebcamLanding(self.tello, self.tracker)
        self.gimbal       = GimbalTracker(self.tello, self.tracker)
        self._scanned_popup_until = 0.0  # popup 顯示到此時間戳

        # 隨機高度控制
        self._alt_next_time = float("inf")  # 起飛前不觸發
        self._alt_ud_cmd    = 0              # 目前高度指令（+上/-下/0靜止）



        # 電量監控
        self._last_battery_check = 0.0
        self._low_battery_triggered = False


        self.current_state  = DroneState.MIDAS
        self.state_start_time = time.time()
        self.manual_mode    = False
        self.running        = True

        pygame.init()
        pygame.display.set_mode((300, 200))
        pygame.display.set_caption("Tello Mission Control")

    def get_keyboard_control(self):
        lr = fb = ud = yv = 0
        manual_active    = False
        quit_flag        = False
        force_state_change = None
        takeoff_command  = False
        land_command     = False

        pygame.event.pump()
        keys = pygame.key.get_pressed()

        SPEED = YAW_SPEED = UD_SPEED = 50

        if keys[pygame.K_w]:       ud = UD_SPEED;    manual_active = True
        if keys[pygame.K_s]:       ud = -UD_SPEED;   manual_active = True
        if keys[pygame.K_a]:       yv = -YAW_SPEED;  manual_active = True
        if keys[pygame.K_d]:       yv = YAW_SPEED;   manual_active = True
        if keys[pygame.K_UP]:      fb = SPEED;        manual_active = True
        if keys[pygame.K_DOWN]:    fb = -SPEED;       manual_active = True
        if keys[pygame.K_LEFT]:    lr = -SPEED;       manual_active = True
        if keys[pygame.K_RIGHT]:   lr = SPEED;        manual_active = True
        if keys[pygame.K_SPACE]:
            lr = fb = ud = yv = 0
            manual_active = True
        if keys[pygame.K_t]:  takeoff_command    = True
        if keys[pygame.K_l]:  land_command       = True
        if keys[pygame.K_1]:  force_state_change = DroneState.MIDAS
        if keys[pygame.K_2]:  force_state_change = DroneState.FORWARD
        if keys[pygame.K_3]:  force_state_change = DroneState.CIRCLE
        if keys[pygame.K_4]:  force_state_change = DroneState.QR_SCAN
        if keys[pygame.K_ESCAPE]: quit_flag = True

        return (manual_active, lr, fb, ud, yv,
                quit_flag, force_state_change, takeoff_command, land_command)

    def change_state(self, new_state, qr_bbox=None):
        old_state          = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()

        if new_state == DroneState.RETURN_HOME:
            self.return_home.start()
            self.gimbal.start_return()
        if new_state == DroneState.FORWARD:
            self.forward.start()
        elif new_state == DroneState.CIRCLE:
            self.circle.start()
        elif new_state == DroneState.QR_SCAN:
            self.qr_scanner.start(qr_bbox)

        print(f"\n🔄 狀態切換: {old_state} → {new_state}")

    def _get_random_alt_cmd(self) -> int:
        """
        MIDAS 巡航時每 ALT_CHANGE_INTERVAL 秒隨機決定高度指令。
        讀取 get_height() 確保不超出上下限。
        回傳 ud 值（正=上升，負=下降，0=靜止）
        """
        cfg = MIDAS_CONFIG
        now = time.time()
        if now >= self._alt_next_time:
            self._alt_next_time = now + cfg["ALT_CHANGE_INTERVAL"]
            try:
                h = float(self.tello.get_height())
            except Exception:
                h = 100.0   # 讀不到時假設中間高度
            if h <= cfg["ALT_MIN_CM"]:
                self._alt_ud_cmd = cfg["ALT_SPEED"]    # 太低，強制上升
            elif h >= cfg["ALT_MAX_CM"]:
                self._alt_ud_cmd = -cfg["ALT_SPEED"]   # 太高，強制下降
            else:
                # 隨機：上升/靜止/下降 各 1/3 機率
                import random
                choice = random.randint(0, 2)
                self._alt_ud_cmd = (
                    cfg["ALT_SPEED"]  if choice == 0 else
                    -cfg["ALT_SPEED"] if choice == 1 else 0)
        return self._alt_ud_cmd

    def _check_battery(self):
        """週期性電量檢查，低電量時切換回航"""
        now = time.time()
        if now - self._last_battery_check < LOW_BATTERY_CONFIG["CHECK_INTERVAL"]:
            return
        self._last_battery_check = now

        try:
            bat = self.tello.get_battery()
        except Exception:
            return

        if bat <= LOW_BATTERY_CONFIG["THRESHOLD"] and not self._low_battery_triggered:
            self._low_battery_triggered = True
            print(f"🔋 低電量！電量={bat}%，自動返航")
            self.change_state(DroneState.RETURN_HOME)

    def run(self):
        print("\n" + "="*50)
        print("Tello 四階段任務控制器啟動 (優化版 v2)")
        print("狀態流程: MIDAS → FORWARD → CIRCLE → QR_SCAN → MIDAS")
        print("="*50)
        print("\n[控制鍵]")
        print("  T: 起飛")
        print("  L: 降落")
        print("  W/S: 上升/下降")
        print("  A/D: 左轉/右轉")
        print("  方向鍵: 前進/後退/左移/右移")
        print("  數字鍵1-4: 強制切換狀態")
        print("  ESC: 緊急停止")
        print("="*50)

        frame_reader      = self.tello.get_frame_read()
        last_control_time = time.time()

        print("\n🛸 請按 T 起飛")

        try:
            while self.running:
                frame = frame_reader.frame
                if frame is None:
                    time.sleep(0.05)
                    continue

                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

                # 低電量檢查（週期性）
                self._check_battery()

                (manual_active, lr, fb, ud, yv,
                 quit_flag, force_state,
                 takeoff_cmd, land_cmd) = self.get_keyboard_control()

                self.tracker.update(self.tello, is_manual=manual_active)
                self.gimbal.update_tracking()
                # ORB-SLAM3: send frame to WSL1, apply correction if available


                if quit_flag:
                    print("使用者中斷程式")
                    break

                if takeoff_cmd:
                    print("🛸 手動起飛")
                    try:
                        self.tello.takeoff()
                    except Exception as e:
                        print(f"❌ 起飛失敗：{e}")
                        print("請確認：")
                        print("  1. Tello 重新開關機後再連線")
                        print("  2. 螺旋槳安裝正確（A/B 不可互換）")
                        print("  3. 放置在平坦地面")
                        print("  4. 電量 > 20%")
                        continue
                    time.sleep(1)
                    time.sleep(1.5)   # 等起飛穩定
                    self.tracker.reset_pose()   # 重設位移/軌跡，起飛點固定不動
                    self.gimbal.start_tracking()
                    # 起飛穩定後 5 秒才開始隨機高度變化
                    self._alt_next_time = time.time() + 5.0
                    self._alt_ud_cmd    = 0
                    print("📍 起飛點固定：(0, 0, 0)，開始軌跡記錄")

                if land_cmd:
                    print("🛬 手動降落")
                    # 停止隨機高度，避免 ud 指令干擾降落
                    self._alt_ud_cmd    = 0
                    self._alt_next_time = float("inf")
                    self.tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.3)
                    self.tello.land()
                    time.sleep(1)

                if force_state:
                    self.change_state(force_state)


                if not manual_active:
                    control_cmd = [0, 0, 0, 0]

                    # ─── MIDAS 模式 ───────────────────────────────────────
                    if self.current_state == DroneState.MIDAS:
                        depth_norm, center_depth = self.midas.process_frame(frame)
                        fbv, yv = self.midas.get_control(center_depth, time.time())
                        # 隨機高度
                        ud_midas = self._get_random_alt_cmd()
                        control_cmd = [0, fbv, ud_midas, yv]

                        depth_display = cv2.applyColorMap(
                            (depth_norm * 255).astype(np.uint8),
                            cv2.COLORMAP_JET
                        )
                        cv2.imshow("Depth Map", depth_display)

                        # 推送深度幀給點雲視窗
                        pose = (self.tracker.x, self.tracker.y,
                                self.tracker.z, self.tracker.yaw)

                        # conf 提高至 0.75
                        results = self.forward.model(frame, conf=box_conf, verbose=False)
                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            boxes = results[0].boxes

                            # 只看通過過濾的框
                            valid_boxes = []
                            for b in boxes:
                                x1, y1, x2, y2 = map(int, b.xyxy[0])
                                if self.forward._is_valid_box(x1, y1, x2, y2):
                                    valid_boxes.append(b)

                            if valid_boxes:
                                best_box = max(valid_boxes, key=lambda b:
                                    (b.xyxy[0][2] - b.xyxy[0][0]) *
                                    (b.xyxy[0][3] - b.xyxy[0][1]))
                                area = ((best_box.xyxy[0][2] - best_box.xyxy[0][0]) *
                                        (best_box.xyxy[0][3] - best_box.xyxy[0][1]))

                                if area > MIDAS_CONFIG["TARGET_FOUND_AREA"]:
                                    print(f"🎯 巡航中找到目標! 面積={area:.0f}")
                                    self.change_state(DroneState.FORWARD)

                        frame = self.midas.draw_overlay(frame, center_depth, fbv, yv)

                    # ─── FORWARD 模式 ─────────────────────────────────────
                    elif self.current_state == DroneState.FORWARD:
                        lr, fb, ud, yv, bbox, area, reached = \
                            self.forward.process_frame(frame)
                        control_cmd = [lr, fb, ud, yv]

                        if bbox:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Area: {area}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        cv2.putText(frame, "MODE: FORWARD TRACK", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        if reached:
                            print(f"🎉 到達目標! 面積={area}，開始環繞")
                            self.tello.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            self.change_state(DroneState.CIRCLE)
                        elif self.forward.should_abort() or self.forward.is_timeout():
                            print("↩️ 返回巡航模式")
                            self.change_state(DroneState.MIDAS)

                    # ─── CIRCLE 模式 ──────────────────────────────────────
                    elif self.current_state == DroneState.CIRCLE:
                        lr, fb, ud, yv, bbox, qr_detected, qr_bbox = \
                            self.circle.process_frame(frame)
                        control_cmd = [lr, fb, ud, yv]

                        if bbox:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            area = (x2 - x1) * (y2 - y1)
                            cv2.putText(
                                frame,
                                f"Area: {area}  Target: {CIRCLE_CONFIG['TARGET_AREA']}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1
                            )

                        if qr_detected and qr_bbox:
                            qx1, qy1, qx2, qy2 = qr_bbox
                            cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (255, 255, 0), 3)
                            cv2.putText(frame, "QR DETECTED!", (qx1, qy1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        cv2.putText(frame, "MODE: CIRCLE SCAN", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"FB: {fb}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        if qr_detected and qr_bbox and self.circle.is_complete():
                            print("🔍 偵測到QR Code，準備靠近掃描")
                            self.tello.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            self.change_state(DroneState.QR_SCAN, qr_bbox)
                        elif self.circle.should_abort() or self.circle.is_timeout():
                            # Bug Fix: 原本 is_timeout() AND is_complete() 雙重條件，
                            # 當沒有 QR 但環繞超時時，因 is_complete() 邏輯混用導致卡死。
                            # 現在只要超時或目標丟失就直接退出。
                            print("↩️ 環繞完成/超時，返回巡航")
                            self.change_state(DroneState.MIDAS)

                    # ─── QR_SCAN 模式 ─────────────────────────────────────
                    elif self.current_state == DroneState.QR_SCAN:
                        lr, fb, ud, yv, bbox, area, reached, decoded, data = \
                            self.qr_scanner.process_frame(frame)
                        control_cmd = [lr, fb, ud, yv]

                        if bbox:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                            cv2.putText(frame, f"QR Area: {area}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        if decoded and data:
                            cv2.putText(frame, f"SCANNED: {data}", (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        cv2.putText(frame, "MODE: QR SCAN", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(frame, f"FB: {fb}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(
                            frame,
                            f"Attempts: {self.qr_scanner.consecutive_failures}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                        )

                        if self.qr_scanner.is_complete() or self.qr_scanner.should_abort():
                            if decoded and data:
                                if data in self.qr_scanner.scanned_set:
                                    print(f"↩️ 條碼已掃描過: {data}，返回巡航")
                                else:
                                    print("✅ QR掃描完成！返回巡航")
                                self._scanned_popup_until = time.time() + 3.0  # 顯示 3 秒

                                self.tello.send_rc_control(0, 0, 50, 0)
                                time.sleep(1)
                                self.change_state(DroneState.MIDAS)
                            else:
                                print("⏰ QR掃描超時，返回巡航")
                                self.change_state(DroneState.MIDAS)


                    # ── RETURN_HOME 返航 ──────────────────────────────────
                    if self.current_state == DroneState.RETURN_HOME:
                        if self.webcam_land.is_active():
                            control_cmd = self.webcam_land.get_rc()
                            frame = self.webcam_land.draw_hud(frame)
                            if self.webcam_land.is_landing():
                                self.change_state(DroneState.MIDAS)
                        else:
                            rc_gimbal = self.gimbal.get_return_rc()
                            if rc_gimbal is not None:
                                # lr/ud 由 Gimbal 對中，fb 由 RTH 飛近
                                rc_rth = self.return_home.get_rc()
                                control_cmd = [
                                    rc_gimbal[0],  # lr  ← Gimbal
                                    rc_rth[1],     # fb  ← RTH
                                    rc_gimbal[2],  # ud  ← Gimbal
                                    0,             # yaw ← 不旋轉
                                ]
                                frame = self.gimbal.draw_hud(frame)
                                if self.gimbal.is_landing():
                                    self.change_state(DroneState.MIDAS)
                            else:
                                control_cmd = self.return_home.get_rc()
                                if self.webcam_land.should_handoff():
                                    self.webcam_land.start()

                    current_time = time.time()
                    if current_time - last_control_time >= CONTROL_INTERVAL:
                        self.tello.send_rc_control(*control_cmd)
                        last_control_time = current_time

                else:
                    self.tello.send_rc_control(lr, fb, ud, yv)
                    cv2.putText(frame, "MANUAL MODE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                cv2.putText(frame, f"State: {self.current_state}",
                            (10, FRAME_H-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Battery: {self.tello.get_battery()}%",
                            (FRAME_W-150, FRAME_H-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "T:Takeoff L:Land",
                            (10, FRAME_H-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # ── SCANNED! popup 覆蓋層 ──
                if time.time() < self._scanned_popup_until:
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w//2-180, h//2-60),
                                  (w//2+180, h//2+60), (0, 200, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, "SCANNED !",
                                (w//2-140, h//2+20),
                                cv2.FONT_HERSHEY_DUPLEX, 1.8,
                                (255, 255, 255), 3)
                cv2.imshow("Tello Mission Control", frame)

                if cv2.waitKey(1) == 27:
                    break

        except Exception as e:
            print(f"錯誤: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        print("\n🧹 清理資源中...")
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        print("⚠️  請記得手動降落")

        # 儲存軌跡、關閉點雲視窗
        self.tracker.save_path_csv()

        self.tello.streamoff()
        pygame.quit()
        cv2.destroyAllWindows()
        print("✅ 程式結束")

# ===================== 程式進入點 =====================
if __name__ == "__main__":
    controller = TelloMissionController()
    controller.run()
