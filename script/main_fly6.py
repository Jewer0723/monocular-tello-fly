"""
Tello 四階段任務控制系統（優化版 v3）
狀態機: MIDAS → FORWARD → CIRCLE → QR_SCAN → MIDAS

修改紀錄：
  [Fix 1] box2 辨識過濾：加入長寬比、畫面佔比、最小面積後處理，conf 提高至 0.75
  [Fix 2] 環繞 yaw 跟不上：降低環繞速度、提高 yaw 修正上限、移除 yaw 縮小係數、縮短控制間隔
  [Fix 3] 條碼掃不到：ROI 強制放大、加銳化/OTSU 預處理、TARGET_AREA 增大、對全帧兜底解碼
  [v3]   新增飛行軌跡紀錄器（航位推算）、低電量自動回航降落、MiDaS 偽點雲即時顯示
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
import math
import threading
from datetime import datetime

# open3d 為可選依賴，沒安裝時關閉點雲視窗
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️  open3d 未安裝，點雲視窗將關閉。安裝：pip install open3d")

# ===================== 全局配置 =====================
FRAME_W, FRAME_H = 640, 480
CONTROL_INTERVAL = 0.05          # [Fix 2] 原 0.1 → 0.05，提高控制頻率
box_conf = 0.7
qr_conf = 0.7

# ===================== 狀態定義 =====================
class DroneState:
    MIDAS       = "MIDAS"        # 巡航避障模式
    FORWARD     = "FORWARD"      # 前進接近目標模式
    CIRCLE      = "CIRCLE"       # 環繞掃描模式（偵測條碼）
    QR_SCAN     = "QR_SCAN"      # QR Code掃描模式（鎖定靠近）
    RETURN_HOME = "RETURN_HOME"  # [v3] 低電量自動回航

# ===================== MidAS 巡航參數 =====================
MIDAS_CONFIG = {
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
    "ORBIT_SPEED":           7,   # [Fix 2] 原 7 → 5，降低環繞速度
    "YAW_CORRECTION_SPEED": 25,   # [Fix 2] 原 15 → 25，提高 yaw 修正上限
    "HEIGHT_CORRECTION_SPEED": 15,
    "MIN_CIRCLE_TIME":        5,
    "MAX_CIRCLE_TIME":       30,
    "TARGET_LOST_TIMEOUT":    1,  # [Fix 2] 原 1 → 2，容許短暫丟失
    "TARGET_AREA":       120000,
    "AREA_TOLERANCE":      5000,
    "KP_FORWARD":         0.0006,
    "MAX_EXECUTION_TIME":    30,
}

# ===================== QR掃描參數 =====================
QR_SCAN_CONFIG = {
    "TARGET_AREA":          60000,  # [Fix 3] 原 20000 → 60000，確保靠得夠近
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
    "MIN_AREA_BEFORE_DECODE": 40000,  # [Fix 3] 原 1000 → 40000
    "CSV_FILE":              "scanned_codes.csv"
}

# ===================== [v3] 低電量回航參數 =====================
LOW_BATTERY_CONFIG = {
    "THRESHOLD":       20,     # 低於此電量(%)觸發回航
    "CHECK_INTERVAL":   5,     # 每幾秒查一次電量
    "RETURN_SPEED":    20,     # 回航飛行速度
    "YAW_KP":         0.8,     # 偏航修正係數（對準起飛方向）
}

# ===================== [v3] 偽點雲參數（MiDaS 反投影）=====================
POINTCLOUD_CONFIG = {
    "UPDATE_INTERVAL":  0.5,   # 點雲更新間隔(秒)
    "MAX_POINTS":      8000,   # 最多保留點數
    "DOWNSAMPLE_STEP":    8,   # 深度圖取樣步長
    # Tello 相機近似內參（FOV≈82.6°, 640×480）
    "FX":             458.0,
    "FY":             458.0,
    "CX":             320.0,
    "CY":             240.0,
    "DEPTH_SCALE":      3.0,   # 相對深度縮放倍率（調整視覺密度感）
}

# ===================== [v3] 飛行軌跡紀錄器（航位推算 Dead Reckoning）=====================
class FlightTracker:
    """
    利用 Tello 的 get_speed_x/y/z() 與 get_yaw() 做航位推算。
    座標系：
      X = 右方（東）  Y = 上方  Z = 前方（北）
    起飛點固定為原點 (0, 0, 0)，yaw=0。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.x   = 0.0   # 右方累積位移 (cm)
        self.y   = 0.0   # 高度累積位移 (cm)
        self.z   = 0.0   # 前方累積位移 (cm)
        self.yaw = 0.0   # 目前偏航角 (度，逆時針為正)
        self.path: list[tuple] = [(0.0, 0.0, 0.0)]   # 軌跡列表 (x, z) 俯視
        self.last_time = time.time()
        self.home      = (0.0, 0.0, 0.0)   # 起飛點

    def update(self, tello: "Tello"):
        """每幀呼叫一次，從感測器讀值更新位置"""
        now = time.time()
        dt  = now - self.last_time
        self.last_time = now

        if dt <= 0 or dt > 1.0:   # 跳過異常 dt
            return

        try:
            # Tello SDK 速度單位：cm/s，已修正為機體座標系
            vx_body = tello.get_speed_x()   # 機體右方
            vy_body = tello.get_speed_y()   # 機體前方（SDK定義）
            vz_body = tello.get_speed_z()   # 機體上方
            yaw_deg = tello.get_yaw()       # 度，順時針為正（Tello 定義）
        except Exception:
            return

        self.yaw = yaw_deg
        yaw_rad  = math.radians(-yaw_deg)   # 轉為逆時針標準角

        # 機體速度 → 世界座標速度（繞 Y 軸旋轉）
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        world_vx =  cos_y * vx_body + sin_y * vy_body
        world_vz = -sin_y * vx_body + cos_y * vy_body
        world_vy =  vz_body

        # 積分位移
        self.x += world_vx * dt
        self.y += world_vy * dt
        self.z += world_vz * dt

        self.path.append((self.x, self.y, self.z))

    def get_return_yaw_error(self) -> float:
        """回傳「指向起飛點」所需的偏航誤差(度)"""
        dx = self.home[0] - self.x
        dz = self.home[2] - self.z
        target_yaw = math.degrees(math.atan2(dx, dz))   # 世界系目標角
        error = target_yaw - (-self.yaw)                 # 對齊 Tello 符號
        # 正規化到 [-180, 180]
        while error >  180: error -= 360
        while error < -180: error += 360
        return error

    def distance_to_home(self) -> float:
        """水平距離(cm)"""
        return math.sqrt((self.x - self.home[0])**2 +
                         (self.z - self.home[2])**2)

    def draw_minimap(self, frame, size=160, margin=10):
        """在畫面右上角繪製俯視軌跡小地圖"""
        h, w = frame.shape[:2]
        x0 = w - size - margin
        y0 = margin

        # 背景
        cv2.rectangle(frame, (x0, y0), (x0+size, y0+size), (30, 30, 30), -1)
        cv2.rectangle(frame, (x0, y0), (x0+size, y0+size), (100, 100, 100), 1)

        if len(self.path) < 2:
            return frame

        # 自動縮放
        xs = [p[0] for p in self.path]
        zs = [p[2] for p in self.path]
        span = max(max(xs)-min(xs), max(zs)-min(zs), 100)   # 最小 100cm
        scale = (size - 20) / span

        cx_map = x0 + size // 2
        cy_map = y0 + size // 2
        ox = (max(xs) + min(xs)) / 2
        oz = (max(zs) + min(zs)) / 2

        def to_px(px, pz):
            return (int(cx_map + (px - ox) * scale),
                    int(cy_map - (pz - oz) * scale))

        # 畫軌跡
        for i in range(1, len(self.path)):
            p1 = to_px(self.path[i-1][0], self.path[i-1][2])
            p2 = to_px(self.path[i][0],   self.path[i][2])
            cv2.line(frame, p1, p2, (0, 200, 255), 1)

        # 起飛點（綠圓）
        hp = to_px(self.home[0], self.home[2])
        cv2.circle(frame, hp, 5, (0, 255, 0), -1)

        # 目前位置（紅圓）
        cp = to_px(self.x, self.z)
        cv2.circle(frame, cp, 5, (0, 0, 255), -1)

        # 距離標示
        dist = self.distance_to_home()
        cv2.putText(frame, f"HOME:{dist:.0f}cm", (x0+2, y0+size-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, "MAP", (x0+2, y0+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        return frame

    def save_path_csv(self, filename="flight_path.csv"):
        """儲存軌跡到 CSV"""
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_cm", "y_cm", "z_cm"])
            writer.writerows(self.path)
        print(f"📁 軌跡已儲存: {filename}")


# ===================== [v3] 偽點雲視窗（open3d，執行緒更新）=====================
class PointCloudVisualizer:
    """
    在獨立執行緒開啟 open3d 視窗。
    主執行緒呼叫 push_frame() 傳入 depth_norm 和當前位姿，
    視窗每 UPDATE_INTERVAL 秒更新一次點雲。
    """
    def __init__(self, tracker: FlightTracker):
        self.tracker  = tracker
        self.enabled  = OPEN3D_AVAILABLE
        self._lock    = threading.Lock()
        self._pending_depth = None
        self._pending_pose  = None
        self._last_update   = 0.0
        self._all_points: list[np.ndarray] = []
        self._all_colors: list[np.ndarray] = []
        self._thread  = None
        self._running = False

        if self.enabled:
            self._vis = o3d.visualization.Visualizer()
            self._pcd = o3d.geometry.PointCloud()
            # 軌跡線段集合
            self._traj = o3d.geometry.LineSet()

    def start(self):
        if not self.enabled:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run_window, daemon=True)
        self._thread.start()

    def _run_window(self):
        self._vis.create_window("Obstacle Point Cloud (pseudo-SLAM)", 800, 600)
        self._vis.add_geometry(self._pcd)
        self._vis.add_geometry(self._traj)

        # 座標軸
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        self._vis.add_geometry(axes)

        cfg = self._vis.get_render_option()
        cfg.background_color = np.array([0.05, 0.05, 0.05])
        cfg.point_size = 1.5

        while self._running:
            self._maybe_update()
            self._vis.poll_events()
            self._vis.update_renderer()
            time.sleep(0.03)

        self._vis.destroy_window()

    def _maybe_update(self):
        now = time.time()
        if now - self._last_update < POINTCLOUD_CONFIG["UPDATE_INTERVAL"]:
            return

        with self._lock:
            depth = self._pending_depth
            pose  = self._pending_pose
            self._pending_depth = None

        if depth is None:
            return

        self._last_update = now
        try:
            new_pts, new_cols = self._depth_to_points(depth, pose)
        except Exception as e:
            print(f"[PointCloud] _depth_to_points error: {e}")
            return

        if new_pts is not None and len(new_pts) > 0:
            self._all_points.append(new_pts)
            self._all_colors.append(new_cols)

            # 限制最大點數
            total = sum(len(p) for p in self._all_points)
            while total > POINTCLOUD_CONFIG["MAX_POINTS"] and self._all_points:
                removed = len(self._all_points[0])
                self._all_points.pop(0)
                self._all_colors.pop(0)
                total -= removed

            pts_all = np.vstack(self._all_points)
            col_all = np.vstack(self._all_colors)
            self._pcd.points = o3d.utility.Vector3dVector(pts_all)
            self._pcd.colors = o3d.utility.Vector3dVector(col_all)

        # 更新軌跡線
        path = self.tracker.path
        if len(path) >= 2:
            pts_traj = np.array([[p[0], p[1], p[2]] for p in path],
                                dtype=np.float64)
            lines    = [[i, i+1] for i in range(len(pts_traj)-1)]
            colors   = [[0.0, 0.8, 1.0] for _ in lines]
            self._traj.points = o3d.utility.Vector3dVector(pts_traj)
            self._traj.lines  = o3d.utility.Vector2iVector(lines)
            self._traj.colors = o3d.utility.Vector3dVector(colors)

        self._vis.update_geometry(self._pcd)
        self._vis.update_geometry(self._traj)

    def _depth_to_points(self, depth_norm, pose):
        """MiDaS 深度圖 → 世界座標點雲（偽 SLAM）"""
        cfg  = POINTCLOUD_CONFIG
        step = cfg["DOWNSAMPLE_STEP"]
        fx, fy = cfg["FX"], cfg["FY"]
        cx, cy = cfg["CX"], cfg["CY"]
        scale  = cfg["DEPTH_SCALE"]

        h, w = depth_norm.shape
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        zd     = depth_norm[ys, xs] * scale + 0.1   # 避免 z=0

        # 反投影到相機座標
        xc = (xs - cx) / fx * zd
        yc = (ys - cy) / fy * zd
        zc = zd

        pts_cam = np.stack([xc.ravel(), yc.ravel(), zc.ravel()], axis=1)

        # 相機 → 世界座標（加上無人機當前位姿）
        if pose is not None:
            px, py, pz, yaw_deg = pose
            yaw_rad = math.radians(-yaw_deg)
            cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
            R = np.array([[cos_y, 0, sin_y],
                          [0,     1, 0    ],
                          [-sin_y,0, cos_y]])
            pts_world = (R @ pts_cam.T).T
            pts_world[:, 0] += px
            pts_world[:, 1] += py
            pts_world[:, 2] += pz
        else:
            pts_world = pts_cam

        # 剔除太近或太遠的點
        z_vals = pts_world[:, 2]
        mask   = (z_vals > 10) & (z_vals < scale * 250)
        pts_world = pts_world[mask]
        depth_vals = depth_norm[ys, xs].ravel()[mask]

        # mask 過濾後可能完全沒有有效點，直接回傳空
        if pts_world.shape[0] == 0:
            return None, None

        # 顏色：深度圖 JET 假彩色
        depth_vals = np.nan_to_num(depth_vals, nan=0.0, posinf=1.0, neginf=0.0)
        depth_u8 = (np.clip(depth_vals, 0.0, 1.0) * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(depth_u8.reshape(-1, 1), cv2.COLORMAP_JET)
        if color_map is None:
            return None, None
        colors = color_map.reshape(-1, 3)[:, ::-1].astype(np.float64) / 255.0

        return pts_world.astype(np.float64), colors

    def push_frame(self, depth_norm: np.ndarray, pose: tuple):
        """主執行緒呼叫：更新待處理深度圖"""
        if not self.enabled:
            return
        with self._lock:
            self._pending_depth = depth_norm.copy()
            self._pending_pose  = pose

    def stop(self):
        self._running = False


# ===================== MidAS 巡航避障控制器 =====================
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

# ===================== 目標追蹤基類 =====================
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
    # [Fix 1] 加入後處理過濾：長寬比、畫面佔比、最小面積
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

    def detect_target(self, frame, conf=box_conf):  # [Fix 1] conf 預設提高至 0.75
        results = self.model(frame, conf=conf, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # [Fix 1] 先過濾無效框
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
        super().__init__("../model/box2.pt", FORWARD_CONFIG)

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
        super().__init__("../model/box2.pt", CIRCLE_CONFIG)
        self.qr_model        = YOLO("../model/barcode1.pt")
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
            # [Fix 2] yaw 修正：移除縮小係數，直接全力修正
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
                        int(FORWARD_CONFIG["KP_YAW"] * error_x),  # [Fix 2] 移除 *0.3
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

# ===================== QR掃描控制器 =====================
class QRScanner(TargetTracker):
    """專門鎖定並掃描QR Code，無法解碼時持續前進"""

    def __init__(self):
        super().__init__("../model/barcode1.pt", QR_SCAN_CONFIG)
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
    # [Fix 3] 強化解碼流程：ROI 放大 + 多種預處理 + 全帧兜底
    # ------------------------------------------------------------------
    def decode_qr_code(self, frame, qr_bbox):
        """在QR Code區域內解碼"""
        if qr_bbox is None:
            return False, None

        x1, y1, x2, y2 = qr_bbox

        pad    = 40                              # [Fix 3] 擴大 padding（原 20）
        roi_x1 = max(0, x1 - pad)
        roi_y1 = max(0, y1 - pad)
        roi_x2 = min(FRAME_W, x2 + pad)
        roi_y2 = min(FRAME_H, y2 + pad)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return False, None

        # [Fix 3] 強制放大 ROI：pyzbar 對小圖解碼率極差
        roi_h, roi_w = roi.shape[:2]
        min_dim      = min(roi_h, roi_w)
        if min_dim < 150:
            scale = max(2, int(300 / (min_dim + 1e-5)))
            roi   = cv2.resize(roi, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # [Fix 3] 擴充預處理方法
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
            cv2.filter2D(gray, -1, sharpening_kernel),                          # [Fix 3] 銳化
            cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],             # [Fix 3] OTSU
            cv2.bitwise_not(gray),                                              # [Fix 3] 反色
            cv2.bitwise_not(
                cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),        # [Fix 3] 反色OTSU
        ]

        for method in methods:
            barcodes = pyzbar.decode(method)
            if barcodes:
                return True, barcodes[0].data.decode("utf-8")

        # [Fix 3] 最後兜底：對整張 frame 解碼（不限 ROI）
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

# ===================== 主控制器 =====================
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

        # [v3] 飛行軌跡紀錄器
        self.tracker = FlightTracker()

        # [v3] 偽點雲視窗
        self.pcl_vis = PointCloudVisualizer(self.tracker)
        self.pcl_vis.start()

        # [v3] 電量監控
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

        if new_state == DroneState.FORWARD:
            self.forward.start()
        elif new_state == DroneState.CIRCLE:
            self.circle.start()
        elif new_state == DroneState.QR_SCAN:
            self.qr_scanner.start(qr_bbox)
        elif new_state == DroneState.RETURN_HOME:     # [v3]
            print("🏠 啟動回航模式")

        print(f"\n🔄 狀態切換: {old_state} → {new_state}")

    def _check_battery(self):
        """[v3] 週期性電量檢查，低電量時切換回航"""
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
            print(f"🔋 低電量警告！電量={bat}%，強制回航")
            self.tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.3)
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

                # [v3] 更新軌跡位置（每幀）
                self.tracker.update(self.tello)

                # [v3] 低電量檢查（週期性）
                self._check_battery()

                (manual_active, lr, fb, ud, yv,
                 quit_flag, force_state,
                 takeoff_cmd, land_cmd) = self.get_keyboard_control()

                if quit_flag:
                    print("使用者中斷程式")
                    break

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

                if not manual_active:
                    control_cmd = [0, 0, 0, 0]

                    # ─── MIDAS 模式 ───────────────────────────────────────
                    if self.current_state == DroneState.MIDAS:
                        depth_norm, center_depth = self.midas.process_frame(frame)
                        fbv, yv = self.midas.get_control(center_depth, time.time())
                        control_cmd = [0, fbv, 0, yv]

                        depth_display = cv2.applyColorMap(
                            (depth_norm * 255).astype(np.uint8),
                            cv2.COLORMAP_JET
                        )
                        cv2.imshow("Depth Map", depth_display)

                        # [v3] 推送深度幀給點雲視窗
                        pose = (self.tracker.x, self.tracker.y,
                                self.tracker.z, self.tracker.yaw)
                        self.pcl_vis.push_frame(depth_norm, pose)

                        # [Fix 1] conf 提高至 0.75
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

                                self.tello.send_rc_control(0, 0, 50, 0)
                                time.sleep(1)
                                self.change_state(DroneState.MIDAS)
                            else:
                                print("⏰ QR掃描超時，返回巡航")
                                self.change_state(DroneState.MIDAS)

                    # ─── RETURN_HOME 模式 [v3] ────────────────────────────
                    elif self.current_state == DroneState.RETURN_HOME:
                        dist  = self.tracker.distance_to_home()
                        yaw_err = self.tracker.get_return_yaw_error()
                        spd   = LOW_BATTERY_CONFIG["RETURN_SPEED"]
                        kp    = LOW_BATTERY_CONFIG["YAW_KP"]

                        if dist < 60:
                            # 到達起飛點附近 → 降落
                            print("🏠 到達起飛點！自動降落")
                            self.tello.send_rc_control(0, 0, 0, 0)
                            time.sleep(0.5)
                            self.tello.land()
                            self.tracker.save_path_csv()
                            break
                        else:
                            # 先對齊方向再前進
                            yaw_cmd = int(kp * yaw_err)
                            yaw_cmd = max(-40, min(40, yaw_cmd))
                            if abs(yaw_err) < 15:
                                fb_cmd = spd     # 方向已對齊，向前飛
                            else:
                                fb_cmd = 0       # 先原地轉向
                            control_cmd = [0, fb_cmd, 0, yaw_cmd]

                        # HUD
                        cv2.putText(frame, "MODE: RETURN HOME", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                        cv2.putText(frame, f"Dist: {dist:.0f}cm  YawErr: {yaw_err:.1f}°",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)

                    current_time = time.time()
                    if current_time - last_control_time >= CONTROL_INTERVAL:
                        self.tello.send_rc_control(*control_cmd)
                        last_control_time = current_time

                else:
                    self.tello.send_rc_control(lr, fb, ud, yv)
                    cv2.putText(frame, "MANUAL MODE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # [v3] 俯視軌跡小地圖（右上角）
                frame = self.tracker.draw_minimap(frame)

                cv2.putText(frame, f"State: {self.current_state}",
                            (10, FRAME_H-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Battery: {self.tello.get_battery()}%",
                            (FRAME_W-150, FRAME_H-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "T:Takeoff L:Land",
                            (10, FRAME_H-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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

        # [v3] 儲存軌跡、關閉點雲視窗
        self.tracker.save_path_csv()
        self.pcl_vis.stop()

        self.tello.streamoff()
        pygame.quit()
        cv2.destroyAllWindows()
        print("✅ 程式結束")

# ===================== 程式入口 =====================
if __name__ == "__main__":
    controller = TelloMissionController()
    controller.run()
