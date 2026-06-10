"""
su4.3.py  –  Tello 雙模式任務控制系統 (去除爬升與右回掃，導入定距控制 + 無高度計定高 + 前後平滑版)
=============================================
功能 1（Mode 1）: 環繞巡檢  — MIDAS巡航 → FORWARD接近 → CIRCLE環繞 → QR_SCAN掃碼
功能 2（Mode 2）: 走道巡檢  — 起飛後直接 roll左掃描QR(5個) → MiDaS判定靠近對向面板
                             → 左轉180度 → 繼續roll左掃描(5個) → 繞到第二走道 → 重複 → 回航降落

修改紀錄:
  [su4.1] 移除起飛爬升 (CLIMB) 邏輯，起飛後直接進入 ROLL_SCAN。
  [su4.1] 移除右偏回朔 (RESCAN_RIGHT) 邏輯，若掃描丟失則直接繼續向左掃描。
  [su4.1] 導入 MiDaS 定距平移邏輯：SCAN 階段不再依賴 YOLO 面積決定前後距離。
  [su4.1-yaml] 將飛行速度、控制參數、模型路徑、任務參數集中到 mission_command.yaml。
  [su4.3] 移植 main_fly9.6 高度計定高控制功能：
          - 新增 maintain_height_enabled / height_kp / max_ud_speed / cruise_altitude_cm 參數。
          - 新增 _get_stable_height() 物理突變過濾器：忽略飛越桌子/貨架底座時高度計暴衝。
          - ROLL_SCAN 階段固定 ud=0，不使用高度計，也不使用 QR 垂直修正，避免掃描時上下跳動。
          - 有目標時垂直控制權由 qr_ud → base_ud，前後/左右/偏航仍交由 QRScanner。
  [su4.4] 新增簡化掃描模式：Mode2 不再每幀用 QR YOLO 鎖框控制，改成定時直接解碼，飛行更流暢。
          - 高度 PID 加入 deadband 與輸出平滑，降低上下震盪。
"""

import csv
import json
import math
import os
import random
import socket
import time
from collections import deque
from datetime import datetime
from typing import List, Dict, Any

import cv2
import numpy as np
import pygame
import torch
import yaml
from djitellopy import Tello
from pyzbar import pyzbar
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────
#  YAML 任務配置載入器
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(BASE_DIR, "mission_command.yaml")

def _resolve_project_path(path_value):
    if path_value is None:
        return None
    path_value = os.path.expandvars(os.path.expanduser(str(path_value)))
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(BASE_DIR, path_value)

def _deep_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d

class MissionLoader:
    def __init__(self, path: str = YAML_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self._cfg: Dict[str, Any] = yaml.safe_load(f)
        print(f"✅ 配置已載入：{path}")

    def get(self, *keys, default=None):
        return _deep_get(self._cfg, *keys, default=default)

    @property
    def frame_w(self):   return self.get("frame", "width",  default=640)
    @property
    def frame_h(self):   return self.get("frame", "height", default=480)
    @property
    def control_interval(self): return self.get("frame", "control_interval", default=0.05)

    @property
    def box_model(self): return _resolve_project_path(self.get("models", "box_model_path"))
    @property
    def qr_model(self):  return _resolve_project_path(self.get("models", "barcode_model_path"))
    @property
    def box_conf(self):  return self.get("models", "box_conf", default=0.70)
    @property
    def qr_conf(self):   return self.get("models", "qr_conf",  default=0.70)

    @property
    def mission_mode(self): return self.get("mission", "mode", default=2)

    def section(self, *keys) -> dict:
        v = self.get(*keys, default={})
        return v if isinstance(v, dict) else {}

CFG = MissionLoader()

def _validate_model_files():
    missing = []
    for label, path in [("box_model_path", CFG.box_model), ("barcode_model_path", CFG.qr_model)]:
        if not path or not os.path.exists(path):
            missing.append((label, path))
    if missing:
        print("\n❌ 找不到 YOLO 模型檔，程式先停止，不連線 Tello。")
        for label, path in missing:
            print(f"   - {label}: {path}")
        raise FileNotFoundError("YOLO model file missing. Check mission_command.yaml models paths.")

FRAME_W          = CFG.frame_w
FRAME_H          = CFG.frame_h
CONTROL_INTERVAL = CFG.control_interval
box_conf         = CFG.box_conf
qr_conf          = CFG.qr_conf

MIDAS_CFG   = CFG.section("midas")
FORWARD_CFG = CFG.section("forward")
CIRCLE_CFG  = CFG.section("circle")
QR_CFG      = CFG.section("qr_scan")
LOWBAT_CFG  = CFG.section("low_battery")
RETURN_CFG  = CFG.section("return_home")
INSP_CFG    = CFG.section("inspection")

# ──────────────────────────────────────────────────────────
#  狀態定義
# ──────────────────────────────────────────────────────────
class DroneState:
    MIDAS        = "MIDAS"
    FORWARD      = "FORWARD"
    CIRCLE       = "CIRCLE"
    QR_SCAN      = "QR_SCAN"
    AISLE_SCAN   = "AISLE_SCAN"      # 起飛後直接進入
    APPROACH     = "APPROACH"
    AISLE_CHANGE = "AISLE_CHANGE"
    RETURN_HOME  = "RETURN_HOME"

class MissionMode:
    MODE1 = 1
    MODE2 = 2

# ──────────────────────────────────────────────────────────
#  RViz UDP 橋接
# ──────────────────────────────────────────────────────────
class RvizBridge:
    def __init__(self):
        rv = CFG.section("rviz_bridge")
        host = rv.get("host", "127.0.0.1")
        port = rv.get("port", 9999)
        self._sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr   = (host, port)
        self._last_t = 0.0
        self._returning = False
    def send(self, tracker):
        now = time.time()
        if now - self._last_t < 0.1: return
        self._last_t = now
        try:
            payload = json.dumps({"x": round(tracker.x, 1), "z": round(tracker.z, 1), "yaw": round(tracker.yaw, 1), "home": [tracker.home[0], tracker.home[2]], "returning": self._returning}).encode()
            self._sock.sendto(payload, self._addr)
        except Exception: pass
    def set_returning(self, val: bool): self._returning = val
    def close(self):
        try: self._sock.close()
        except Exception: pass

# ──────────────────────────────────────────────────────────
#  飛行軌跡紀錄器
# ──────────────────────────────────────────────────────────
class FlightTracker:
    def __init__(self): self.reset()
    def reset(self):
        self.x = self.y = self.z = self.yaw = 0.0
        self.path: List[tuple] = [(0.0, 0.0, 0.0, False)]
        self.last_time = time.time()
        self.home = (0.0, 0.0, 0.0)
    def reset_pose(self):
        self.x = self.y = self.z = self.yaw = 0.0
        self.path = [(0.0, 0.0, 0.0, False)]
        self.last_time = time.time()
    def update(self, tello: "Tello", is_manual: bool = False):
        now = time.time()
        dt  = now - self.last_time
        self.last_time = now
        if dt <= 0 or dt > 1.0: return
        try:
            vx, vy, vz = float(tello.get_speed_x()), float(tello.get_speed_y()), float(tello.get_speed_z())
            self.yaw = float(tello.get_yaw())
        except Exception: return
        self.z += (-vx) * dt
        self.x += (-vy) * dt
        self.y +=   vz  * dt
        self.path.append((self.x, self.y, self.z, is_manual))
    def distance_to_home(self) -> float:
        return math.sqrt((self.x - self.home[0])**2 + (self.z - self.home[2])**2)
    def draw_minimap(self, frame, size=160, margin=10):
        h, w = frame.shape[:2]
        x0, y0 = w - size - margin, margin
        cv2.rectangle(frame, (x0, y0), (x0+size, y0+size), (30, 30, 30), -1)
        cv2.rectangle(frame, (x0, y0), (x0+size, y0+size), (100, 100, 100), 1)
        if len(self.path) < 2: return frame
        xs, zs = [p[0] for p in self.path], [p[2] for p in self.path]
        span = max(max(xs)-min(xs), max(zs)-min(zs), 100)
        scale = (size - 20) / span
        cx_map, cy_map = x0 + size//2, y0 + size//2
        ox, oz = (max(xs)+min(xs))/2, (max(zs)+min(zs))/2
        def to_px(px, pz): return (int(cx_map+(px-ox)*scale), int(cy_map-(pz-oz)*scale))
        for i in range(1, len(self.path)):
            p1, p2 = to_px(self.path[i-1][0], self.path[i-1][2]), to_px(self.path[i][0], self.path[i][2])
            is_man = len(self.path[i]) > 3 and self.path[i][3]
            cv2.line(frame, p1, p2, (0,140,255) if is_man else (0,200,255), 1)
        cv2.circle(frame, to_px(self.home[0], self.home[2]), 5, (0,255,0), -1)
        cv2.circle(frame, to_px(self.x, self.z), 5, (0,0,255), -1)
        cv2.putText(frame, f"HOME:{self.distance_to_home():.0f}cm", (x0+2, y0+size-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        return frame
    def save_path_csv(self, filename="flight_path.csv"):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x_cm", "y_cm", "z_cm", "is_manual"])
            writer.writerows(self.path)

# ──────────────────────────────────────────────────────────
#  MiDaS 避障巡航控制器
# ──────────────────────────────────────────────────────────
class MidASCruiser:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform
        self.center_q = deque(maxlen=MIDAS_CFG.get("smoothing_window", 5))
        self.state, self.turn_start_time = "FORWARD", 0

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            pred = self.midas(inp)
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
        depth_norm = cv2.normalize(pred.cpu().numpy(), None, 0, 1, cv2.NORM_MINMAX)
        h, w = depth_norm.shape
        center = depth_norm[h//3:h//3*2, w//3:w//3*2]
        self.center_q.append(np.median(center) if center.size else 0.5)
        return depth_norm, float(np.mean(self.center_q))

    def get_control(self, center_depth, now):
        obs_th, clr_th = MIDAS_CFG.get("obstacle_threshold", 0.35), MIDAS_CFG.get("clear_threshold", 0.25)
        turn_d = MIDAS_CFG.get("turn_duration_sec", 1.5)
        fwd_sp, turn_sp = MIDAS_CFG.get("base_forward_speed", 20), MIDAS_CFG.get("turn_speed", 40)
        if self.state == "FORWARD":
            if center_depth > obs_th: self.state, self.turn_start_time = "TURNING", now
        else:
            if now - self.turn_start_time >= turn_d:
                if center_depth < clr_th: self.state = "FORWARD"
                else: self.turn_start_time = now
        return (fwd_sp, 0) if self.state == "FORWARD" else (0, turn_sp)

    def draw_overlay(self, frame, center_depth, fbv, yv):
        color = (0,255,0) if center_depth < MIDAS_CFG.get("clear_threshold", 0.25) else ((0,0,255) if center_depth > MIDAS_CFG.get("obstacle_threshold", 0.35) else (0,255,255))
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//3,h//3), (2*w//3,2*h//3), color, 2)
        cv2.putText(frame, "MODE: MIDAS CRUISE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        return frame

# ──────────────────────────────────────────────────────────
#  目標追蹤基類（YOLO）
# ──────────────────────────────────────────────────────────
class TargetTracker:
    def __init__(self, model_path, config: dict):
        self.model = YOLO(model_path)
        self.config = config
        self.target_lost_time = self.start_time = None
        self.has_target = False
        self.target_center_history = deque(maxlen=5)

    def start(self):
        self.start_time, self.target_lost_time, self.has_target = time.time(), time.time(), False
        self.target_center_history.clear()

    def _is_valid_box(self, x1, y1, x2, y2):
        w, h = x2-x1, y2-y1
        area, aspect = w * h, w / (h + 1e-5)
        if not (self.config.get("min_aspect", 0.3) < aspect < self.config.get("max_aspect", 3.0)): return False
        if area / (FRAME_W * FRAME_H) > self.config.get("max_area_ratio", 0.70): return False
        if area < self.config.get("min_box_area", 8000): return False
        return True

    def detect_target(self, frame, conf=None):
        results = self.model(frame, conf=conf or box_conf, verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            valid = [b for b in results[0].boxes if self._is_valid_box(*map(int, b.xyxy[0]))]
            if not valid:
                if self.target_lost_time is None: self.target_lost_time = time.time()
                self.has_target = False
                return False, 0, 0, 0, None
            best = max(valid, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, best.xyxy[0])
            cx, cy, area = (x1+x2)//2, (y1+y2)//2, (x2-x1)*(y2-y1)
            self.target_center_history.append((cx, cy))
            acx, acy = int(np.mean([c[0] for c in self.target_center_history])), int(np.mean([c[1] for c in self.target_center_history]))
            self.has_target, self.target_lost_time = True, None
            return True, acx, acy, area, (x1,y1,x2,y2)
        if self.target_lost_time is None: self.target_lost_time = time.time()
        self.has_target = False
        return False, 0, 0, 0, None

    def calculate_control(self, tcx, tcy, tarea, tarea_goal):
        ex, ey, ea = tcx - FRAME_W//2, tcy - FRAME_H//2, tarea_goal - tarea
        dz, ms = self.config.get("deadzone", 20), self.config.get("max_speed", 20)
        lr = yaw = ud = fb = 0
        if abs(ex) > dz:
            if abs(ex) > 120: lr = max(-ms, min(ms, int(self.config.get("kp_yaw",0.3)*ex)))
            else: yaw = max(-ms, min(ms, int(self.config.get("kp_yaw",0.3)*ex)))
        if abs(ey) > dz: ud = max(-ms, min(ms, int(-self.config.get("kp_updown",0.3)*ey)))
        if abs(ea) > self.config.get("area_tolerance", 5000): fb = max(-ms, min(ms, int(self.config.get("kp_forward",0.0006)*ea)))
        return lr, fb, ud, yaw

    def should_abort(self):
        return not self.has_target and self.target_lost_time and (time.time() - self.target_lost_time > self.config.get("target_lost_timeout", 2))

    def is_timeout(self):
        return self.start_time and (time.time()-self.start_time > self.config.get("max_execution_time",30))

class ForwardTracker(TargetTracker):
    def __init__(self): super().__init__(CFG.box_model, FORWARD_CFG)
    def process_frame(self, frame):
        det, cx, cy, area, bbox = self.detect_target(frame)
        if det:
            lr, fb, ud, yaw = self.calculate_control(cx, cy, area, self.config.get("target_area",100000))
            return lr, fb, ud, yaw, bbox, area, area >= self.config.get("target_area",100000)
        return 0, 0, 0, 0, None, 0, False

class CircleScanner(TargetTracker):
    def __init__(self):
        super().__init__(CFG.box_model, CIRCLE_CFG)
        self.qr_model = YOLO(CFG.qr_model)
        self.smooth_center = deque(maxlen=3)
    def start(self):
        super().start()
        self.smooth_center.clear()
    def process_frame(self, frame):
        det, cx, cy, area, bbox = self.detect_target(frame)
        qr_det, qr_bbox = False, None
        lr, fb, ud, yaw = CIRCLE_CFG.get("orbit_speed", 7), 0, 0, 0
        if det:
            self.smooth_center.append((cx, cy))
            acx, acy = int(np.mean([c[0] for c in self.smooth_center])), int(np.mean([c[1] for c in self.smooth_center]))
            ex, ey, ea = acx - FRAME_W//2, acy - FRAME_H//2, CIRCLE_CFG.get("target_area",120000) - area
            ymax, udmax = CIRCLE_CFG.get("yaw_correction_speed", 25), CIRCLE_CFG.get("height_correction_speed", 15)
            if abs(ex) > 120:
                lr, yaw = 0, max(-ymax, min(ymax, int(FORWARD_CFG.get("kp_yaw",0.3)*ex)))
            elif abs(ex) > FORWARD_CFG.get("deadzone", 20):
                yaw = max(-ymax, min(ymax, int(FORWARD_CFG.get("kp_yaw",0.3)*ex)))
            if abs(ey) > FORWARD_CFG.get("deadzone", 20): ud = max(-udmax, min(udmax, int(-FORWARD_CFG.get("kp_updown",0.3)*ey*0.5)))
            if abs(ea) > CIRCLE_CFG.get("area_tolerance", 5000):
                fb_limit = CIRCLE_CFG.get("forward_correction_max_speed", FORWARD_CFG.get("max_speed", 20))
                fb = max(-fb_limit, min(fb_limit, int(CIRCLE_CFG.get("kp_forward",0.0006)*ea)))
            
            if bbox:
                x1, y1, x2, y2 = max(0,bbox[0]-50), max(0,bbox[1]-50), min(FRAME_W,bbox[2]+50), min(FRAME_H,bbox[3]+50)
                res = self.qr_model(frame[y1:y2, x1:x2], conf=qr_conf, verbose=False)
                if res[0].boxes is not None and len(res[0].boxes) > 0:
                    best = max(res[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
                    qx1, qy1, qx2, qy2 = map(int, best.xyxy[0])
                    qr_det, qr_bbox = True, (qx1+x1, qy1+y1, qx2+x1, qy2+y1)
        return lr, fb, ud, yaw, bbox, qr_det, qr_bbox
    def is_complete(self):
        return self.start_time is not None and time.time()-self.start_time >= CIRCLE_CFG.get("min_circle_time_sec", 5)

# ──────────────────────────────────────────────────────────
#  QR 解碼輔助
# ──────────────────────────────────────────────────────────
def decode_qr_from_frame(frame, bbox=None) -> tuple:
    def _try_decode(img):
        try:
            barcodes = pyzbar.decode(img)
            if barcodes: return True, barcodes[0].data.decode("utf-8", errors="ignore")
        except Exception: pass
        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            data, _, _ = cv2.QRCodeDetector().detectAndDecode(bgr)
            if data: return True, data
        except Exception: pass
        return False, None

    def _build_methods(img_bgr):
        if img_bgr is None or img_bgr.size == 0: return []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        h, w = gray.shape[:2]
        methods = [gray]
        if min(h, w) < 220:
            scale = max(2, int(360 / (min(h, w) + 1e-5)))
            methods.append(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        methods.append(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

        sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        for g in list(methods):
            try:
                methods.extend([
                    cv2.GaussianBlur(g, (3, 3), 0),
                    cv2.equalizeHist(g),
                    cv2.filter2D(g, -1, sharpening),
                    cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                    cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                    cv2.bitwise_not(g)
                ])
            except Exception: pass
        return methods

    rois = []
    if bbox is not None:
        pad = 60
        roi = frame[max(0, bbox[1]-pad):min(FRAME_H, bbox[3]+pad), max(0, bbox[0]-pad):min(FRAME_W, bbox[2]+pad)]
        if roi.size > 0: rois.append(roi)
    rois.append(frame) 

    for roi in rois:
        for img in _build_methods(roi):
            ok, data = _try_decode(img)
            if ok and data: return True, data
    return False, None

class QRScanner(TargetTracker):
    def __init__(self):
        super().__init__(CFG.qr_model, QR_CFG)
        self.scanned_set = set()
        self.scan_count = self.last_scan_time = self.consecutive_failures = 0
        self.scan_complete = False
        self.csv_file = QR_CFG.get("csv_file", "scanned_codes.csv")
        self.event_csv_file = QR_CFG.get("event_csv_file", "scan_events.csv")
        self.direct_decode_enabled = QR_CFG.get("direct_decode_enabled", True)
        self.direct_decode_interval_sec = QR_CFG.get("direct_decode_interval_sec", 0.25)
        self._last_direct_decode_t = 0.0
        self.last_debug_status = "QR:INIT"
        self.context_provider = None
        self._load_csv()

    def _is_valid_box(self, x1, y1, x2, y2):
        w, h = x2-x1, y2-y1
        area, aspect = w * h, w / (h + 1e-5)
        if not (self.config.get("min_aspect", 0.15) < aspect < self.config.get("max_aspect", 8.0)): return False
        if area / (FRAME_W * FRAME_H) > self.config.get("max_area_ratio", 0.90): return False
        if area < self.config.get("min_box_area", 200): return False
        return True

    def _load_csv(self):
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, "r", newline="", encoding="utf-8") as f:
                    for row in list(csv.reader(f))[1:]:
                        if len(row) >= 2: self.scanned_set.add(row[1])
            except Exception: pass
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["時間", "資料"])
        if not os.path.exists(self.event_csv_file):
            with open(self.event_csv_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "time", "data", "duplicate", "mission_mode", "drone_state",
                    "aisle_no", "face_no", "qr_count", "target_count",
                    "battery_pct", "x_cm", "y_cm", "z_cm", "bbox", "bbox_area"
                ])

    def set_context_provider(self, fn):
        self.context_provider = fn

    def _log_event(self, data, duplicate=False, bbox=None, area=0):
        ctx = {}
        if self.context_provider is not None:
            try: ctx = self.context_provider() or {}
            except Exception: ctx = {}
        with open(self.event_csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data, int(bool(duplicate)),
                ctx.get("mission_mode", ""), ctx.get("drone_state", ""), ctx.get("aisle_no", ""),
                ctx.get("face_no", ""), ctx.get("qr_count", ""), ctx.get("target_count", ""),
                ctx.get("battery_pct", ""), ctx.get("x_cm", ""), ctx.get("y_cm", ""),
                ctx.get("z_cm", ""), str(bbox) if bbox is not None else "", int(area) if area is not None else 0,
            ])

    def _record_decode(self, data, bbox=None, area=0):
        is_dup = data in self.scanned_set
        if not is_dup:
            self.scanned_set.add(data)
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data])
            print(f"✅ 新條碼: {data}")
        self._log_event(data, duplicate=is_dup, bbox=bbox, area=area)
        self.scan_count += 1
        self.scan_complete = True
        self.consecutive_failures = 0
        self.last_debug_status = f"QR:DECODED {data[:18]}"
        return True, data

    def start(self, qr_bbox=None):
        super().start()
        self.scan_complete, self.last_scan_time, self.consecutive_failures, self._last_direct_decode_t = False, 0, 0, 0.0
        self.last_debug_status = "QR:SCANNING"
        if qr_bbox:
            cx, cy = (qr_bbox[0]+qr_bbox[2])//2, (qr_bbox[1]+qr_bbox[3])//2
            self.target_center_history.append((cx,cy))
            self.has_target = True

    def process_frame(self, frame):
        qr_decoded, decoded_data, lr, fb, ud, yaw, reached, now = False, None, 0, 0, 0, 0, False, time.time()
        det, cx, cy, area, bbox = self.detect_target(frame, conf=qr_conf)

        if det:
            self.target_lost_time = None
            self.last_debug_status = f"QR:YOLO area={int(area)} fail={self.consecutive_failures}"
            lr, fb, ud, yaw = self.calculate_control(cx, cy, area, self.config.get("target_area", 60000))
            if area < self.config.get("min_area_before_decode", 40000): fb = self.config.get("max_speed", 15)
            reached = area >= self.config.get("target_area", 60000)

            if now - self.last_scan_time > self.config.get("qr_scan_interval_sec", 0.3):
                ok, data = decode_qr_from_frame(frame, bbox)
                if ok and data: qr_decoded, decoded_data = self._record_decode(data, bbox, area)
                else: self.consecutive_failures += 1; self.last_debug_status = f"QR:YOLO no-decode fail={self.consecutive_failures}"
                self.last_scan_time = now

            return lr, fb, ud, yaw, bbox, area, reached, qr_decoded, decoded_data

        if self.target_lost_time is None: self.target_lost_time = time.time()
        self.has_target = False

        if self.direct_decode_enabled and (now - self._last_direct_decode_t > self.direct_decode_interval_sec):
            ok, data = decode_qr_from_frame(frame, bbox=None)
            self._last_direct_decode_t = now
            if ok and data:
                qr_decoded, decoded_data = self._record_decode(data, None, 0)
                return 0, 0, 0, 0, None, 0, False, qr_decoded, decoded_data

        self.last_debug_status = "QR:NO YOLO / DIRECT SCAN..."
        return 0, 0, 0, 0, None, 0, False, False, None

    def should_abort(self): return False if self.scan_complete else super().should_abort()

    def is_complete(self):
        if self.scan_complete:
            return True
        return bool(self.start_time and (time.time() - self.start_time > self.config.get("max_execution_time", 30)))


# ──────────────────────────────────────────────────────────
#  走道巡檢狀態機（Mode 2）
# ──────────────────────────────────────────────────────────
class AisleInspector:
    def __init__(self, tello, midas, qr_scanner, tracker):
        self.tello      = tello
        self.midas      = midas
        self.qr_scanner = qr_scanner
        self.tracker    = tracker
        
        self._aisle_steps  = INSP_CFG.get("aisle_change", {}).get("steps", [])
        self._target_count = INSP_CFG.get("qr_target_count", 5)
        self._roll_l       = INSP_CFG.get("roll_scan_speed", -12)
        
        self._panel_depth  = INSP_CFG.get("panel_approach_depth", 0.35)
        self._turn_speed   = INSP_CFG.get("turn_180_speed", 35)
        self._turn_tol     = INSP_CFG.get("turn_tolerance_deg", 8)

        # 距離維持配置 (透過 MiDaS PID 控制 fb)
        self._maintain_dist = INSP_CFG.get("maintain_distance_enabled", True)
        self._target_depth  = INSP_CFG.get("target_depth", 0.35)
        self._depth_tol     = INSP_CFG.get("depth_tolerance", 0.03)
        self._depth_kp      = INSP_CFG.get("depth_kp", 100)
        self._max_fb        = INSP_CFG.get("max_fb_speed", 20)
        self._fb_deadband_depth = INSP_CFG.get("fb_deadband_depth", self._depth_tol)
        self._fb_smooth_alpha = INSP_CFG.get("fb_smooth_alpha", 0.20)
        self._fb_last_cmd = 0
        self._disable_height_control = INSP_CFG.get("disable_height_control", True)

        # 高度維持配置 (透過高度計 PID 控制 ud) [su4.3 移植自 main_fly9.6]
        self._backup_altitude = INSP_CFG.get("backup_altitude", 80)
        self._left_offset_before_hold_sec  = INSP_CFG.get("left_offset_before_hold_sec", 0.5)
        self._maintain_height  = INSP_CFG.get("maintain_height_enabled", True)
        self._hold_height_initialized = False
        self._hold_height_target = None
        self._roll_start_time = 0
        self._height_kp        = INSP_CFG.get("height_kp", 0.8)
        self._max_ud           = INSP_CFG.get("max_ud_speed", 20)
        # [su4.4] 高度控制防震盪：小誤差不修正，輸出做平滑
        self._height_deadband_cm = INSP_CFG.get("height_deadband_cm", 6)
        self._height_smooth_alpha = INSP_CFG.get("height_smooth_alpha", 0.25)
        self._base_ud_smooth = 0.0

        # [su4.4] 簡化掃描流程：Mode2 只定時直接解碼，不再每幀 QR YOLO 鎖框修正
        self._simplified_scan = INSP_CFG.get("simplified_qr_scan_enabled", True)
        self._simple_decode_interval = INSP_CFG.get("simplified_decode_interval_sec", 0.8)
        self._last_simple_decode_t = 0.0

        # 物理突變過濾器狀態變數 [su4.3]
        self._last_valid_h = 0.0

        self.reset()

    def reset(self):
        # 起飛後直接進入 ROLL_SCAN
        self._state    = "ROLL_SCAN"
        self._step_idx = 0
        self._step_t   = 0.0
        self._aisle_no = 1
        self._face     = 1
        self._qr_count = 0
        self._turn_target_yaw = None
        self._mission_scanned = set()
        self._last_valid_h    = 0.0  # [su4.3] 突變過濾器重置
        self._fb_last_cmd     = 0
        self._roll_start_time = time.time()
        self._hold_height_initialized = False
        self._hold_height_target = None
        self._base_ud_smooth = 0.0
        self._last_simple_decode_t = 0.0

        # 立刻啟動掃描器
        self.qr_scanner.start()

    def snapshot(self) -> dict:
        return {
            "state": self._state,
            "step_idx": self._step_idx,
            "step_t": self._step_t,
            "aisle_no": self._aisle_no,
            "face": self._face,
            "qr_count": self._qr_count,
            "turn_target_yaw": self._turn_target_yaw,
            "mission_scanned": set(self._mission_scanned),
        }

    def restore(self, snap: dict):
        if not snap:
            return
        self._state    = snap.get("state", "ROLL_SCAN")
        self._step_idx = snap.get("step_idx", 0)
        self._step_t   = snap.get("step_t", 0.0)
        self._aisle_no = snap.get("aisle_no", 1)
        self._face     = snap.get("face", 1)
        self._qr_count = snap.get("qr_count", 0)
        self._turn_target_yaw = snap.get("turn_target_yaw", None)
        self._mission_scanned = set(snap.get("mission_scanned", set()))

    def _get_yaw(self) -> float:
        try:
            return float(self.tello.get_yaw())
        except Exception:
            return 0.0

    def _get_stable_height(self) -> float:
        """安全獲取目前高度，並加入突變過濾(Outlier Rejection)以忽略下方桌子/障礙物。
        [su4.3 移植自 main_fly9.6]
        若瞬間高度突變超過 25cm（0.05 秒內不可能掉落 25cm），
        認定為下方有桌子或貨架底座，忽略此數值，回傳上一次的有效高度。
        """
        try:
            raw_h = float(self.tello.get_height())
            if self._last_valid_h == 0.0:
                self._last_valid_h = raw_h
                return raw_h
            if abs(raw_h - self._last_valid_h) > 25:
                return self._last_valid_h
            self._last_valid_h = raw_h
            return raw_h
        except Exception:
            return float(self._backup_altitude)

    @staticmethod
    def _norm_ang(a: float) -> float:
        while a > 180:
            a -= 360
        while a < -180:
            a += 360
        return a

    def _start_relative_turn(self, delta_deg: float):
        self._turn_target_yaw = self._norm_ang(self._get_yaw() + delta_deg)

    def _run_turn_to_target(self, yaw_speed: int) -> tuple:
        if self._turn_target_yaw is None:
            return 0, True
        err = self._norm_ang(self._turn_target_yaw - self._get_yaw())
        if abs(err) <= self._turn_tol:
            self._turn_target_yaw = None
            return 0, True
        speed = abs(int(yaw_speed))
        cmd = speed if err > 0 else -speed
        return cmd, False

    def process(self, frame, depth_norm, center_depth) -> tuple:
        lr = fb = ud = yaw = 0
        done = False
        status = self._state

        if self._state == "ROLL_SCAN":
            # [su4.4] 簡化掃描：避免 QR YOLO 每幀鎖框造成 lr/yaw/ud 抖動與運算卡頓。
            qr_lr = qr_fb = qr_ud = qr_yaw = 0
            bbox, area, reached, decoded, data = None, 0, False, False, None

            if self._simplified_scan:
                now_scan = time.time()
                if now_scan - self._last_simple_decode_t >= self._simple_decode_interval:
                    ok, decoded_text = decode_qr_from_frame(frame, bbox=None)
                    self._last_simple_decode_t = now_scan
                    if ok and decoded_text:
                        decoded, data = self.qr_scanner._record_decode(decoded_text, bbox=None, area=0)
            else:
                qr_lr, qr_fb, qr_ud, qr_yaw, bbox, area, reached, decoded, data = \
                    self.qr_scanner.process_frame(frame)

            # MiDaS 距離維持控制 (fb)：
            # 加大死區 + 平滑輸出，避免前後距離讀值抖動造成一直抽動。
            base_fb = 0
            if self._maintain_dist:
                err_depth = self._target_depth - center_depth
                if abs(err_depth) > self._fb_deadband_depth:
                    raw_fb = int(err_depth * self._depth_kp)
                    raw_fb = max(-self._max_fb, min(self._max_fb, raw_fb))
                else:
                    raw_fb = 0
                base_fb = int((1.0 - self._fb_smooth_alpha) * self._fb_last_cmd + self._fb_smooth_alpha * raw_fb)
                if abs(base_fb) <= 1:
                    base_fb = 0

                # 掃描時防止一直往後飄：
                # 預設不允許負 fb，避免 MiDaS 誤判太近就一直後退。
                min_fb = int(INSP_CFG.get("scan_min_fb_speed", 0))
                if base_fb < min_fb:
                    base_fb = min_fb

                self._fb_last_cmd = base_fb

            # 高度控制：
            # Mode 2 掃描時完全不使用高度計，也不使用 QR 垂直修正。
            # 讓 Tello 自己維持高度，避免高度計/QR 框抖動造成上下跑。
            base_ud = 0

            # 預設維持向左飄，並借用 base_fb 穩定距離；ud 固定為 0
            lr, fb, ud, yaw = self._roll_l, base_fb, base_ud, 0

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, f"Locked", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 發現目標時，左右/旋轉 交給 QRScanner 鎖定
                # 前後(fb) 依然強制維持 MiDaS 的 base_fb
                # 垂直(ud) 改用高度計 base_ud，丟棄 YOLO 算出的 qr_ud [su4.3]
                lr, ud, yaw = qr_lr, base_ud, qr_yaw

            if decoded and data:
                if data not in self._mission_scanned:
                    self._mission_scanned.add(data)
                    self._qr_count += 1
                    print(f"📦 走道{self._aisle_no}面{self._face} QR "
                          f"[{self._qr_count}/{self._target_count}]: {data}")
                else:
                    print(f"⚠️ 本次任務已掃過，略過：{data}")

                self.qr_scanner.start()
                lr, fb, ud, yaw = self._roll_l, base_fb, base_ud, 0

            if (not self._simplified_scan) and (self.qr_scanner.should_abort() or self.qr_scanner.is_timeout()):
                # 移除右轉回掃，直接重新開始向左掃
                self.qr_scanner.start()
                lr, fb, ud, yaw = self._roll_l, base_fb, base_ud, 0

            if self._qr_count >= self._target_count:
                if self._face == 1:
                    print("✅ 第一面掃完，準備轉向 180 度")
                    self._state = "TURN_OPPOSITE"
                    self._start_relative_turn(-180)
                else:
                    if self._aisle_no == 1:
                        print("✅ 走道1兩面掃完，開始繞到走道2")
                        self._state    = "AISLE_CHANGE"
                        self._step_idx = 0
                        self._step_t   = time.time()
                        self._turn_target_yaw = None
                    else:
                        print("🎉 全部走道掃描完成！")
                        self._state = "DONE"
                lr, fb, ud, yaw = 0, 0, 0, 0

            status = f"ROLL_SCAN A{self._aisle_no}F{self._face} [{self._qr_count}/{self._target_count}]"

        elif self._state == "TURN_OPPOSITE":
            yaw, finished = self._run_turn_to_target(-abs(self._turn_speed))
            status = "TURN_180_TO_OPPOSITE"
            if finished:
                print("↪️ 已完成左轉 180 度，開始靠近對向面板")
                self._state = "APPROACH_PANEL"

        elif self._state == "APPROACH_PANEL":
            fb = MIDAS_CFG.get("base_forward_speed", 20)
            if center_depth >= self._panel_depth:
                print(f"🏗️ 靠近對向面板 depth={center_depth:.3f}，開始掃對向")
                self._face     = 2
                self._qr_count = 0
                self._state    = "ROLL_SCAN"
                self._roll_start_time = time.time()
                self._hold_height_initialized = False
                self._hold_height_target = None
                self._base_ud_smooth = 0.0
                self.qr_scanner.start()
                fb = 0
            status = f"APPROACH_PANEL depth={center_depth:.3f}"

        elif self._state == "AISLE_CHANGE":
            if self._step_idx >= len(self._aisle_steps):
                self._aisle_no = 2
                self._face     = 1
                self._qr_count = 0
                self._step_idx = 0
                print("🔄 進入走道 2")
                self._state = "ROLL_SCAN"
                self._roll_start_time = time.time()
                self._hold_height_initialized = False
                self._hold_height_target = None
                self._base_ud_smooth = 0.0
                self.qr_scanner.start()
            else:
                step     = self._aisle_steps[self._step_idx]
                action   = step.get("action", "")
                speed    = int(step.get("speed", 20))
                duration = float(step.get("duration_sec", 0))
                clr_th   = step.get("midas_clear_threshold", MIDAS_CFG.get("clear_threshold", 0.25))
                target_deg = float(step.get("target_deg", 0))

                if action == "roll_left":
                    lr = -abs(speed)
                    step_done = (duration > 0 and time.time()-self._step_t >= duration) or \
                                (duration == 0 and center_depth < clr_th)
                elif action == "roll_right":
                    lr = abs(speed)
                    step_done = (duration > 0 and time.time()-self._step_t >= duration) or \
                                (duration == 0 and center_depth < clr_th)
                elif action == "obstacle_forward":
                    fb = abs(speed)
                    step_done = center_depth < clr_th
                elif action == "yaw_right":
                    if target_deg > 0:
                        if self._turn_target_yaw is None:
                            self._start_relative_turn(abs(target_deg))
                        yaw, step_done = self._run_turn_to_target(abs(speed))
                    else:
                        yaw = abs(speed)
                        step_done = duration > 0 and time.time()-self._step_t >= duration
                elif action == "yaw_left":
                    if target_deg > 0:
                        if self._turn_target_yaw is None:
                            self._start_relative_turn(-abs(target_deg))
                        yaw, step_done = self._run_turn_to_target(-abs(speed))
                    else:
                        yaw = -abs(speed)
                        step_done = duration > 0 and time.time()-self._step_t >= duration
                else:
                    step_done = True

                if step_done:
                    self._step_idx += 1
                    self._step_t = time.time()
                    self._turn_target_yaw = None

            status = f"AISLE_CHANGE step={self._step_idx}"

        elif self._state == "DONE":
            done = True
            status = "DONE"

        return lr, fb, ud, yaw, status, done

    def get_hud_info(self) -> dict:
        return {
            "internal_state": self._state,
            "aisle_no": self._aisle_no,
            "face_no": self._face,
            "qr_count": self._qr_count,
            "target_count": self._target_count,
        }

# ──────────────────────────────────────────────────────────
#  自動返航控制器
# ──────────────────────────────────────────────────────────
class ReturnHomeController:
    ARRIVE_CM    = RETURN_CFG.get("arrive_radius_cm", 80)
    HOVER_SEC    = RETURN_CFG.get("hover_sec",         2.0)
    SPEED        = RETURN_CFG.get("fly_speed",         50)
    DESCEND_SPD  = RETURN_CFG.get("descend_speed",    -10)
    TARGET_H_CM  = RETURN_CFG.get("target_height_cm", 50)
    YAW_SPEED    = RETURN_CFG.get("yaw_speed",         40)
    RETURN_ALT   = RETURN_CFG.get("return_altitude_cm", 150)

    def __init__(self, tello, tracker):
        self.tello   = tello
        self.tracker = tracker
        self._phase  = "idle"
        self._t      = 0.0
        self._last_yaw_err = 0
        self._yaw_int      = 0

    def start(self):
        dist = self.tracker.distance_to_home()
        print(f"[ReturnHome] 啟動，距起飛點={dist:.0f}cm")
        self._phase = "climb"
        self._t     = time.time()
        self._yaw_int = 0

    def get_rc(self) -> list:
        if self._phase == "idle":
            return [0,0,0,0]

        if self._phase == "climb":
            try:    h = float(self.tello.get_height())
            except: h = 80
            if h < self.RETURN_ALT - 10:
                return [0, 0, RETURN_CFG.get("climb_speed", 20), 0]
            else:
                self._phase = "fly"
                self._t     = time.time()
                return [0,0,0,0]

        if self._phase == "fly":
            dx = self.tracker.home[0] - self.tracker.x
            dz = self.tracker.home[2] - self.tracker.z
            dist = math.sqrt(dx**2+dz**2) * 2.0
            if dist <= self.ARRIVE_CM:
                print("[ReturnHome] 到達起飛點，懸停")
                self._phase = "hover"
                self._t     = time.time()
                return [0,0,0,0]
            tgt_yaw  = math.degrees(math.atan2(dx, dz)) if (abs(dx)>0.1 or abs(dz)>0.1) else 0
            yaw_err  = tgt_yaw - self.tracker.yaw
            while yaw_err > 180:  yaw_err -= 360
            while yaw_err < -180: yaw_err += 360
            p = 0.8*yaw_err
            self._yaw_int += yaw_err*0.01
            self._yaw_int  = max(-100, min(100, self._yaw_int))
            d = 0.1*(yaw_err - self._last_yaw_err)
            yaw_cmd = int(max(-self.YAW_SPEED, min(self.YAW_SPEED, p+0.05*self._yaw_int+d)))
            self._last_yaw_err = yaw_err
            if abs(yaw_err) < 30:
                fb_v = min(self.SPEED, max(10, int(dist/8)))
            else:
                fb_v = 0
            try:    cur_h = float(self.tello.get_height())
            except: cur_h = 80
            ud_v = self.DESCEND_SPD if cur_h > self.TARGET_H_CM else 0
            return [0, fb_v, ud_v, yaw_cmd]

        if self._phase == "hover":
            if time.time()-self._t >= self.HOVER_SEC:
                print("[ReturnHome] 降落")
                self._phase = "land"
                self.tello.land()
            return [0,0,0,0]

        return [0,0,0,0]

    def is_active(self):   return self._phase != "idle"
    def is_landing(self):  return self._phase == "land"

# ──────────────────────────────────────────────────────────
#  主控制器
# ──────────────────────────────────────────────────────────
class TelloMissionController:
    def __init__(self):
        _validate_model_files()
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.tello.set_speed(CFG.get("flight_speed", "tello_default_speed", default=50))
        print(f"🔋 電量: {self.tello.get_battery()}%")

        self.midas       = MidASCruiser()
        self.forward     = ForwardTracker()
        self.circle      = CircleScanner()
        self.qr_scanner  = QRScanner()
        self.tracker     = FlightTracker()
        self.rviz_bridge = RvizBridge()
        self.return_home = ReturnHomeController(self.tello, self.tracker)
        self.inspector   = AisleInspector(self.tello, self.midas, self.qr_scanner, self.tracker)

        mission_cfg = CFG.section("mission")
        raw_mode = CFG.mission_mode
        self.hybrid_enabled = str(raw_mode).lower() == "auto" or mission_cfg.get("hybrid_enabled", False)
        self.auto_switch_enabled = mission_cfg.get("auto_switch_enabled", self.hybrid_enabled)
        self.auto_switch_box_area = mission_cfg.get("auto_switch_box_area", MIDAS_CFG.get("target_found_area", 10000))
        self.auto_switch_hold_frames = mission_cfg.get("auto_switch_hold_frames", 8)
        self.hybrid_mode1_timeout_sec = mission_cfg.get("hybrid_mode1_timeout_sec", 15)
        self.isolated_box_min_area = mission_cfg.get("isolated_box_min_area", self.auto_switch_box_area)
        self.isolated_box_max_area_ratio = mission_cfg.get("isolated_box_max_area_ratio", 0.55)
        self.isolated_box_center_margin_px = mission_cfg.get("isolated_box_center_margin_px", 180)
        self.isolated_box_second_ratio_max = mission_cfg.get("isolated_box_second_ratio_max", 0.60)
        self.isolated_box_single_only = mission_cfg.get("isolated_box_single_only", True)
        self._auto_switch_count = 0
        self._resume_mode2_after_mode1 = False
        self._saved_inspector_state = None
        self._hybrid_mode1_start_t = None

        if str(raw_mode) == "1":
            self.mission_mode = MissionMode.MODE1
        else:
            self.mission_mode = MissionMode.MODE2

        mode_label = "環繞巡檢(1)" if self.mission_mode == MissionMode.MODE1 else "走道巡檢(2)"
        if self.hybrid_enabled:
            mode_label += " + 混合切換(auto)"
        print(f"🎯 任務模式: {mode_label}")

        self.current_state   = (DroneState.MIDAS if self.mission_mode == MissionMode.MODE1
                                else DroneState.AISLE_SCAN)
        self.state_start_t   = time.time()
        self.manual_mode     = False
        self.running         = True
        self.is_flying       = False
        self._scanned_popup  = 0.0

        self._prev_t_key = False
        self._prev_l_key = False
        self._last_bat_check = 0.0
        self._low_battery_triggered = False
        self._alt_next_t = float("inf")
        self._alt_ud_cmd = 0

        pygame.init()
        pygame.display.set_mode((300, 200))
        pygame.display.set_caption("Tello Mission Control")

        self.qr_scanner.set_context_provider(self._get_scan_context)

    def get_keyboard_control(self):
        lr = fb = ud = yv = 0
        manual_active = quit_flag = takeoff_cmd = land_cmd = False
        force_state   = None
        switch_mode   = None

        pygame.event.pump()
        keys = pygame.key.get_pressed()
        SPD  = CFG.get("manual", "keyboard_speed", default=50)

        if keys[pygame.K_w]:      ud = SPD;   manual_active = True
        if keys[pygame.K_s]:      ud = -SPD;  manual_active = True
        if keys[pygame.K_a]:      yv = -SPD;  manual_active = True
        if keys[pygame.K_d]:      yv = SPD;   manual_active = True
        if keys[pygame.K_UP]:     fb = SPD;   manual_active = True
        if keys[pygame.K_DOWN]:   fb = -SPD;  manual_active = True
        if keys[pygame.K_LEFT]:   lr = -SPD;  manual_active = True
        if keys[pygame.K_RIGHT]:  lr = SPD;   manual_active = True
        if keys[pygame.K_SPACE]:  lr = fb = ud = yv = 0; manual_active = True
        if keys[pygame.K_t]:      takeoff_cmd = True
        if keys[pygame.K_l]:      land_cmd    = True
        if keys[pygame.K_ESCAPE]: quit_flag   = True
        if keys[pygame.K_F1]:     switch_mode = MissionMode.MODE1
        if keys[pygame.K_F2]:     switch_mode = MissionMode.MODE2
        if keys[pygame.K_1]:      force_state = DroneState.MIDAS
        if keys[pygame.K_2]:      force_state = DroneState.FORWARD
        if keys[pygame.K_3]:      force_state = DroneState.CIRCLE
        if keys[pygame.K_4]:      force_state = DroneState.QR_SCAN

        return (manual_active, lr, fb, ud, yv,
                quit_flag, force_state, takeoff_cmd, land_cmd, switch_mode)

    def change_state(self, new_state, qr_bbox=None):
        old = self.current_state
        self.current_state = new_state
        self.state_start_t = time.time()
        if new_state == DroneState.RETURN_HOME:
            self.return_home.start()
        if new_state == DroneState.FORWARD:
            self.forward.start()
        elif new_state == DroneState.CIRCLE:
            self.circle.start()
        elif new_state == DroneState.QR_SCAN:
            self.qr_scanner.start(qr_bbox)
        print(f"\n🔄 狀態切換: {old} → {new_state}")

    def _get_random_alt(self) -> int:
        cfg = MIDAS_CFG
        now = time.time()
        if now >= self._alt_next_t:
            self._alt_next_t = now + cfg.get("alt_change_interval_sec", 3.0)
            try:
                h = float(self.tello.get_height())
            except Exception:
                h = 100.0
            if h <= cfg.get("alt_min_cm", 50):
                self._alt_ud_cmd = cfg.get("alt_speed", 12)
            elif h >= cfg.get("alt_max_cm", 180):
                self._alt_ud_cmd = -cfg.get("alt_speed", 12)
            else:
                c = random.randint(0, 2)
                alt_sp = cfg.get("alt_speed", 12)
                self._alt_ud_cmd = alt_sp if c == 0 else (-alt_sp if c == 1 else 0)
        return self._alt_ud_cmd

    def _check_battery(self):
        if hasattr(self, "is_flying") and not self.is_flying:
            return
        now = time.time()
        if now - self._last_bat_check < LOWBAT_CFG.get("check_interval_sec", 5):
            return
        self._last_bat_check = now
        try:
            bat = self.tello.get_battery()
        except Exception:
            return
        if bat <= LOWBAT_CFG.get("threshold_pct", 30) and not self._low_battery_triggered:
            self._low_battery_triggered = True
            print(f"🔋 低電量 {bat}%！自動返航")
            self.change_state(DroneState.RETURN_HOME)

    def _clear_hybrid_flags(self):
        self._auto_switch_count = 0
        self._resume_mode2_after_mode1 = False
        self._saved_inspector_state = None
        self._hybrid_mode1_start_t = None

    def _get_scan_context(self):
        info = self.inspector.get_hud_info()
        try:
            bat = self.tello.get_battery()
        except Exception:
            bat = ""
        return {
            "mission_mode": self.mission_mode,
            "drone_state": self.current_state,
            "aisle_no": info.get("aisle_no", ""),
            "face_no": info.get("face_no", ""),
            "qr_count": info.get("qr_count", ""),
            "target_count": info.get("target_count", ""),
            "battery_pct": bat,
            "x_cm": round(self.tracker.x, 1),
            "y_cm": round(self.tracker.y, 1),
            "z_cm": round(self.tracker.z, 1),
        }

    def _draw_extra_hud(self, frame):
        info = self.inspector.get_hud_info()
        cv2.putText(frame,
            f"Aisle:{info['aisle_no']}  Face:{info['face_no']}  QR:{info['qr_count']}/{info['target_count']}",
            (10, FRAME_H - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 255), 2)
        hybrid_label = "ON" if self.hybrid_enabled else "OFF"
        cv2.putText(frame,
            f"Hybrid:{hybrid_label}  AutoSwitchCnt:{self._auto_switch_count}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 255, 150), 1)
        return frame

    def _detect_isolated_box(self, frame):
        try:
            results = self.forward.model(frame, conf=box_conf, verbose=False)
        except Exception:
            return False, None
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return False, None
        valid = [b for b in results[0].boxes if self.forward._is_valid_box(*map(int, b.xyxy[0]))]
        if not valid:
            return False, None

        boxes = []
        for b in valid:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            boxes.append((area, cx, cy, (x1, y1, x2, y2)))
        boxes.sort(key=lambda x: x[0], reverse=True)
        best_area, best_cx, best_cy, best_bbox = boxes[0]

        if best_area < self.isolated_box_min_area:
            return False, None
        if best_area / float(FRAME_W * FRAME_H) > self.isolated_box_max_area_ratio:
            return False, None
        if abs(best_cx - FRAME_W // 2) > self.isolated_box_center_margin_px:
            return False, None
        if self.isolated_box_single_only and len(boxes) > 1:
            return False, None
        if len(boxes) > 1 and (boxes[1][0] / float(best_area + 1e-6)) > self.isolated_box_second_ratio_max:
            return False, None
        return True, best_bbox

    def _handoff_to_mode1(self, reason="AUTO"):
        if self.current_state == DroneState.RETURN_HOME:
            return
        if not self._resume_mode2_after_mode1:
            self._saved_inspector_state = self.inspector.snapshot()
            self._resume_mode2_after_mode1 = True
        self._hybrid_mode1_start_t = time.time()
        self.mission_mode = MissionMode.MODE1
        print(f"🔀 混合任務切換至 Mode 1（{reason}）")
        self.change_state(DroneState.MIDAS)

    def _resume_mode2(self):
        print("↩️ Mode 1 完成，回到 Mode 2 原走道流程")
        self.mission_mode = MissionMode.MODE2
        if self._saved_inspector_state:
            self.inspector.restore(self._saved_inspector_state)
        self.current_state = DroneState.AISLE_SCAN
        self._auto_switch_count = 0
        self._resume_mode2_after_mode1 = False
        self._saved_inspector_state = None
        self._hybrid_mode1_start_t = None

    def run(self):
        print("\n" + "=" * 55)
        print("Tello 任務控制器 (去爬升與定距版)")
        print("F1=環繞模式  F2=走道巡檢模式")
        print("T=起飛  L=降落  方向鍵=移動  WSAD=升降轉  ESC=停止")
        print("=" * 55)

        frame_reader = self.tello.get_frame_read()
        last_ctrl_t  = time.time()

        try:
            while self.running:
                frame = frame_reader.frame
                if frame is None:
                    time.sleep(0.05)
                    continue
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

                self._check_battery()

                (manual_active, lr, fb, ud, yv,
                 quit_flag, force_state, t_held, l_held,
                 switch_mode) = self.get_keyboard_control()

                takeoff_cmd = t_held and not self._prev_t_key
                land_cmd    = l_held and not self._prev_l_key
                self._prev_t_key = t_held
                self._prev_l_key = l_held

                self.tracker.update(self.tello, is_manual=manual_active)
                self.rviz_bridge.send(self.tracker)

                if quit_flag:
                    print("使用者中斷")
                    break

                if takeoff_cmd and not self.is_flying:
                    print("🛸 手動起飛...")
                    try:
                        self.tello.takeoff()
                        self.is_flying = True
                        stab = CFG.get("takeoff", "stabilize_wait_sec", default=2.5)
                        print(f"⏳ 等待起飛穩定 {stab} 秒...")
                        time.sleep(stab)
                        self.tracker.reset_pose()
                        self._alt_next_t = time.time() + 5.0
                        self._alt_ud_cmd = 0
                        self._auto_switch_count = 0
                        if self.mission_mode == MissionMode.MODE2:
                            self.inspector.reset()
                            self.current_state = DroneState.AISLE_SCAN
                        else:
                            self.current_state = DroneState.MIDAS
                        print("✅ 起飛完成，正式進入自動巡檢")
                    except Exception as e:
                        self.is_flying = False
                        print(f"❌ 起飛失敗: {e}")
                    continue

                if land_cmd and self.is_flying:
                    self._alt_ud_cmd = 0
                    self._alt_next_t = float("inf")
                    self.is_flying = False
                    try:
                        self.tello.send_rc_control(0, 0, 0, 0)
                        time.sleep(0.3)
                        self.tello.land()
                    except Exception as e:
                        print(f"⚠️ 降落指令失敗: {e}")

                if switch_mode is not None and switch_mode != self.mission_mode:
                    self._clear_hybrid_flags()
                    self.mission_mode = switch_mode
                    label = "環繞巡檢(1)" if switch_mode == MissionMode.MODE1 else "走道巡檢(2)"
                    print(f"🔀 切換模式 → {label}")
                    if switch_mode == MissionMode.MODE1:
                        self.change_state(DroneState.MIDAS)
                    else:
                        self.inspector.reset()
                        self.current_state = DroneState.AISLE_SCAN

                if force_state:
                    self.change_state(force_state)

                if self.is_flying:
                    if not manual_active:
                        control_cmd = [0, 0, 0, 0]

                        if self.mission_mode == MissionMode.MODE1:
                            control_cmd = self._run_mode1(frame)
                        elif self.mission_mode == MissionMode.MODE2:
                            control_cmd, frame = self._run_mode2(frame)

                        if self.current_state == DroneState.RETURN_HOME:
                            control_cmd = self.return_home.get_rc()
                            self.rviz_bridge.set_returning(True)
                            if self.return_home.is_landing():
                                self.is_flying = False
                        else:
                            self.rviz_bridge.set_returning(False)

                        now = time.time()
                        if now - last_ctrl_t >= CONTROL_INTERVAL:
                            self.tello.send_rc_control(*control_cmd)
                            last_ctrl_t = now
                    else:
                        self.tello.send_rc_control(lr, fb, ud, yv)
                        cv2.putText(frame, "MANUAL MODE", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "STANDBY (Press T to Takeoff)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                self.tracker.draw_minimap(frame)
                cv2.putText(frame,
                    f"State:{self.current_state}  Mode:{'1-Circle' if self.mission_mode == 1 else '2-Aisle'}",
                    (10, FRAME_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, f"Bat:{self.tello.get_battery()}%",
                    (FRAME_W - 130, FRAME_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "T:Takeoff  L:Land  F1/F2:Mode",
                    (10, FRAME_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                self._draw_extra_hud(frame)

                if time.time() < self._scanned_popup:
                    h_, w_ = frame.shape[:2]
                    ov = frame.copy()
                    cv2.rectangle(ov, (w_ // 2 - 180, h_ // 2 - 60), (w_ // 2 + 180, h_ // 2 + 60), (0, 200, 0), -1)
                    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, "SCANNED !", (w_ // 2 - 130, h_ // 2 + 20),
                                cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 3)

                cv2.imshow("Tello Mission Control", frame)
                if cv2.waitKey(1) == 27:
                    break

        except Exception as e:
            print(f"❌ 錯誤: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.cleanup()

    def _run_mode1(self, frame) -> list:
        ctrl = [0, 0, 0, 0]

        if self.current_state == DroneState.MIDAS:
            depth_norm, center_depth = self.midas.process_frame(frame)
            fbv, yv = self.midas.get_control(center_depth, time.time())
            ud_m    = self._get_random_alt()
            ctrl    = [0, fbv, ud_m, yv]

            depth_disp = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("Depth Map", depth_disp)

            results = self.forward.model(frame, conf=box_conf, verbose=False)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                valid = [b for b in results[0].boxes
                         if self.forward._is_valid_box(*map(int, b.xyxy[0]))]
                if valid:
                    best = max(valid, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                    area = (best.xyxy[0][2] - best.xyxy[0][0]) * (best.xyxy[0][3] - best.xyxy[0][1])
                    if area > MIDAS_CFG.get("target_found_area", 10000):
                        self.change_state(DroneState.FORWARD)

            frame[:] = self.midas.draw_overlay(frame, center_depth, fbv, yv)

            if self._resume_mode2_after_mode1 and self._hybrid_mode1_start_t is not None:
                if time.time() - self._hybrid_mode1_start_t > self.hybrid_mode1_timeout_sec:
                    self._resume_mode2()
                    return [0, 0, 0, 0]

        elif self.current_state == DroneState.FORWARD:
            lr, fb, ud, yaw, bbox, area, reached = self.forward.process_frame(frame)
            ctrl = [lr, fb, ud, yaw]
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "MODE: FORWARD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if reached:
                self.tello.send_rc_control(0, 0, 0, 0); time.sleep(CIRCLE_CFG.get("pre_circle_hover_sec", 1.0))
                self.change_state(DroneState.CIRCLE)
            elif self.forward.should_abort() or self.forward.is_timeout():
                if self._resume_mode2_after_mode1:
                    self._resume_mode2()
                    return [0, 0, 0, 0]
                self.change_state(DroneState.MIDAS)

        elif self.current_state == DroneState.CIRCLE:
            lr, fb, ud, yaw, bbox, qr_det, qr_bbox = self.circle.process_frame(frame)
            ctrl = [lr, fb, ud, yaw]
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if qr_det and qr_bbox:
                qx1, qy1, qx2, qy2 = qr_bbox
                cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (255, 255, 0), 3)
            cv2.putText(frame, "MODE: CIRCLE SCAN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if qr_det and qr_bbox and self.circle.is_complete():
                self.tello.send_rc_control(0, 0, 0, 0); time.sleep(QR_CFG.get("pre_qr_hover_sec", 1.0))
                self.change_state(DroneState.QR_SCAN, qr_bbox)
            elif self.circle.should_abort() or self.circle.is_timeout():
                if self._resume_mode2_after_mode1:
                    self._resume_mode2()
                    return [0, 0, 0, 0]
                self.change_state(DroneState.MIDAS)

        elif self.current_state == DroneState.QR_SCAN:
            lr, fb, ud, yaw, bbox, area, reached, decoded, data = self.qr_scanner.process_frame(frame)
            ctrl = [lr, fb, ud, yaw]
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            if decoded and data:
                cv2.putText(frame, f"SCANNED:{data}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self._scanned_popup = time.time() + 3.0
            cv2.putText(frame, "MODE: QR SCAN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, self.qr_scanner.last_debug_status, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            if self.qr_scanner.is_complete() or self.qr_scanner.should_abort():
                self.tello.send_rc_control(0, 0, QR_CFG.get("after_scan_up_speed", 50), 0); time.sleep(QR_CFG.get("after_scan_up_sec", 1.0))
                if self._resume_mode2_after_mode1:
                    self._resume_mode2()
                    return [0, 0, 0, 0]
                self.change_state(DroneState.MIDAS)

        return ctrl

    def _run_mode2(self, frame) -> tuple:
        if self.current_state == DroneState.RETURN_HOME:
            return [0, 0, 0, 0], frame

        depth_norm, center_depth = self.midas.process_frame(frame)
        depth_disp = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("Depth Map", depth_disp)

        if self.hybrid_enabled and self.auto_switch_enabled and self.inspector._state in ("ROLL_SCAN",):
            detected, bbox = self._detect_isolated_box(frame)
            if detected:
                self._auto_switch_count += 1
            else:
                self._auto_switch_count = 0
            if self._auto_switch_count >= self.auto_switch_hold_frames:
                self._auto_switch_count = 0
                self._handoff_to_mode1(reason="AUTO isolated-box")
                return [0, 0, 0, 0], frame

        lr, fb, ud, yaw, status, done = self.inspector.process(frame, depth_norm, center_depth)

        if done:
            print("🏁 全部掃完，自動返航")
            self.change_state(DroneState.RETURN_HOME)
            return [0, 0, 0, 0], frame

        cv2.putText(frame, f"AISLE: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(frame, f"Depth:{center_depth:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
        cv2.putText(frame, self.qr_scanner.last_debug_status, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)

        return [lr, fb, ud, yaw], frame

    def cleanup(self):
        print("\n🧹 清理中...")
        try:
            self.rviz_bridge.close()
        except Exception:
            pass
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception:
            pass
        try:
            self.tracker.save_path_csv()
        except Exception as e:
            print(f"⚠️ 軌跡儲存失敗: {e}")
        try:
            self.tello.streamoff()
        except Exception:
            pass
        try:
            pygame.quit()
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("✅ 程式結束")

# ──────────────────────────────────────────────────────────
#  程式入口
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    controller = TelloMissionController()
    controller.run()