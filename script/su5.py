"""
su4.4_optimized.py  –  Tello 雙模式任務控制系統 (高度閉迴路 + QR定距 + 防漏掃版)
=============================================
功能 1（Mode 1）: 環繞巡檢  — MIDAS巡航 → FORWARD接近 → CIRCLE環繞 → QR_SCAN掃碼
功能 2（Mode 2）: 走道巡檢  — 起飛後直接 roll左掃描QR(5個) → MiDaS判定靠近對向面板
                             → 左轉180度 → 繼續roll左掃描(5個) → 繞到第二走道 → 重複 → 回航降落

修改紀錄:
  [su4.1] 移除起飛爬升 (CLIMB) 邏輯，起飛後直接進入 ROLL_SCAN。
  [su4.1] 移除右偏回朔 (RESCAN_RIGHT) 邏輯，若掃描丟失則直接繼續向左掃描。
  [su4.1] 導入 MiDaS 定距平移邏輯：SCAN 階段不再依賴 YOLO 面積決定前後距離。
  [su4.1-yaml] 將飛行速度、控制參數、模型路徑、任務參數集中到 mission_command.yaml。
  [su4.4-opt] 針對高度不穩、QR距離不穩、漏掃進行優化：
          - ROLL_SCAN 新增高度閉迴路，預設鎖定起飛後高度或 YAML 指定 target_height_cm。
          - MiDaS 定距改為 EMA + PD + 速度斜率限制，減少前後抖動。
          - QR 鎖定後改用 QR bbox 面積做近距離定距，並短暫停掃提高解碼成功率。
          - QR 解碼支援多候選框、detectAndDecodeMulti、CLAHE/銳化/二值化等多方法。
          - 自動控制指令加入平滑限制，避免 Tello 因瞬間大幅 rc 變化而高度/距離晃動。
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
YAML_PATH = os.path.join(BASE_DIR, "mission_su5.yaml")

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

def _clamp(value, low, high):
    return max(low, min(high, value))

def _rate_limit(value, last_value, max_delta):
    if max_delta is None or max_delta <= 0:
        return value
    return _clamp(value, last_value - max_delta, last_value + max_delta)

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

def _validate_model_files():
    missing = []
    for label, path in [("box_model_path", CFG.box_model), ("barcode_model_path", CFG.qr_model)]:
        if not path or not os.path.exists(path):
            missing.append((label, path))
    if missing:
        print("\n❌ 找不到 YOLO 模型檔，程式先停止，不連線 Tello。")
        for label, path in missing:
            print(f"   - {label}: {path}")
        raise FileNotFoundError("YOLO model file missing. Check mission_su5.yaml models paths.")

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
    """多策略 QR 解碼。
    1) 先解 YOLO bbox 附近 ROI，再解全畫面。
    2) 同時使用 pyzbar、OpenCV detectAndDecode、detectAndDecodeMulti。
    3) 對 ROI 做放大、CLAHE、銳化、Otsu、自適應二值化，提高模糊/偏暗時的成功率。
    """
    def _try_decode(img):
        try:
            barcodes = pyzbar.decode(img)
            if barcodes:
                # 多個條碼時，優先取面積最大的，減少誤讀邊角小雜訊。
                best = max(barcodes, key=lambda b: b.rect.width * b.rect.height)
                data = best.data.decode("utf-8", errors="ignore").strip()
                if data:
                    return True, data
        except Exception:
            pass

        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(bgr)
            if data:
                return True, data.strip()

            # OpenCV 支援一次解多個 QR，適合畫面中同時出現兩張以上 QR 的情況。
            ok_multi, decoded_info, _, _ = detector.detectAndDecodeMulti(bgr)
            if ok_multi:
                for d in decoded_info:
                    if d and str(d).strip():
                        return True, str(d).strip()
        except Exception:
            pass
        return False, None

    def _build_methods(img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        h, w = gray.shape[:2]
        methods = [gray]

        # QR 在畫面偏小時，先放大再解碼；過度放大會拖慢，所以限制倍率。
        min_side = max(1, min(h, w))
        scale = 1
        if min_side < 160:
            scale = 3
        elif min_side < 260:
            scale = 2
        if scale > 1:
            methods.append(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))

        # 固定再補一個 2x 版本，對輕微模糊或遠距 QR 有幫助。
        methods.append(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

        sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for g in list(methods):
            try:
                blur = cv2.GaussianBlur(g, (3, 3), 0)
                eq = cv2.equalizeHist(g)
                cl = clahe.apply(g)
                sharp = cv2.filter2D(g, -1, sharpening)
                otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                adap = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
                morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                methods.extend([blur, eq, cl, sharp, otsu, adap, morph, cv2.bitwise_not(otsu)])
            except Exception:
                pass
        return methods

    rois = []
    if bbox is not None:
        # 增加兩種 padding：小 ROI 解得快，大 ROI 保留 finder pattern 周圍資訊。
        for pad in (50, 90):
            roi = frame[max(0, bbox[1]-pad):min(FRAME_H, bbox[3]+pad),
                        max(0, bbox[0]-pad):min(FRAME_W, bbox[2]+pad)]
            if roi.size > 0:
                rois.append(roi)
    rois.append(frame)

    for roi in rois:
        for img in _build_methods(roi):
            ok, data = _try_decode(img)
            if ok and data:
                return True, data
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

        # 防漏掃：保留短時間 bbox，降低 YOLO 單幀漏檢造成的掃描中斷。
        self.bbox_hold_sec = QR_CFG.get("bbox_hold_sec", 0.35)
        self.last_bbox = None
        self.last_bbox_time = 0.0
        self.last_area = 0
        self.last_all_boxes = []

        # 同一張 QR 剛解過時先冷卻，避免一直重複讀同一張導致後面的 QR 被漏掉。
        self.same_data_cooldown_sec = QR_CFG.get("same_data_cooldown_sec", 1.2)
        self._recent_decode_t = {}
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

    def _can_emit_data(self, data: str, now: float) -> bool:
        last_t = self._recent_decode_t.get(data, 0.0)
        if now - last_t < self.same_data_cooldown_sec:
            return False
        self._recent_decode_t[data] = now
        return True

    def _detect_all_targets(self, frame):
        try:
            results = self.model(frame, conf=qr_conf, verbose=False)
        except Exception:
            return []
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        detections = []
        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            if not self._is_valid_box(x1, y1, x2, y2):
                continue
            area = (x2 - x1) * (y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append({"bbox": (x1, y1, x2, y2), "cx": cx, "cy": cy, "area": area})

        # 面積大的通常比較容易解；但保留全部，避免只盯著一個重複 QR。
        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections[:int(QR_CFG.get("max_decode_candidates", 4))]

    def process_frame(self, frame):
        qr_decoded, decoded_data, lr, fb, ud, yaw, reached, now = False, None, 0, 0, 0, 0, False, time.time()
        detections = self._detect_all_targets(frame)
        det = len(detections) > 0
        if det:
            best = detections[0]
            bbox, area = best["bbox"], best["area"]
            cx, cy = best["cx"], best["cy"]
            self.last_bbox, self.last_bbox_time, self.last_area = bbox, now, area
            self.last_all_boxes = [d["bbox"] for d in detections]
        elif self.last_bbox is not None and now - self.last_bbox_time <= self.bbox_hold_sec:
            # 短暫沿用上一幀 bbox，避免 QR 還在畫面中但 YOLO 偶發漏檢。
            bbox, area = self.last_bbox, self.last_area
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            det = True
        else:
            bbox, area, cx, cy = None, 0, 0, 0

        if det:
            self.target_lost_time = None
            self.has_target = True
            self.last_debug_status = f"QR:YOLO area={int(area)} cand={len(detections) if detections else 1} fail={self.consecutive_failures}"
            lr, fb, ud, yaw = self.calculate_control(cx, cy, area, self.config.get("target_area", 60000))
            if area < self.config.get("min_area_before_decode", 40000):
                fb = self.config.get("max_speed", 15)
            reached = area >= self.config.get("target_area", 60000)

            if now - self.last_scan_time > self.config.get("qr_scan_interval_sec", 0.22):
                decode_targets = detections if detections else [{"bbox": bbox, "area": area}]
                found_any = False
                for d in decode_targets:
                    dbbox, darea = d["bbox"], d["area"]
                    ok, data = decode_qr_from_frame(frame, dbbox)
                    if ok and data:
                        found_any = True
                        if self._can_emit_data(data, now):
                            qr_decoded, decoded_data = self._record_decode(data, dbbox, darea)
                            bbox, area = dbbox, darea
                            break
                if not found_any:
                    self.consecutive_failures += 1
                    self.last_debug_status = f"QR:YOLO no-decode fail={self.consecutive_failures}"
                elif not qr_decoded:
                    self.last_debug_status = "QR:decoded but cooldown suppress"
                self.last_scan_time = now

            return lr, fb, ud, yaw, bbox, area, reached, qr_decoded, decoded_data

        if self.target_lost_time is None: self.target_lost_time = time.time()
        self.has_target = False

        if self.direct_decode_enabled and (now - self._last_direct_decode_t > self.direct_decode_interval_sec):
            ok, data = decode_qr_from_frame(frame, bbox=None)
            self._last_direct_decode_t = now
            if ok and data and self._can_emit_data(data, now):
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

        # QR 視覺輔助定高：不使用高度計，只在看見 QR 時依 bbox 上下位置微調 ud。
        self._qr_height_enabled = INSP_CFG.get("qr_height_enabled", True)
        self._qr_height_deadzone = INSP_CFG.get("qr_height_deadzone", 45)
        self._qr_height_kp = INSP_CFG.get("qr_height_kp", 0.12)
        self._qr_height_max_ud = INSP_CFG.get("qr_height_max_ud", 10)

        # 高度閉迴路：target_height_cm <= 0 時，鎖定 reset 當下 Tello 高度。
        self._height_hold_enabled = INSP_CFG.get("height_hold_enabled", True)
        self._target_height_cfg_cm = INSP_CFG.get("target_height_cm", 0)
        self._height_deadzone_cm = INSP_CFG.get("height_deadzone_cm", 8)
        self._height_kp = INSP_CFG.get("height_kp", 0.35)
        self._height_kd = INSP_CFG.get("height_kd", 0.08)
        self._height_max_ud = INSP_CFG.get("height_max_ud", 14)
        self._height_alpha = INSP_CFG.get("height_smoothing_alpha", 0.35)
        self._height_target_cm = None
        self._height_ema = None
        self._last_height_err = 0.0
        self._last_height_t = 0.0
        self._last_ud_cmd = 0

        # MiDaS 定距改為平滑 PD，避免 normalized depth 每幀跳動造成前後抽動。
        self._depth_alpha = INSP_CFG.get("depth_smoothing_alpha", 0.35)
        self._depth_kd = INSP_CFG.get("depth_kd", 25)
        self._max_fb_delta = INSP_CFG.get("max_fb_delta", 5)
        self._depth_ema = None
        self._last_depth_err = 0.0
        self._last_depth_t = 0.0
        self._last_fb_cmd = 0

        # QR 鎖定/防漏掃：看到 QR 後短暫停住或慢速修正，優先解碼。
        self._qr_lock_hold_sec = INSP_CFG.get("qr_lock_hold_sec", 0.75)
        self._qr_lock_until = 0.0
        self._qr_lock_lr_max = INSP_CFG.get("qr_lock_lr_max", 6)
        self._qr_lock_yaw_max = INSP_CFG.get("qr_lock_yaw_max", 10)
        self._post_decode_pause_sec = INSP_CFG.get("post_decode_pause_sec", 0.35)
        self._pause_until = 0.0

        # QR 近距離時使用 bbox 面積定距，比 MiDaS 全畫面 normalized depth 更貼近 QR 本身。
        self._qr_distance_enabled = INSP_CFG.get("qr_distance_enabled", True)
        self._qr_target_area = INSP_CFG.get("qr_target_area", QR_CFG.get("target_area", 60000))
        self._qr_area_tolerance = INSP_CFG.get("qr_area_tolerance", QR_CFG.get("area_tolerance", 6500))
        self._qr_distance_kp = INSP_CFG.get("qr_distance_kp", QR_CFG.get("kp_forward", 0.00045))
        self._qr_distance_max_fb = INSP_CFG.get("qr_distance_max_fb", 14)

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
        # 立刻啟動掃描器
        self.qr_scanner.start()
        self._reset_filters()

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


    def _reset_filters(self):
        self._depth_ema = None
        self._last_depth_err = 0.0
        self._last_depth_t = 0.0
        self._last_fb_cmd = 0
        self._height_ema = None
        self._last_height_err = 0.0
        self._last_height_t = 0.0
        self._last_ud_cmd = 0
        self._qr_lock_until = 0.0
        self._pause_until = 0.0
        self._set_height_target()

    def _set_height_target(self):
        cfg_target = _safe_float(self._target_height_cfg_cm, 0.0)
        if cfg_target > 0:
            self._height_target_cm = cfg_target
            return
        try:
            h = float(self.tello.get_height())
        except Exception:
            h = 100.0
        # Tello 剛建立物件或還沒起飛時可能回 0，這時先給安全預設值。
        if h < 30:
            h = INSP_CFG.get("fallback_target_height_cm", 100)
        self._height_target_cm = h

    def _calc_height_hold_ud(self) -> int:
        if not self._height_hold_enabled:
            return 0
        if self._height_target_cm is None:
            self._set_height_target()
        now = time.time()
        try:
            h = float(self.tello.get_height())
        except Exception:
            return 0
        if h <= 0:
            return 0

        if self._height_ema is None:
            self._height_ema = h
        else:
            a = _clamp(float(self._height_alpha), 0.05, 1.0)
            self._height_ema = a * h + (1.0 - a) * self._height_ema

        err = float(self._height_target_cm) - self._height_ema
        if abs(err) <= self._height_deadzone_cm:
            desired = 0
        else:
            dt = max(1e-3, now - self._last_height_t) if self._last_height_t else 0.05
            derr = (err - self._last_height_err) / dt
            desired = int(self._height_kp * err + self._height_kd * derr)

        desired = int(_clamp(desired, -self._height_max_ud, self._height_max_ud))
        desired = int(_rate_limit(desired, self._last_ud_cmd, INSP_CFG.get("max_ud_delta", 4)))
        self._last_height_err = err
        self._last_height_t = now
        self._last_ud_cmd = desired
        return desired

    def _calc_depth_fb(self, center_depth: float) -> int:
        if not self._maintain_dist:
            return 0
        now = time.time()
        d = _safe_float(center_depth, self._target_depth)
        if self._depth_ema is None:
            self._depth_ema = d
        else:
            a = _clamp(float(self._depth_alpha), 0.05, 1.0)
            self._depth_ema = a * d + (1.0 - a) * self._depth_ema

        err_depth = self._target_depth - self._depth_ema
        if abs(err_depth) <= self._depth_tol:
            desired = 0
        else:
            dt = max(1e-3, now - self._last_depth_t) if self._last_depth_t else 0.05
            derr = (err_depth - self._last_depth_err) / dt
            desired = int(self._depth_kp * err_depth + self._depth_kd * derr)

        desired = int(_clamp(desired, -self._max_fb, self._max_fb))
        desired = int(_rate_limit(desired, self._last_fb_cmd, self._max_fb_delta))
        self._last_depth_err = err_depth
        self._last_depth_t = now
        self._last_fb_cmd = desired
        return desired

    def _calc_qr_distance_fb(self, area: int, fallback_fb: int) -> int:
        if (not self._qr_distance_enabled) or area <= 0:
            return fallback_fb
        err_area = float(self._qr_target_area) - float(area)
        if abs(err_area) <= self._qr_area_tolerance:
            desired = 0
        else:
            desired = int(self._qr_distance_kp * err_area)
        desired = int(_clamp(desired, -self._qr_distance_max_fb, self._qr_distance_max_fb))
        # 和 MiDaS 做少量融合，避免 QR bbox 抖動時前後突變。
        blended = int(0.75 * desired + 0.25 * fallback_fb)
        out = int(_rate_limit(blended, self._last_fb_cmd, self._max_fb_delta))
        self._last_fb_cmd = out
        return out

    def _calc_qr_lock_lr_yaw(self, bbox, qr_yaw: int):
        if bbox is None:
            return self._roll_l, 0
        x1, _, x2, _ = bbox
        cx = (x1 + x2) // 2
        ex = cx - (FRAME_W // 2)
        if abs(ex) < 45:
            lr = 0
        else:
            lr = int(_clamp(0.05 * ex, -self._qr_lock_lr_max, self._qr_lock_lr_max))
        yaw = int(_clamp(qr_yaw, -self._qr_lock_yaw_max, self._qr_lock_yaw_max))
        return lr, yaw

    def _calc_qr_height_ud(self, bbox) -> int:
        """用 QR 在畫面中的上下位置做相對定高。
        QR 太靠上：往上修正；QR 太靠下：往下修正。
        沒有 QR 時不修正，避免掃描途中上下亂跑。
        """
        if (not self._qr_height_enabled) or bbox is None:
            return 0
        _, y1, _, y2 = bbox
        qr_cy = (y1 + y2) // 2
        err_y = qr_cy - (FRAME_H // 2)
        if abs(err_y) <= self._qr_height_deadzone:
            return 0
        ud = int(-self._qr_height_kp * err_y)
        return max(-self._qr_height_max_ud, min(self._qr_height_max_ud, ud))

    def process(self, frame, depth_norm, center_depth) -> tuple:
        lr = fb = ud = yaw = 0
        done = False
        status = self._state

        if self._state == "ROLL_SCAN":
            now = time.time()
            status = f"ROLL_SCAN A{self._aisle_no}F{self._face} [{self._qr_count}/{self._target_count}]"

            # 掃到 QR 後短暫停住，讓機身穩定，也避免同一張 QR 重複被讀到太多次。
            if now < self._pause_until:
                ud = self._calc_height_hold_ud()
                return 0, 0, ud, 0, status + " PAUSE", False

            qr_lr, qr_fb, qr_ud, qr_yaw, bbox, area, reached, decoded, data = \
                self.qr_scanner.process_frame(frame)

            # 高度閉迴路：沒有 QR 時也會定高，不再讓 ud 長時間固定 0。
            base_ud = self._calc_height_hold_ud()

            # MiDaS 定距：使用 EMA + PD + 指令斜率限制，降低前後距離抖動。
            base_fb = self._calc_depth_fb(center_depth)

            # 預設繼續向左掃描，同時維持高度與距離。
            lr, fb, ud, yaw = self._roll_l, base_fb, base_ud, 0

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, "QR LOCK", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 看到 QR 時先鎖住一小段時間，不再高速 roll 掠過，降低漏掃。
                self._qr_lock_until = max(self._qr_lock_until, now + self._qr_lock_hold_sec)

                qr_height_ud = self._calc_qr_height_ud(bbox)
                qr_dist_fb = self._calc_qr_distance_fb(area, base_fb)
                lock_lr, lock_yaw = self._calc_qr_lock_lr_yaw(bbox, qr_yaw)

                if now < self._qr_lock_until and not decoded:
                    lr = lock_lr
                    yaw = lock_yaw
                else:
                    lr = qr_lr
                    yaw = qr_yaw

                fb = qr_dist_fb
                ud = int(_clamp(base_ud + qr_height_ud, -self._height_max_ud, self._height_max_ud))

            if decoded and data:
                if data not in self._mission_scanned:
                    self._mission_scanned.add(data)
                    self._qr_count += 1
                    self._pause_until = now + self._post_decode_pause_sec
                    print(f"📦 走道{self._aisle_no}面{self._face} QR "
                          f"[{self._qr_count}/{self._target_count}]: {data}")
                else:
                    # 本次任務已讀過的 QR 不再讓機身停太久，避免卡在同一張上。
                    print(f"⚠️ 本次任務已掃過，繼續往左略過：{data}")

                self.qr_scanner.start()
                self._qr_lock_until = 0.0
                lr, fb, ud, yaw = self._roll_l, base_fb, base_ud, 0

            if self.qr_scanner.should_abort() or self.qr_scanner.is_timeout():
                # 不做大幅右回掃；只重啟掃描器並保持穩定掃描速度。
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

        elif self._state == "TURN_OPPOSITE":
            yaw, finished = self._run_turn_to_target(-abs(self._turn_speed))
            status = "TURN_180_TO_OPPOSITE"
            if finished:
                print("↪️ 已完成左轉 180 度，開始靠近對向面板")
                self._state = "APPROACH_PANEL"

        elif self._state == "APPROACH_PANEL":
            # 靠近對向面板時也維持高度，避免轉完後高度漂移造成第一張 QR 漏掃。
            ud = self._calc_height_hold_ud()
            fb = min(MIDAS_CFG.get("base_forward_speed", 20), max(8, abs(self._calc_depth_fb(center_depth))))
            if center_depth >= self._panel_depth:
                print(f"🏗️ 靠近對向面板 depth={center_depth:.3f}，開始掃對向")
                self._face     = 2
                self._qr_count = 0
                self._state    = "ROLL_SCAN"
                self.qr_scanner.start()
                self._reset_filters()
                fb = 0
            status = f"APPROACH_PANEL depth={center_depth:.3f}"

        elif self._state == "AISLE_CHANGE":
            ud = self._calc_height_hold_ud()
            if self._step_idx >= len(self._aisle_steps):
                self._aisle_no = 2
                self._face     = 1
                self._qr_count = 0
                self._step_idx = 0
                print("🔄 進入走道 2")
                self._state = "ROLL_SCAN"
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
        self._last_auto_cmd = [0, 0, 0, 0]
        self._max_rc_delta = CFG.get("control", "max_rc_delta", default=8)

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

    def _get_mode1_height_hold(self) -> int:
        """Mode 1 原本會隨機上下飛，容易被誤認為高度不穩。
        預設改為定高；若 YAML 設 midas.random_alt_enabled: true 才使用舊隨機高度。
        """
        if not hasattr(self, "_mode1_height_ema"):
            self._mode1_height_ema = None
            self._mode1_last_err = 0.0
            self._mode1_last_t = 0.0
            self._mode1_last_ud = 0
        target_h = _safe_float(MIDAS_CFG.get("target_height_cm", INSP_CFG.get("target_height_cm", 100)), 100)
        if target_h <= 0:
            target_h = 100
        try:
            h = float(self.tello.get_height())
        except Exception:
            return 0
        if h <= 0:
            return 0
        a = _clamp(MIDAS_CFG.get("height_smoothing_alpha", 0.35), 0.05, 1.0)
        self._mode1_height_ema = h if self._mode1_height_ema is None else a * h + (1 - a) * self._mode1_height_ema
        err = target_h - self._mode1_height_ema
        deadzone = MIDAS_CFG.get("height_deadzone_cm", 10)
        if abs(err) <= deadzone:
            desired = 0
        else:
            now = time.time()
            dt = max(1e-3, now - self._mode1_last_t) if self._mode1_last_t else 0.05
            derr = (err - self._mode1_last_err) / dt
            desired = int(MIDAS_CFG.get("height_kp", 0.30) * err + MIDAS_CFG.get("height_kd", 0.05) * derr)
            self._mode1_last_t = now
            self._mode1_last_err = err
        desired = int(_clamp(desired, -MIDAS_CFG.get("height_max_ud", 14), MIDAS_CFG.get("height_max_ud", 14)))
        desired = int(_rate_limit(desired, self._mode1_last_ud, MIDAS_CFG.get("max_ud_delta", 4)))
        self._mode1_last_ud = desired
        return desired

    def _get_random_alt(self) -> int:
        cfg = MIDAS_CFG
        if not cfg.get("random_alt_enabled", False):
            return self._get_mode1_height_hold()
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

    def _smooth_rc_cmd(self, cmd: list) -> list:
        """限制自動模式 RC 指令每次變化量，減少高度/距離瞬間抽動。"""
        if not isinstance(cmd, (list, tuple)) or len(cmd) != 4:
            return [0, 0, 0, 0]
        out = []
        for i, v in enumerate(cmd):
            v = int(_clamp(_safe_int(v, 0), -100, 100))
            out.append(int(_rate_limit(v, self._last_auto_cmd[i], self._max_rc_delta)))
        self._last_auto_cmd = out[:]
        return out

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
                        self._last_auto_cmd = [0, 0, 0, 0]
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
                            control_cmd = self._smooth_rc_cmd(control_cmd)
                            self.tello.send_rc_control(*control_cmd)
                            last_ctrl_t = now
                    else:
                        self._last_auto_cmd = [lr, fb, ud, yv]
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
        try:
            cv2.putText(frame, f"H:{self.tello.get_height()}cm  HT:{self.inspector._height_target_cm:.0f}cm",
                        (FRAME_W - 230, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1)
        except Exception:
            pass
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