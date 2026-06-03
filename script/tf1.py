"""
main_fly9.5_fixed.py  –  Tello 雙模式任務控制系統 (終極防干擾定距定高版)
=============================================
功能 1（Mode 1）: 環繞巡檢  — MIDAS巡航 → FORWARD接近 → CIRCLE環繞 → QR_SCAN掃碼
功能 2（Mode 2）: 走道巡檢  — 起飛爬升 → roll左掃描QR(5個) → MiDaS判定靠近對向面板
                             → 繼續roll左掃描(5個) → 繞到第二走道 → 重複 → 回航降落

修改紀錄:
  [9.5-A~D] 保留 su3高容錯QR解碼 / 定速計時起飛 / 畫面追蹤框 / 語法修正。
  [9.5-E] 導入 MiDaS 定距平移邏輯：解決靠近面板時前後不一的抖動。
  [9.5-F] 導入 高度計定高邏輯 + 物理突變過濾器：解決平移掉高度問題，並防止飛越桌子時暴衝上升。
  [9.5-FIX1] 統一 RC 發送節流：手動 / 自動模式都在主迴圈底部統一發送，避免指令衝突。
  [9.5-FIX2] MiDaS 深度值加入一階低通濾波 (α=0.3)，減少 fb 前後抖動。
  [9.5-FIX3] _get_stable_height 改為指數平滑，防止把真實掉高也過濾掉。
  [9.5-FIX4] 狀態切換時先發 [0,0,0,0] 煞停，避免帶速衝進下一狀態。
  [9.5-FIX5] ROLL_SCAN 發現 bbox 時 fb 固定由 MiDaS base_fb 控制，QR 只管 lr/ud/yaw。
  [9.5-FIX6] 主迴圈加入低電量強制返航保護。
"""

import csv
import json
import math
import os
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
YAML_PATH = os.path.join(os.path.dirname(__file__), "mission_command.yaml")

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
    def box_model(self): return self.get("models", "box_model_path")
    @property
    def qr_model(self):  return self.get("models", "barcode_model_path")
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

class DroneState:
    MIDAS        = "MIDAS"
    FORWARD      = "FORWARD"
    CIRCLE       = "CIRCLE"
    QR_SCAN      = "QR_SCAN"
    CLIMB        = "CLIMB"
    AISLE_SCAN   = "AISLE_SCAN"
    APPROACH     = "APPROACH"
    AISLE2_SCAN  = "AISLE2_SCAN"
    AISLE_CHANGE = "AISLE_CHANGE"
    RETURN_HOME  = "RETURN_HOME"

class MissionMode:
    MODE1 = 1
    MODE2 = 2

# ──────────────────────────────────────────────────────────
#  RViz UDP 橋接 & 飛行軌跡紀錄器 (精簡保留)
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
#  [FIX2] 加入一階低通濾波，平滑 center_depth，減少 fb 抖動
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
        # [FIX2] 低通濾波狀態
        self._smooth_depth = 0.5

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
        # [FIX2] 一階低通濾波 (α=0.3)，對新值保守，抑制快速抖動
        raw = float(np.mean(self.center_q))
        self._smooth_depth = 0.7 * self._smooth_depth + 0.3 * raw
        return depth_norm, self._smooth_depth

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
            if abs(ea) > CIRCLE_CFG.get("area_tolerance", 5000): fb = max(-20, min(20, int(CIRCLE_CFG.get("kp_forward",0.0006)*ea)))

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
#  QR 解碼輔助（完整保留 su3 版高容錯多重預處理）
# ──────────────────────────────────────────────────────────
def decode_qr_from_frame(frame, bbox=None) -> tuple:
    """
    嘗試解碼 QR/條碼，回傳 (success, data_str)。
    完整還原 su3 版：
      1. 不只依賴 YOLO bbox；bbox=None 時也會全畫面嘗試解碼。
      2. 同時支援 pyzbar 與 OpenCV QRCodeDetector。
      3. 針對模糊、太小、反光、黑白反相，嘗試多種前處理版本。
    """
    def _try_decode(img):
        try:
            barcodes = pyzbar.decode(img)
            if barcodes:
                return True, barcodes[0].data.decode("utf-8", errors="ignore")
        except Exception:
            pass

        try:
            if len(img.shape) == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img
            detector = cv2.QRCodeDetector()
            data, points, _ = detector.detectAndDecode(bgr)
            if data:
                return True, data
        except Exception:
            pass

        return False, None

    def _build_methods(img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        h, w = gray.shape[:2]
        methods = []

        methods.append(gray)
        if min(h, w) < 220:
            scale = max(2, int(360 / (min(h, w) + 1e-5)))
            methods.append(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        methods.append(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

        sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        base_list = list(methods)
        for g in base_list:
            try:
                methods.append(cv2.GaussianBlur(g, (3, 3), 0))
                methods.append(cv2.equalizeHist(g))
                methods.append(cv2.filter2D(g, -1, sharpening))
                methods.append(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
                methods.append(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 11, 2))
                methods.append(cv2.bitwise_not(g))
            except Exception:
                pass
        return methods

    rois = []
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        pad = 60
        roi = frame[max(0, y1 - pad):min(FRAME_H, y2 + pad),
                    max(0, x1 - pad):min(FRAME_W, x2 + pad)]
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
        self.scanned_set          = set()
        self.scan_count           = 0
        self.last_scan_time       = 0
        self.scan_complete        = False
        self.qr_lost_time         = None
        self.scanned_data         = None
        self.consecutive_failures = 0
        self.csv_file             = QR_CFG.get("csv_file", "scanned_codes.csv")
        self.event_csv_file       = QR_CFG.get("event_csv_file", "scan_events.csv")
        self.context_provider     = None

        self.direct_decode_enabled = QR_CFG.get("direct_decode_enabled", True)
        self.direct_decode_interval_sec = QR_CFG.get("direct_decode_interval_sec", 0.25)
        self._last_direct_decode_t = 0.0
        self.debug_enabled = QR_CFG.get("debug", True)
        self.last_debug_status = "QR:INIT"
        self.last_detect_bbox = None
        self.last_detect_area = 0

        self._load_csv()

    def _is_valid_box(self, x1, y1, x2, y2):
        w, h   = x2-x1, y2-y1
        area   = w * h
        aspect = w / (h + 1e-5)

        min_aspect = self.config.get("min_aspect", 0.15)
        max_aspect = self.config.get("max_aspect", 8.0)
        min_area   = self.config.get("min_box_area", 200)
        max_ratio  = self.config.get("max_area_ratio", 0.90)

        if not (min_aspect < aspect < max_aspect):
            return False
        if area / (FRAME_W * FRAME_H) > max_ratio:
            return False
        if area < min_area:
            return False
        return True

    def _load_csv(self):
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, "r", newline="", encoding="utf-8") as f:
                    for row in list(csv.reader(f))[1:]:
                        if len(row) >= 2:
                            self.scanned_set.add(row[1])
                print(f"📚 已載入 {len(self.scanned_set)} 筆歷史掃描資料")
            except Exception as e:
                print(f"⚠️ 載入歷史資料出錯: {e}")
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
            try:
                ctx = self.context_provider() or {}
            except Exception:
                ctx = {}
        with open(self.event_csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data,
                int(bool(duplicate)),
                ctx.get("mission_mode", ""),
                ctx.get("drone_state", ""),
                ctx.get("aisle_no", ""),
                ctx.get("face_no", ""),
                ctx.get("qr_count", ""),
                ctx.get("target_count", ""),
                ctx.get("battery_pct", ""),
                ctx.get("x_cm", ""),
                ctx.get("y_cm", ""),
                ctx.get("z_cm", ""),
                str(bbox) if bbox is not None else "",
                int(area) if area is not None else 0,
            ])

    def _record_decode(self, data, bbox=None, area=0):
        is_duplicate = data in self.scanned_set
        if not is_duplicate:
            self.scanned_set.add(data)
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data])
            print(f"✅ 新條碼: {data}")
        else:
            print(f"⚠️ 重複: {data}")

        self._log_event(data, duplicate=is_duplicate, bbox=bbox, area=area)
        self.scan_count   += 1
        self.scanned_data  = data
        self.scan_complete = True
        self.consecutive_failures = 0
        self.last_debug_status = f"QR:DECODED {data[:18]}"
        return True, data

    def start(self, qr_bbox=None):
        super().start()
        self.scan_complete        = False
        self.scanned_data         = None
        self.qr_lost_time         = time.time()
        self.last_scan_time       = 0
        self.consecutive_failures = 0
        self._last_direct_decode_t = 0.0
        self.last_debug_status = "QR:SCANNING"
        if qr_bbox:
            cx   = (qr_bbox[0]+qr_bbox[2])//2
            cy   = (qr_bbox[1]+qr_bbox[3])//2
            area = (qr_bbox[2]-qr_bbox[0])*(qr_bbox[3]-qr_bbox[1])
            self.target_center_history.append((cx,cy))
            self.last_bbox      = qr_bbox
            self.last_bbox_area = area
            self.has_target     = True
        print("📸 QR Code 掃描模式啟動")

    def process_frame(self, frame):
        qr_decoded, decoded_data = False, None
        lr = fb = ud = yaw = 0
        bbox = None
        area = 0
        reached = False
        now = time.time()

        det, cx, cy, area, bbox = self.detect_target(frame, conf=qr_conf)
        self.last_detect_bbox = bbox
        self.last_detect_area = area

        if det:
            self.qr_lost_time = None
            self.last_debug_status = f"QR:YOLO area={int(area)} fail={self.consecutive_failures}"
            lr, fb, ud, yaw = self.calculate_control(cx, cy, area,
                                    self.config.get("target_area", 60000))
            min_area = self.config.get("min_area_before_decode", 40000)
            if self.config.get("forward_when_no_decode", True) and not self.scan_complete:
                if area < min_area:
                    fb = self.config.get("max_speed", 15)
            reached = area >= self.config.get("target_area", 60000)

            if now - self.last_scan_time > self.config.get("qr_scan_interval_sec", 0.3):
                ok, data = decode_qr_from_frame(frame, bbox)
                if ok and data:
                    qr_decoded, decoded_data = self._record_decode(data, bbox=bbox, area=area)
                else:
                    self.consecutive_failures += 1
                    self.last_debug_status = f"QR:YOLO no-decode fail={self.consecutive_failures}"
                self.last_scan_time = now

            return lr, fb, ud, yaw, bbox, area, reached, qr_decoded, decoded_data

        if self.qr_lost_time is None:
            self.qr_lost_time = time.time()
        self.has_target = False

        if self.direct_decode_enabled and (now - self._last_direct_decode_t > self.direct_decode_interval_sec):
            ok, data = decode_qr_from_frame(frame, bbox=None)
            self._last_direct_decode_t = now
            if ok and data:
                print(f"✅ 全畫面直接解碼成功: {data}")
                qr_decoded, decoded_data = self._record_decode(data, bbox=None, area=0)
                return 0, 0, 0, 0, None, 0, False, qr_decoded, decoded_data

        self.last_debug_status = "QR:NO YOLO / DIRECT SCAN..."
        return 0, 0, 0, 0, None, 0, False, False, None

    def is_complete(self):
        if self.scan_complete:
            return True
        if self.start_time and time.time()-self.start_time > self.config.get("max_execution_time",30):
            return True
        return False

    def should_abort(self):
        if self.scan_complete:
            return False
        return super().should_abort()

# ──────────────────────────────────────────────────────────
#  走道巡檢狀態機（Mode 2）
#  [FIX3] _get_stable_height 改為指數平滑
#  [FIX4] 狀態切換時先發 RC(0,0,0,0) 煞停
#  [FIX5] ROLL_SCAN 發現 bbox 時 fb 強制由 MiDaS base_fb 控制
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
        self._roll_r       = INSP_CFG.get("roll_rescan_speed", 12)
        self._rescan_wait  = INSP_CFG.get("rescan_wait_sec", 1.5)

        self._climb_target   = CFG.get("takeoff", "cruise_altitude_cm", default=135)
        self._climb_speed    = INSP_CFG.get("climb_speed", 30)
        self._climb_duration = INSP_CFG.get("climb_duration_sec", 2.0)

        self._panel_depth  = INSP_CFG.get("panel_approach_depth", 0.45)
        self._turn_speed   = INSP_CFG.get("turn_180_speed", 35)
        self._turn_tol     = INSP_CFG.get("turn_tolerance_deg", 8)

        self._maintain_dist = INSP_CFG.get("maintain_distance_enabled", True)
        self._target_depth  = INSP_CFG.get("target_depth", 0.45)
        self._depth_tol     = INSP_CFG.get("depth_tolerance", 0.03)
        self._depth_kp      = INSP_CFG.get("depth_kp", 100)
        self._max_fb        = INSP_CFG.get("max_fb_speed", 15)

        self._maintain_height = INSP_CFG.get("maintain_height_enabled", True)
        self._height_kp       = INSP_CFG.get("height_kp", 0.8)
        self._max_ud          = INSP_CFG.get("max_ud_speed", 20)

        # [FIX3] 指數平滑高度濾波狀態
        self._last_valid_h    = 0.0

        self.reset()

    def reset(self):
        self._state    = "CLIMB"
        self._step_idx = 0
        self._step_t   = 0.0
        self._aisle_no = 1
        self._face     = 1
        self._qr_count = 0
        self._climb_start_t = 0.0
        self._turn_target_yaw = None
        self._mission_scanned = set()
        self._last_valid_h    = 0.0

    def _get_yaw(self):
        try: return float(self.tello.get_yaw())
        except Exception: return 0.0

    def _get_stable_height(self) -> float:
        """
        [FIX3] 改為指數平滑，避免把真實掉高也過濾掉。
        - 突變超過 25cm：alpha=0.05（保守，幾乎不信新值，認定是桌面/障礙物）
        - 正常緩慢變化：alpha=0.3（適度跟隨真實高度）
        """
        try:
            raw_h = float(self.tello.get_height())
            if self._last_valid_h == 0.0:
                self._last_valid_h = raw_h
                return raw_h
            alpha = 0.05 if abs(raw_h - self._last_valid_h) > 25 else 0.3
            self._last_valid_h = alpha * raw_h + (1 - alpha) * self._last_valid_h
            return self._last_valid_h
        except Exception:
            return float(self._climb_target)

    def _norm_ang(self, a):
        while a > 180: a -= 360
        while a < -180: a += 360
        return a

    def _run_turn_to_target(self, speed):
        if self._turn_target_yaw is None: return 0, True
        err = self._norm_ang(self._turn_target_yaw - self._get_yaw())
        if abs(err) <= self._turn_tol: return 0, True
        return (speed if err > 0 else -speed), False

    def _brake(self):
        """[FIX4] 立即煞停輔助：發送零指令並稍作等待"""
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        time.sleep(0.15)

    def process(self, frame, depth_norm, center_depth) -> tuple:
        lr = fb = ud = yaw = 0
        done = False
        status = self._state

        if self._state == "CLIMB":
            if self._climb_start_t == 0.0:
                self._climb_start_t = time.time()
            elapsed = time.time() - self._climb_start_t

            if elapsed < self._climb_duration:
                ud = self._climb_speed
                status = f"CLIMB {elapsed:.1f}/{self._climb_duration}s"
            else:
                print(f"✅ 爬升完成，開始掃描走道 {self._aisle_no} 面 {self._face}")
                self._brake()  # [FIX4]
                self._state = "ROLL_SCAN"
                self.qr_scanner.start()

        elif self._state == "ROLL_SCAN":
            qr_lr, qr_fb, qr_ud, qr_yaw, bbox, area, reached, decoded, data = self.qr_scanner.process_frame(frame)

            # 1. MiDaS 距離維持 PID 控制 (fb)
            base_fb = 0
            if self._maintain_dist:
                err_depth = self._target_depth - center_depth
                if abs(err_depth) > self._depth_tol:
                    base_fb = int(err_depth * self._depth_kp)
                    base_fb = max(-self._max_fb, min(self._max_fb, base_fb))

            # 2. 高度維持 PID + 指數平滑高度 (ud)
            base_ud = 0
            if self._maintain_height:
                err_height = self._climb_target - self._get_stable_height()
                base_ud = int(err_height * self._height_kp)
                base_ud = max(-self._max_ud, min(self._max_ud, base_ud))

            # 預設：向左飄 + MiDaS 維持距離 + 高度維持
            lr, fb, ud, yaw = self._roll_l, base_fb, base_ud, 0

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, f"Locked", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # [FIX5] 發現 bbox 時：lr/ud/yaw 交給 QRScanner，
                #        fb 強制保留 MiDaS base_fb，不被 qr_fb 覆蓋
                lr  = qr_lr
                ud  = qr_ud
                yaw = qr_yaw
                fb  = base_fb  # ← FIX5：前後距離由 MiDaS 控制

            if decoded and data and data not in self._mission_scanned:
                self._mission_scanned.add(data)
                self._qr_count += 1
                print(f"📦 走道{self._aisle_no}面{self._face} [{self._qr_count}/{self._target_count}]: {data}")
                self.qr_scanner.start()
                lr, fb, ud, yaw = self._roll_l, base_fb, base_ud, 0

            if self.qr_scanner.should_abort() or self.qr_scanner.is_timeout():
                self._brake()  # [FIX4]
                self._state = "RESCAN_RIGHT"
                self._rescan_t = time.time()
                lr, fb, ud, yaw = self._roll_r, base_fb, base_ud, 0

            if self._qr_count >= self._target_count:
                if self._face == 1:
                    print("✅ 第一面掃完，準備轉向 180 度")
                    self._brake()  # [FIX4]
                    self._state = "TURN_OPPOSITE"
                    self._turn_target_yaw = self._norm_ang(self._get_yaw() - 180)
                else:
                    if self._aisle_no == 1:
                        self._brake()  # [FIX4]
                        self._state, self._step_idx, self._step_t = "AISLE_CHANGE", 0, time.time()
                    else:
                        self._brake()  # [FIX4]
                        self._state = "DONE"
                lr, fb, ud, yaw = 0, 0, 0, 0

            status = f"SCAN A{self._aisle_no}F{self._face} [{self._qr_count}/{self._target_count}]"

        elif self._state == "RESCAN_RIGHT":
            base_fb = 0
            if self._maintain_dist:
                err_depth = self._target_depth - center_depth
                if abs(err_depth) > self._depth_tol:
                    base_fb = int(err_depth * self._depth_kp)
                    base_fb = max(-self._max_fb, min(self._max_fb, base_fb))

            base_ud = 0
            if self._maintain_height:
                err_height = self._climb_target - self._get_stable_height()
                base_ud = int(err_height * self._height_kp)
                base_ud = max(-self._max_ud, min(self._max_ud, base_ud))

            lr = self._roll_r
            fb = base_fb
            ud = base_ud
            if time.time() - self._rescan_t >= self._rescan_wait:
                self._brake()  # [FIX4]
                self._state = "ROLL_SCAN"
                self.qr_scanner.start()
            status = "RESCAN_RIGHT"

        elif self._state == "TURN_OPPOSITE":
            yaw, finished = self._run_turn_to_target(self._turn_speed)
            if finished:
                self._brake()  # [FIX4]
                self._state = "APPROACH_PANEL"
            status = "TURN_180"

        elif self._state == "APPROACH_PANEL":
            fb = MIDAS_CFG.get("base_forward_speed", 20)
            if center_depth >= self._panel_depth:
                self._brake()  # [FIX4]
                self._face, self._qr_count, self._state = 2, 0, "ROLL_SCAN"
                self.qr_scanner.start()
                fb = 0
            status = f"APPROACH D:{center_depth:.2f}"

        elif self._state == "AISLE_CHANGE":
            if self._step_idx >= len(self._aisle_steps):
                self._brake()  # [FIX4]
                self._aisle_no, self._face, self._qr_count, self._state = 2, 1, 0, "ROLL_SCAN"
                self.qr_scanner.start()
            else:
                step = self._aisle_steps[self._step_idx]
                act, spd, dur = step.get("action",""), step.get("speed",20), step.get("duration_sec",0)
                cth = step.get("midas_clear_threshold", 0.20)

                if act == "roll_left": lr = -spd
                if act == "yaw_right":
                    if self._turn_target_yaw is None: self._turn_target_yaw = self._norm_ang(self._get_yaw() + step.get("target_deg",90))
                    yaw, step_done = self._run_turn_to_target(spd)

                step_done = (dur > 0 and time.time()-self._step_t >= dur) or (dur == 0 and center_depth < cth)
                if act == "yaw_right": step_done = self._run_turn_to_target(spd)[1]

                if step_done:
                    self._brake()  # [FIX4]
                    self._step_idx += 1
                    self._step_t = time.time()
                    self._turn_target_yaw = None
            status = f"CHANGE step {self._step_idx}"

        elif self._state == "DONE": done = True

        return lr, fb, ud, yaw, status, done

# ──────────────────────────────────────────────────────────
#  自動返航控制器
# ──────────────────────────────────────────────────────────
class ReturnHomeController:
    def __init__(self, tello, tracker):
        self.tello, self.tracker = tello, tracker
        self._phase = "idle"
        self._t = self._last_err = self._int = 0

    def start(self):
        self._phase, self._t, self._int = "climb", time.time(), 0

    def get_rc(self):
        if self._phase == "idle": return [0,0,0,0]
        if self._phase == "climb":
            try: h = self.tello.get_height()
            except: h = 80
            if h < RETURN_CFG.get("return_altitude_cm", 150) - 10: return [0,0,20,0]
            self._phase, self._t = "fly", time.time()
            return [0,0,0,0]
        if self._phase == "fly":
            dx, dz = self.tracker.home[0]-self.tracker.x, self.tracker.home[2]-self.tracker.z
            d = math.sqrt(dx**2+dz**2)*2
            if d <= RETURN_CFG.get("arrive_radius_cm", 80):
                self._phase, self._t = "hover", time.time()
                return [0,0,0,0]
            tyaw = math.degrees(math.atan2(dx, dz)) if (abs(dx)>0.1 or abs(dz)>0.1) else 0
            err = tyaw - self.tracker.yaw
            while err > 180: err-=360
            while err < -180: err+=360
            self._int = max(-100, min(100, self._int + err*0.01))
            yaw = int(max(-40, min(40, 0.8*err + 0.05*self._int + 0.1*(err-self._last_err))))
            self._last_err = err
            fb = min(50, max(10, int(d/8))) if abs(err) < 30 else 0
            try: cur_h = self.tello.get_height()
            except: cur_h = 80
            ud = -10 if cur_h > 50 else 0
            return [0, fb, ud, yaw]
        if self._phase == "hover" and time.time()-self._t >= 2.0:
            self._phase = "land"
            self.tello.land()
        return [0,0,0,0]

    def is_landing(self): return self._phase == "land"

# ──────────────────────────────────────────────────────────
#  主控制器
#  [FIX1] 統一 RC 發送節流：所有模式都在主迴圈底部統一發送
#  [FIX6] 低電量強制返航保護
# ──────────────────────────────────────────────────────────
class TelloMissionController:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.tello.set_speed(50)
        self.midas, self.forward, self.circle, self.qr_scanner = MidASCruiser(), ForwardTracker(), CircleScanner(), QRScanner()
        self.tracker, self.rviz_bridge = FlightTracker(), RvizBridge()
        self.return_home = ReturnHomeController(self.tello, self.tracker)
        self.inspector = AisleInspector(self.tello, self.midas, self.qr_scanner, self.tracker)

        self.mission_mode = MissionMode.MODE1 if str(CFG.mission_mode) == "1" else MissionMode.MODE2
        self.current_state = DroneState.MIDAS if self.mission_mode == MissionMode.MODE1 else DroneState.CLIMB
        self.running, self.is_flying = True, False

        pygame.init()
        pygame.display.set_mode((300, 200))

    def run(self):
        fr = self.tello.get_frame_read()
        last_t = time.time()
        prev_t = prev_l = False
        try:
            while self.running:
                frame = fr.frame
                if frame is None: time.sleep(0.05); continue
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                pygame.event.pump()
                k = pygame.key.get_pressed()

                # ── 搖桿輸入 ──
                lr = fb = ud = yv = 0
                man = False
                if k[pygame.K_w]: ud, man = 50, True
                if k[pygame.K_s]: ud, man = -50, True
                if k[pygame.K_a]: yv, man = -50, True
                if k[pygame.K_d]: yv, man = 50, True
                if k[pygame.K_UP]: fb, man = 50, True
                if k[pygame.K_DOWN]: fb, man = -50, True
                if k[pygame.K_LEFT]: lr, man = -50, True
                if k[pygame.K_RIGHT]: lr, man = 50, True
                if k[pygame.K_ESCAPE]: break

                t_cmd = k[pygame.K_t] and not prev_t
                l_cmd = k[pygame.K_l] and not prev_l
                prev_t, prev_l = k[pygame.K_t], k[pygame.K_l]

                self.tracker.update(self.tello, man)
                self.rviz_bridge.send(self.tracker)

                # ── 起飛 / 降落 ──
                if t_cmd and not self.is_flying:
                    self.tello.takeoff()
                    self.is_flying = True
                    time.sleep(2.5)
                    self.tracker.reset_pose()
                    if self.mission_mode == MissionMode.MODE2:
                        self.inspector.reset()
                        self.current_state = DroneState.CLIMB

                if l_cmd and self.is_flying:
                    self.is_flying = False
                    self.tello.send_rc_control(0, 0, 0, 0)
                    self.tello.land()

                # ── [FIX6] 低電量強制返航 ──
                if self.is_flying and self.current_state != DroneState.RETURN_HOME:
                    try:
                        bat = self.tello.get_battery()
                    except Exception:
                        bat = 100
                    if bat < LOWBAT_CFG.get("critical_pct", 15):
                        print(f"🔋 低電量 {bat}%，強制返航！")
                        self.current_state = DroneState.RETURN_HOME
                        self.return_home.start()

                # ── 自動任務邏輯（計算 cmd） ──
                cmd = [0, 0, 0, 0]
                if self.is_flying and not man:
                    if self.mission_mode == MissionMode.MODE1:
                        pass  # Mode1 邏輯保留擴充位置
                    elif self.mission_mode == MissionMode.MODE2:
                        if self.current_state != DroneState.RETURN_HOME:
                            dn, cd = self.midas.process_frame(frame)
                            lr, fb, ud, yaw, st, done = self.inspector.process(frame, dn, cd)
                            cmd = [lr, fb, ud, yaw]
                            cv2.putText(frame, f"AISLE: {st}  D:{cd:.2f}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            if done:
                                self.current_state = DroneState.RETURN_HOME
                                self.return_home.start()

                    if self.current_state == DroneState.RETURN_HOME:
                        cmd = self.return_home.get_rc()
                        if self.return_home.is_landing():
                            self.is_flying = False

                # ── [FIX1] 統一節流發送：手動 & 自動都在這裡送出，避免重複/衝突 ──
                now = time.time()
                if now - last_t >= CONTROL_INTERVAL:
                    if self.is_flying:
                        if man:
                            self.tello.send_rc_control(lr, fb, ud, yv)
                        else:
                            self.tello.send_rc_control(*cmd)
                    last_t = now

                # ── HUD ──
                self.tracker.draw_minimap(frame)
                try:
                    bat_disp = self.tello.get_battery()
                except Exception:
                    bat_disp = "?"
                cv2.putText(frame, f"Bat: {bat_disp}%", (10, FRAME_H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Tello Mission Control v9.5-fixed", frame)
                if cv2.waitKey(1) == 27:
                    break

        finally:
            self.tello.send_rc_control(0, 0, 0, 0)
            self.tracker.save_path_csv()
            self.tello.streamoff()
            pygame.quit()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    TelloMissionController().run()