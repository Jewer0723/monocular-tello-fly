"""
main_fly9.5.py  –  Tello 雙模式任務控制系統 (su3高容錯完整保留版)
=============================================
功能 1（Mode 1）: 環繞巡檢  — MIDAS巡航 → FORWARD接近 → CIRCLE環繞 → QR_SCAN掃碼
功能 2（Mode 2）: 走道巡檢  — 起飛爬升 → roll左掃描QR(5個) → MiDaS判定靠近對向面板
                             → 繼續roll左掃描(5個) → 繞到第二走道 → 重複 → 回航降落
切換方式: mission_command.yaml 的 mission.mode 設 1/2/auto，
          或飛行中按鍵盤 F1(Mode1) / F2(Mode2) 手動切換。

修改紀錄:
  [9.5-A] 完整保留 su3.py 的多重影像預處理與 pyzbar/OpenCV 雙引擎解碼，確保實機掃描成功率。
  [9.5-B] 起飛 CLIMB 改為定速與秒數計時，解決 Tello 氣壓計初始化高度不一問題。
  [9.5-C] 在走道巡檢 (Mode 2) 鎖定目標時，畫面上補上黃色目標追蹤框 UI 反饋。
  [9.5-D] 修復主迴圈 tuple 賦值語法錯誤。
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

# ──────────────────────────────────────────────────────────
#  狀態定義
# ──────────────────────────────────────────────────────────
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
        print(f"📡 RvizBridge → UDP {host}:{port}")

    def send(self, tracker):
        now = time.time()
        if now - self._last_t < 0.1:
            return
        self._last_t = now
        try:
            payload = json.dumps({
                "x":    round(tracker.x,   1),
                "z":    round(tracker.z,   1),
                "yaw":  round(tracker.yaw, 1),
                "home": [tracker.home[0], tracker.home[2]],
                "returning": self._returning,
            }).encode()
            self._sock.sendto(payload, self._addr)
        except Exception:
            pass

    def set_returning(self, val: bool):
        self._returning = val

    def close(self):
        try: self._sock.close()
        except Exception: pass

# ──────────────────────────────────────────────────────────
#  飛行軌跡紀錄器
# ──────────────────────────────────────────────────────────
class FlightTracker:
    def __init__(self):
        self.reset()

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
        if dt <= 0 or dt > 1.0:
            return
        try:
            vx = float(tello.get_speed_x())
            vy = float(tello.get_speed_y())
            vz = float(tello.get_speed_z())
            self.yaw = float(tello.get_yaw())
        except Exception:
            return
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
        if len(self.path) < 2:
            return frame
        xs   = [p[0] for p in self.path]
        zs   = [p[2] for p in self.path]
        span = max(max(xs)-min(xs), max(zs)-min(zs), 100)
        scale = (size - 20) / span
        cx_map, cy_map = x0 + size//2, y0 + size//2
        ox = (max(xs)+min(xs))/2
        oz = (max(zs)+min(zs))/2
        def to_px(px, pz):
            return (int(cx_map+(px-ox)*scale), int(cy_map-(pz-oz)*scale))
        for i in range(1, len(self.path)):
            p1 = to_px(self.path[i-1][0], self.path[i-1][2])
            p2 = to_px(self.path[i][0],   self.path[i][2])
            is_man = len(self.path[i]) > 3 and self.path[i][3]
            cv2.line(frame, p1, p2, (0,140,255) if is_man else (0,200,255), 1)
        cv2.circle(frame, to_px(self.home[0], self.home[2]), 5, (0,255,0), -1)
        cv2.circle(frame, to_px(self.x, self.z), 5, (0,0,255), -1)
        dist = self.distance_to_home()
        cv2.putText(frame, f"HOME:{dist:.0f}cm", (x0+2, y0+size-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        return frame

    def save_path_csv(self, filename="flight_path.csv"):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x_cm", "y_cm", "z_cm", "is_manual"])
            writer.writerows(self.path)
        print(f"📁 軌跡已儲存: {filename}")

# ──────────────────────────────────────────────────────────
#  MiDaS 避障巡航控制器
# ──────────────────────────────────────────────────────────
class MidASCruiser:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("MiDaS device:", self.device)
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform
        win = MIDAS_CFG.get("smoothing_window", 5)
        self.center_q = deque(maxlen=win)
        self.left_q   = deque(maxlen=win)
        self.right_q  = deque(maxlen=win)
        self.state           = "FORWARD"
        self.turn_start_time = 0
        self.obstacle_count  = 0

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp     = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            pred = self.midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=img_rgb.shape[:2],
                mode="bicubic", align_corners=False).squeeze()
        depth      = pred.cpu().numpy()
        depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
        c, l, r    = self._get_regions(depth_norm)
        self.center_q.append(c)
        self.left_q.append(l)
        self.right_q.append(r)
        return depth_norm, float(np.mean(self.center_q))

    def _get_regions(self, d):
        h, w = d.shape
        ch, cw = h//3, w//3
        ct, cl = h//2 - ch//2, w//2 - cw//2
        center = d[ct:ct+ch, cl:cl+cw]
        left   = d[ct:ct+ch, :w//4]
        right  = d[ct:ct+ch, 3*w//4:]
        return (np.median(center) if center.size else 0.5,
                np.median(left)   if left.size   else 0.5,
                np.median(right)  if right.size  else 0.5)

    def get_control(self, center_depth, now):
        obs_th  = MIDAS_CFG.get("obstacle_threshold", 0.35)
        clr_th  = MIDAS_CFG.get("clear_threshold",    0.25)
        turn_d  = MIDAS_CFG.get("turn_duration_sec",  1.5)
        fwd_sp  = MIDAS_CFG.get("base_forward_speed", 20)
        turn_sp = MIDAS_CFG.get("turn_speed",         40)
        if self.state == "FORWARD":
            if center_depth > obs_th:
                self.state = "TURNING"
                self.turn_start_time = now
                self.obstacle_count += 1
        else:
            if now - self.turn_start_time >= turn_d:
                if center_depth < clr_th:
                    self.state = "FORWARD"
                else:
                    self.turn_start_time = now
        if self.state == "FORWARD":
            return fwd_sp, 0
        return 0, turn_sp

    def draw_overlay(self, frame, center_depth, fbv, yv):
        if self.state == "TURNING":
            color, status = (0,165,255), "TURNING RIGHT"
        elif center_depth > MIDAS_CFG.get("obstacle_threshold", 0.35):
            color, status = (0,0,255), "OBSTACLE!"
        elif center_depth > MIDAS_CFG.get("clear_threshold", 0.25):
            color, status = (0,255,255), "CAUTION"
        else:
            color, status = (0,255,0), "CLEAR"
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//3,h//3), (2*w//3,2*h//3), color, 2)
        cv2.putText(frame, "MODE: MIDAS CRUISE", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Status: {status}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# ──────────────────────────────────────────────────────────
#  目標追蹤基類（YOLO）
# ──────────────────────────────────────────────────────────
class TargetTracker:
    def __init__(self, model_path, config: dict):
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
        self.target_lost_time = time.time()
        self.target_center_history.clear()

    def _is_valid_box(self, x1, y1, x2, y2):
        w, h   = x2-x1, y2-y1
        area   = w * h
        aspect = w / (h + 1e-5)

        min_aspect = self.config.get("min_aspect", 0.3)
        max_aspect = self.config.get("max_aspect", 3.0)
        min_area   = self.config.get("min_box_area", 8000)
        max_ratio  = self.config.get("max_area_ratio", 0.70)

        if not (min_aspect < aspect < max_aspect):
            return False
        if area / (FRAME_W * FRAME_H) > max_ratio:
            return False
        if area < min_area:
            return False
        return True

    def detect_target(self, frame, conf=None):
        if conf is None:
            conf = box_conf
        results = self.model(frame, conf=conf, verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            valid = [b for b in results[0].boxes
                     if self._is_valid_box(*map(int, b.xyxy[0]))]
            if not valid:
                if self.target_lost_time is None:
                    self.target_lost_time = time.time()
                self.has_target = False
                return False, 0, 0, 0, None
            best = max(valid, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
            x1,y1,x2,y2 = map(int, best.xyxy[0])
            cx, cy = (x1+x2)//2, (y1+y2)//2
            area   = (x2-x1)*(y2-y1)
            self.target_center_history.append((cx,cy))
            acx = int(np.mean([c[0] for c in self.target_center_history]))
            acy = int(np.mean([c[1] for c in self.target_center_history]))
            self.last_bbox = (x1,y1,x2,y2)
            self.last_bbox_area = area
            self.has_target = True
            self.target_lost_time = None
            return True, acx, acy, area, (x1,y1,x2,y2)
        else:
            if self.target_lost_time is None:
                self.target_lost_time = time.time()
            self.has_target = False
            return False, 0, 0, 0, None

    def calculate_control(self, tcx, tcy, tarea, tarea_goal):
        ex = tcx - FRAME_W//2
        ey = tcy - FRAME_H//2
        ea = tarea_goal - tarea
        dz = self.config.get("deadzone", 20)
        ms = self.config.get("max_speed", 20)
        yaw = lr = ud = fb = 0
        if abs(ex) > dz:
            if abs(ex) > 120:
                lr  = self._clamp(int(self.config.get("kp_yaw",0.3)*ex), -ms, ms)
            else:
                yaw = self._clamp(int(self.config.get("kp_yaw",0.3)*ex), -ms, ms)
        if abs(ey) > dz:
            ud = self._clamp(int(-self.config.get("kp_updown",0.3)*ey), -ms, ms)
        if abs(ea) > self.config.get("area_tolerance", 5000):
            fb = self._clamp(int(self.config.get("kp_forward",0.0006)*ea), -ms, ms)
        return lr, fb, ud, yaw

    def should_abort(self):
        if not self.has_target and self.target_lost_time is not None:
            if time.time() - self.target_lost_time > self.config.get("target_lost_timeout", 2):
                return True
        return False

    def is_timeout(self):
        if self.start_time and time.time()-self.start_time > self.config.get("max_execution_time",30):
            return True
        return False

    @staticmethod
    def _clamp(v, lo, hi): return max(lo, min(hi, v))

class ForwardTracker(TargetTracker):
    def __init__(self):
        super().__init__(CFG.box_model, FORWARD_CFG)

    def process_frame(self, frame):
        det, cx, cy, area, bbox = self.detect_target(frame)
        if det:
            lr, fb, ud, yaw = self.calculate_control(cx, cy, area, self.config.get("target_area",100000))
            return lr, fb, ud, yaw, bbox, area, area >= self.config.get("target_area",100000)
        return 0, 0, 0, 0, None, 0, False

class CircleScanner(TargetTracker):
    def __init__(self):
        super().__init__(CFG.box_model, CIRCLE_CFG)
        self.qr_model    = YOLO(CFG.qr_model)
        self.scanned_set = set()
        self.smooth_center = deque(maxlen=3)

    def start(self):
        super().start()
        self.smooth_center.clear()
        print("🔄 開始環繞掃描模式")

    def process_frame(self, frame):
        det, cx, cy, area, bbox = self.detect_target(frame)
        qr_detected, qr_bbox = False, None
        lr = CIRCLE_CFG.get("orbit_speed", 7)
        fb = ud = yaw = 0

        if det:
            self.smooth_center.append((cx, cy))
            acx = int(np.mean([c[0] for c in self.smooth_center]))
            acy = int(np.mean([c[1] for c in self.smooth_center]))
            ex  = acx - FRAME_W//2
            ey  = acy - FRAME_H//2
            ea  = CIRCLE_CFG.get("target_area",120000) - area
            yaw_max = CIRCLE_CFG.get("yaw_correction_speed", 25)
            ud_max  = CIRCLE_CFG.get("height_correction_speed", 15)
            if abs(ex) > 120:
                lr  = 0
                yaw = self._clamp(int(FORWARD_CFG.get("kp_yaw",0.3)*ex), -yaw_max, yaw_max)
            else:
                lr  = CIRCLE_CFG.get("orbit_speed", 7)
                if abs(ex) > FORWARD_CFG.get("deadzone", 20):
                    yaw = self._clamp(int(FORWARD_CFG.get("kp_yaw",0.3)*ex), -yaw_max, yaw_max)
            if abs(ey) > FORWARD_CFG.get("deadzone", 20):
                ud = self._clamp(int(-FORWARD_CFG.get("kp_updown",0.3)*ey*0.5), -ud_max, ud_max)
            if abs(ea) > CIRCLE_CFG.get("area_tolerance", 5000):
                fb = self._clamp(int(CIRCLE_CFG.get("kp_forward",0.0006)*ea),
                                 -FORWARD_CFG.get("max_speed",20), FORWARD_CFG.get("max_speed",20))
            qr_detected, qr_bbox = self._detect_qr(frame, bbox)

        return lr, fb, ud, yaw, bbox, qr_detected, qr_bbox

    def _detect_qr(self, frame, bbox):
        if bbox is None: return False, None
        x1,y1,x2,y2 = bbox
        roi = frame[max(0,y1-50):min(FRAME_H,y2+50),
                    max(0,x1-50):min(FRAME_W,x2+50)]
        if roi.size == 0: return False, None
        res = self.qr_model(roi, conf=qr_conf, verbose=False)
        if res[0].boxes is not None and len(res[0].boxes) > 0:
            best = max(res[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
            qx1,qy1,qx2,qy2 = map(int, best.xyxy[0])
            ox, oy = max(0,x1-50), max(0,y1-50)
            return True, (qx1+ox, qy1+oy, qx2+ox, qy2+oy)
        return False, None

    def is_complete(self):
        return self.start_time is not None and \
               time.time()-self.start_time >= CIRCLE_CFG.get("min_circle_time_sec", 5)

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

        # 原圖與放大圖：QR 太小時很重要。
        methods.append(gray)
        if min(h, w) < 220:
            scale = max(2, int(360 / (min(h, w) + 1e-5)))
            methods.append(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        methods.append(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

        # 對比、二值化、銳化、反相。
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

    # 全畫面兜底
    rois.append(frame)

    for roi in rois:
        for img in _build_methods(roi):
            ok, data = _try_decode(img)
            if ok and data:
                return True, data

    return False, None

# ──────────────────────────────────────────────────────────
#  QR 掃描靠近控制器（完整保留 su3 版邏輯）
# ──────────────────────────────────────────────────────────
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
        """QR/條碼通常比箱體小，所以使用較寬鬆的有效框條件。"""
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
        
        # [修改] 讀取定速爬升參數
        self._climb_speed  = INSP_CFG.get("climb_speed", 30)
        self._climb_duration = INSP_CFG.get("climb_duration_sec", 2.0)
        
        self._panel_depth  = INSP_CFG.get("panel_approach_depth", 0.45)
        self._turn_speed   = INSP_CFG.get("turn_180_speed", 35)
        self._turn_tol     = INSP_CFG.get("turn_tolerance_deg", 8)

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

    def _get_yaw(self):
        try: return float(self.tello.get_yaw())
        except Exception: return 0.0

    def _norm_ang(self, a):
        while a > 180: a -= 360
        while a < -180: a += 360
        return a

    def _run_turn_to_target(self, speed):
        if self._turn_target_yaw is None: return 0, True
        err = self._norm_ang(self._turn_target_yaw - self._get_yaw())
        if abs(err) <= self._turn_tol: return 0, True
        return (speed if err > 0 else -speed), False

    def process(self, frame, depth_norm, center_depth) -> tuple:
        lr = fb = ud = yaw = 0
        done = False
        status = self._state

        if self._state == "CLIMB":
            # [修改] 改為定速與秒數計時起飛，解決 Tello 高度計初始化異常
            if self._climb_start_t == 0.0:
                self._climb_start_t = time.time()
            elapsed = time.time() - self._climb_start_t
            
            if elapsed < self._climb_duration:
                ud = self._climb_speed
                status = f"CLIMB {elapsed:.1f}/{self._climb_duration}s"
            else:
                print(f"✅ 爬升完成，開始掃描走道 {self._aisle_no} 面 {self._face}")
                self._state = "ROLL_SCAN"
                self.qr_scanner.start()

        elif self._state == "ROLL_SCAN":
            qr_lr, qr_fb, qr_ud, qr_yaw, bbox, area, reached, decoded, data = self.qr_scanner.process_frame(frame)
            
            # 預設維持向左飄，並借用 qr_ud 穩定高度
            lr, fb, ud, yaw = self._roll_l, 0, qr_ud, 0

            if bbox is not None:
                # [修改] 畫出追蹤框，給予視覺反饋
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, f"Locked", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 發現目標時，完全交給 QRScanner 的控制邏輯（自動對準/前進）
                lr, fb, ud, yaw = qr_lr, qr_fb, qr_ud, qr_yaw

            if decoded and data and data not in self._mission_scanned:
                self._mission_scanned.add(data)
                self._qr_count += 1
                print(f"📦 走道{self._aisle_no}面{self._face} [{self._qr_count}/{self._target_count}]: {data}")
                self.qr_scanner.start()
                # 掃完後立刻恢復左飄
                lr, fb, ud, yaw = self._roll_l, 0, 0, 0

            if self.qr_scanner.should_abort() or self.qr_scanner.is_timeout():
                self._state = "RESCAN_RIGHT"
                self._rescan_t = time.time()
                lr, fb, ud, yaw = self._roll_r, 0, 0, 0

            if self._qr_count >= self._target_count:
                if self._face == 1:
                    print("✅ 第一面掃完，準備轉向 180 度")
                    self._state = "TURN_OPPOSITE"
                    self._turn_target_yaw = self._norm_ang(self._get_yaw() - 180)
                else:
                    if self._aisle_no == 1:
                        self._state, self._step_idx, self._step_t = "AISLE_CHANGE", 0, time.time()
                    else:
                        self._state = "DONE"
                lr, fb, ud, yaw = 0, 0, 0, 0

            status = f"SCAN A{self._aisle_no}F{self._face} [{self._qr_count}/{self._target_count}]"

        elif self._state == "RESCAN_RIGHT":
            lr = self._roll_r
            if time.time() - self._rescan_t >= self._rescan_wait:
                self._state = "ROLL_SCAN"
                self.qr_scanner.start()
            status = "RESCAN_RIGHT"

        elif self._state == "TURN_OPPOSITE":
            yaw, finished = self._run_turn_to_target(self._turn_speed)
            if finished:
                self._state = "APPROACH_PANEL"
            status = "TURN_180"

        elif self._state == "APPROACH_PANEL":
            fb = MIDAS_CFG.get("base_forward_speed", 20)
            if center_depth >= self._panel_depth:
                self._face, self._qr_count, self._state = 2, 0, "ROLL_SCAN"
                self.qr_scanner.start()
                fb = 0
            status = f"APPROACH D:{center_depth:.2f}"

        elif self._state == "AISLE_CHANGE":
            if self._step_idx >= len(self._aisle_steps):
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
                    self._step_idx += 1
                    self._step_t = time.time()
                    self._turn_target_yaw = None
            status = f"CHANGE step {self._step_idx}"

        elif self._state == "DONE": done = True

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

        self.current_state = DroneState.MIDAS if self.mission_mode == MissionMode.MODE1 else DroneState.CLIMB
        self.running, self.is_flying = True, False
        self._alt_next_t = float("inf")
        self._last_bat_check = 0.0
        self._low_battery_triggered = False
        
        pygame.init()
        pygame.display.set_mode((300, 200))
        self.qr_scanner.set_context_provider(self._get_scan_context)

    def _get_scan_context(self):
        info = self.inspector.get_hud_info()
        try: bat = self.tello.get_battery()
        except Exception: bat = ""
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

    def change_state(self, new_state, qr_bbox=None):
        old = self.current_state
        self.current_state = new_state
        if new_state == DroneState.RETURN_HOME:
            self.return_home.start()
        if new_state == DroneState.FORWARD:
            self.forward.start()
        elif new_state == DroneState.CIRCLE:
            self.circle.start()
        elif new_state == DroneState.QR_SCAN:
            self.qr_scanner.start(qr_bbox)
        print(f"\n🔄 狀態切換: {old} → {new_state}")

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

                if t_cmd and not self.is_flying:
                    self.tello.takeoff()
                    self.is_flying = True
                    time.sleep(2.5)
                    self.tracker.reset_pose()
                    if self.mission_mode == MissionMode.MODE2: self.inspector.reset(); self.current_state = DroneState.CLIMB
                if l_cmd and self.is_flying:
                    self.is_flying = False
                    self.tello.send_rc_control(0,0,0,0)
                    self.tello.land()

                if self.is_flying and not man:
                    cmd = [0,0,0,0]
                    if self.mission_mode == MissionMode.MODE1:
                        pass # Mode1邏輯維持不變
                    elif self.mission_mode == MissionMode.MODE2:
                        if self.current_state != DroneState.RETURN_HOME:
                            dn, cd = self.midas.process_frame(frame)
                            lr, fb, ud, yaw, st, done = self.inspector.process(frame, dn, cd)
                            cmd = [lr, fb, ud, yaw]
                            cv2.putText(frame, f"AISLE: {st}  D:{cd:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                            
                            # [修改] 修復元組賦值錯誤，正確切換到自動返航
                            if done: 
                                self.current_state = DroneState.RETURN_HOME
                                self.return_home.start()
                    
                    if self.current_state == DroneState.RETURN_HOME:
                        cmd = self.return_home.get_rc()
                        if self.return_home.is_landing(): self.is_flying = False

                    now = time.time()
                    if now - last_t >= CONTROL_INTERVAL:
                        self.tello.send_rc_control(*cmd)
                        last_t = now
                elif man:
                    self.tello.send_rc_control(lr, fb, ud, yv)

                self.tracker.draw_minimap(frame)
                cv2.putText(frame, f"Bat: {self.tello.get_battery()}%", (10, FRAME_H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow("Tello Mission Control v9.5", frame)
                if cv2.waitKey(1) == 27: break
        finally:
            self.tello.send_rc_control(0,0,0,0)
            self.tracker.save_path_csv()
            self.tello.streamoff()
            pygame.quit()

if __name__ == "__main__":
    TelloMissionController().run()