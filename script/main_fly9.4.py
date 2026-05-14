"""
main_fly9.4.py  –  Tello 雙模式任務控制系統 (修復起飛衝突版)
=============================================
功能 1（Mode 1）: 環繞巡檢  — MIDAS巡航 → FORWARD接近 → CIRCLE環繞 → QR_SCAN掃碼
功能 2（Mode 2）: 走道巡檢  — 起飛爬升 → roll左掃描QR(5個) → MiDaS判定靠近對向面板
                             → 繼續roll左掃描(5個) → 繞到第二走道 → 重複 → 回航降落
切換方式: mission_command.yaml 的 mission.mode 設 1/2/auto，
          或飛行中按鍵盤 F1(Mode1) / F2(Mode2) 手動切換。

修改紀錄:
  [9.4-A~F] 新增 Mode 2 走道巡檢、YAML 讀取、roll 平移掃描等
  [9.4-Fix2] 新增 is_flying 飛行鎖定，避免地面待機時 AisleInspector 
             狂送爬升 RC 指令導致與 takeoff 指令衝突卡死。
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
import threading

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
    """讀取 mission_command.yaml，提供各模組取用"""

    def __init__(self, path: str = YAML_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self._cfg: Dict[str, Any] = yaml.safe_load(f)
        print(f"✅ 配置已載入：{path}")

    def get(self, *keys, default=None):
        return _deep_get(self._cfg, *keys, default=default)

    # ── 各區段快速存取 ────────────────────────────────────
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

# ── 全域常數（從 YAML 讀取）──────────────────────────────
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
    # Mode 1（環繞巡檢）
    MIDAS        = "MIDAS"
    FORWARD      = "FORWARD"
    CIRCLE       = "CIRCLE"
    QR_SCAN      = "QR_SCAN"
    # Mode 2（走道巡檢）
    CLIMB        = "CLIMB"           # 起飛後爬升到巡邏高度
    AISLE_SCAN   = "AISLE_SCAN"      # roll左掃QR
    APPROACH     = "APPROACH"        # MiDaS靠近對向面板
    AISLE2_SCAN  = "AISLE2_SCAN"     # 對向面板掃QR
    AISLE_CHANGE = "AISLE_CHANGE"    # 繞行到另一條走道
    # 共用
    RETURN_HOME  = "RETURN_HOME"

class MissionMode:
    MODE1 = 1   # 環繞巡檢
    MODE2 = 2   # 走道巡檢

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
#  飛行軌跡紀錄器（航位推算）
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

    def save_path_csv(self, filename="flight_path.csv"):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_cm","y_cm","z_cm"])
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

    def is_clear(self, center_depth: float) -> bool:
        """前方無障礙物（供走道切換步驟用）"""
        return center_depth < MIDAS_CFG.get("clear_threshold", 0.25)

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
        if not (0.3 < aspect < 3.0): return False
        if area / (FRAME_W * FRAME_H) > 0.70: return False
        if area < 8000: return False
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

# ──────────────────────────────────────────────────────────
#  前進追蹤控制器（Mode 1）
# ──────────────────────────────────────────────────────────
class ForwardTracker(TargetTracker):
    def __init__(self):
        super().__init__(CFG.box_model, FORWARD_CFG)

    def process_frame(self, frame):
        det, cx, cy, area, bbox = self.detect_target(frame)
        if det:
            lr, fb, ud, yaw = self.calculate_control(cx, cy, area, self.config.get("target_area",100000))
            return lr, fb, ud, yaw, bbox, area, area >= self.config.get("target_area",100000)
        return 0, 0, 0, 0, None, 0, False

# ──────────────────────────────────────────────────────────
#  環繞掃描控制器（Mode 1）
# ──────────────────────────────────────────────────────────
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
#  QR 解碼輔助（供 Mode 1 & Mode 2 共用）
# ──────────────────────────────────────────────────────────
def decode_qr_from_frame(frame, bbox=None) -> tuple:
    """嘗試從 bbox ROI（或全幀）解碼 QR/條碼，回傳 (success, data_str)"""
    def _try_decode(img_gray):
        bc = pyzbar.decode(img_gray)
        if bc:
            return True, bc[0].data.decode("utf-8")
        return False, None

    sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

    if bbox is not None:
        x1,y1,x2,y2 = bbox
        pad  = 40
        roi  = frame[max(0,y1-pad):min(FRAME_H,y2+pad),
                     max(0,x1-pad):min(FRAME_W,x2+pad)]
        if roi.size > 0:
            rh, rw = roi.shape[:2]
            mn = min(rh, rw)
            if mn < 150:
                scale = max(2, int(300/(mn+1e-5)))
                roi   = cv2.resize(roi, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            methods = [
                g,
                cv2.GaussianBlur(g,(3,3),0),
                cv2.equalizeHist(g),
                cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2),
                cv2.filter2D(g,-1,sharpening),
                cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
                cv2.bitwise_not(g),
                cv2.bitwise_not(cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]),
            ]
            for m in methods:
                ok, data = _try_decode(m)
                if ok: return True, data

    # 全幀兜底
    gf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return _try_decode(gf)

# ──────────────────────────────────────────────────────────
#  QR 掃描靠近控制器（Mode 1 & Mode 2 共用）
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
        self._load_csv()

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
            with open(self.csv_file, "w", newline="") as f:
                csv.writer(f).writerow(["時間","資料"])

    def start(self, qr_bbox=None):
        super().start()
        self.scan_complete        = False
        self.scanned_data         = None
        self.qr_lost_time         = time.time()
        self.last_scan_time       = 0
        self.consecutive_failures = 0
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
        det, cx, cy, area, bbox = self.detect_target(frame, conf=box_conf)
        qr_decoded, decoded_data = False, None

        if det:
            self.qr_lost_time = None
            lr, fb, ud, yaw   = self.calculate_control(cx, cy, area,
                                    self.config.get("target_area", 60000))
            min_area = self.config.get("min_area_before_decode", 40000)
            if self.config.get("forward_when_no_decode", True) and not self.scan_complete:
                if area < min_area:
                    fb = self.config.get("max_speed", 15)

            reached  = area >= self.config.get("target_area", 60000)
            now      = time.time()
            if now - self.last_scan_time > self.config.get("qr_scan_interval_sec", 0.3):
                ok, data = decode_qr_from_frame(frame, bbox)
                if ok:
                    if data not in self.scanned_set:
                        self.scanned_set.add(data)
                        with open(self.csv_file, "a", newline="") as f:
                            csv.writer(f).writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data])
                        print(f"✅ 新條碼: {data}")
                    else:
                        print(f"⚠️ 重複: {data}")
                    self.scan_count   += 1
                    self.scanned_data  = data
                    self.scan_complete = True
                    qr_decoded, decoded_data = True, data
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                self.last_scan_time = now

            return lr, fb, ud, yaw, bbox, area, reached, qr_decoded, decoded_data
        else:
            if self.qr_lost_time is None:
                self.qr_lost_time = time.time()
            return 0, 0, 0, 0, None, 0, False, False, None

    def is_complete(self):
        if self.scan_complete: return True
        if self.start_time and time.time()-self.start_time > self.config.get("max_execution_time",30):
            return True
        return False

    def should_abort(self):
        if self.scan_complete: return False
        return super().should_abort()

# ──────────────────────────────────────────────────────────
#  走道巡檢狀態機（Mode 2）
# ──────────────────────────────────────────────────────────
class AisleInspector:
    """
    走道巡檢核心邏輯。
    由 TelloMissionController 每幀呼叫 process()，
    回傳 RC 指令 [lr, fb, ud, yaw] 和狀態文字。
    """

    def __init__(self, tello: "Tello", midas: MidASCruiser,
                 qr_scanner: QRScanner, tracker: FlightTracker):
        self.tello      = tello
        self.midas      = midas
        self.qr_scanner = qr_scanner
        self.tracker    = tracker

        self._state     = "CLIMB"       # 內部子狀態
        self._step_idx  = 0             # 走道切換步驟索引
        self._step_t    = 0.0           # 步驟開始時間
        self._aisle_no  = 1             # 目前走道號（1 or 2）
        self._face       = 1            # 目前掃描面（1=出發面,2=對向面）
        self._qr_count  = 0            # 本走道本側已掃數量
        self._side_done = False         # 本側掃完旗標

        # 走道切換步驟配置
        self._aisle_steps = INSP_CFG.get("aisle_change", {}).get("steps", [])
        self._target_count = INSP_CFG.get("qr_target_count", 5)
        self._roll_l  = INSP_CFG.get("roll_scan_speed",   -12)
        self._roll_r  = INSP_CFG.get("roll_rescan_speed",  12)
        self._rescan_wait = INSP_CFG.get("rescan_wait_sec", 1.5)
        self._panel_depth = INSP_CFG.get("panel_approach_depth", 0.55)
        self._panel_roll  = INSP_CFG.get("panel_roll_speed", -10)

        self._climb_target = CFG.get("takeoff", "cruise_altitude_cm", default=120)
        self._last_qr_data = None
        self._rescan_t     = 0.0

        # 已掃 QR set（本次任務）
        self._mission_scanned: set = set()

        print("🏭 AisleInspector 初始化完成")

    # ── 公開 API ──────────────────────────────────────────
    def reset(self):
        self._state   = "CLIMB"
        self._step_idx = 0
        self._aisle_no = 1
        self._face     = 1
        self._qr_count = 0
        self._side_done = False
        self._mission_scanned.clear()

    def process(self, frame, depth_norm, center_depth) -> tuple:
        """
        每幀呼叫。
        回傳: (lr, fb, ud, yaw, status_str, done)
          done=True 時表示全部走道掃完，主控器切換回航
        """
        lr = fb = ud = yaw = 0
        done = False
        status = self._state

        # ── 爬升到巡邏高度 ────────────────────────────────
        if self._state == "CLIMB":
            try:
                h = float(self.tello.get_height())
            except Exception:
                h = 0
            if h < self._climb_target - 10:
                ud = 25
                status = f"CLIMB {h:.0f}/{self._climb_target}cm"
            else:
                print(f"✅ 爬升完成 {h:.0f}cm，開始掃描走道 {self._aisle_no} 面 {self._face}")
                self._state = "ROLL_SCAN"
                self.qr_scanner.start()

        # ── roll 左掃描 QR ────────────────────────────────
        elif self._state == "ROLL_SCAN":
            lr = self._roll_l   # 向左平移（負值）
            rc, _, _, _, bbox, area, reached, decoded, data = \
                self.qr_scanner.process_frame(frame)
            ud   = rc   # 借用上下修正
            # 鎖定並靠近 QR
            if bbox is not None:
                lr = 0   # 停止平移，讓 QRScanner 控制
                lr, fb, ud, yaw = (rc, *_[1:])  # 展開時 rc 已是 lr
                # 重新從 process_frame 拿完整控制
                lr, fb, ud, yaw, bbox, area, reached, decoded, data = \
                    self.qr_scanner.process_frame(frame)

            if decoded and data and data not in self._mission_scanned:
                self._mission_scanned.add(data)
                self._qr_count += 1
                print(f"📦 走道{self._aisle_no}面{self._face} QR [{self._qr_count}/{self._target_count}]: {data}")
                self.qr_scanner.start()   # 重設，繼續掃下一個

            # 未找到新 QR → 短暫右飄回掃
            if self.qr_scanner.should_abort() or self.qr_scanner.is_timeout():
                print("↔️ 未找到新QR，右飄回掃")
                self._state  = "RESCAN_RIGHT"
                self._rescan_t = time.time()

            # 達到目標數量
            if self._qr_count >= self._target_count:
                self._side_done = True
                self._handle_side_complete()

            status = f"ROLL_SCAN A{self._aisle_no}F{self._face} [{self._qr_count}/{self._target_count}]"

        # ── 右飄回掃 ──────────────────────────────────────
        elif self._state == "RESCAN_RIGHT":
            lr = self._roll_r
            if time.time() - self._rescan_t >= self._rescan_wait:
                print("↩️ 回到左掃")
                self._state = "ROLL_SCAN"
                self.qr_scanner.start()
            status = "RESCAN_RIGHT"

        # ── 靠近對向面板（MiDaS 判定）────────────────────
        elif self._state == "APPROACH_PANEL":
            fb  = MIDAS_CFG.get("base_forward_speed", 20)
            lr  = 0
            if center_depth >= self._panel_depth:
                print(f"🏗️ 靠近對向面板 depth={center_depth:.3f}，開始掃對向")
                self._face     = 2
                self._qr_count = 0
                self._state    = "ROLL_SCAN"
                self.qr_scanner.start()
                fb = 0
            status = f"APPROACH depth={center_depth:.3f}"

        # ── 走道切換繞行 ──────────────────────────────────
        elif self._state == "AISLE_CHANGE":
            lr, fb, ud, yaw, step_done = self._run_aisle_step(center_depth)
            if step_done:
                self._step_idx += 1
                if self._step_idx >= len(self._aisle_steps):
                    # 繞行完成，進入第二走道
                    self._aisle_no = 2
                    self._face     = 1
                    self._qr_count = 0
                    self._step_idx = 0
                    print("🔄 進入走道 2")
                    self._state = "ROLL_SCAN"
                    self.qr_scanner.start()
                else:
                    self._step_t = time.time()
            status = f"AISLE_CHANGE step={self._step_idx}"

        # ── 全部完成 ──────────────────────────────────────
        elif self._state == "DONE":
            done   = True
            status = "DONE"

        return lr, fb, ud, yaw, status, done

    def _handle_side_complete(self):
        if self._face == 1:
            print(f"✅ 走道{self._aisle_no}出發面掃完，靠近對向面板")
            self._state = "APPROACH_PANEL"
        else:
            if self._aisle_no == 1:
                print("✅ 走道1兩面掃完，開始繞到走道2")
                self._state    = "AISLE_CHANGE"
                self._step_idx = 0
                self._step_t   = time.time()
            else:
                print("🎉 全部走道掃描完成！")
                self._state = "DONE"
        self._side_done = False

    def _run_aisle_step(self, center_depth) -> tuple:
        """執行當前繞行步驟，回傳 (lr,fb,ud,yaw, step_done)"""
        if self._step_idx >= len(self._aisle_steps):
            return 0, 0, 0, 0, True

        step     = self._aisle_steps[self._step_idx]
        action   = step.get("action", "")
        speed    = step.get("speed", 20)
        duration = step.get("duration_sec", 0)
        clr_th   = step.get("midas_clear_threshold",
                            MIDAS_CFG.get("clear_threshold", 0.25))

        lr = fb = ud = yaw = 0

        if action == "roll_left":
            lr = -speed
            done = (duration > 0 and time.time()-self._step_t >= duration) \
                   or (duration == 0 and center_depth < clr_th)
        elif action == "roll_right":
            lr = speed
            done = (duration > 0 and time.time()-self._step_t >= duration) \
                   or (duration == 0 and center_depth < clr_th)
        elif action == "yaw_right":
            yaw  = speed
            done = duration > 0 and time.time()-self._step_t >= duration
        elif action == "yaw_left":
            yaw  = -speed
            done = duration > 0 and time.time()-self._step_t >= duration
        elif action == "obstacle_forward":
            fb   = speed
            done = center_depth < clr_th
        else:
            done = True

        return lr, fb, ud, yaw, done

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
        self._phase = "climb"    # 先爬升到安全高度
        self._t     = time.time()
        self._yaw_int = 0

    def get_rc(self) -> list:
        if self._phase == "idle":
            return [0,0,0,0]

        # 1. 先爬升
        if self._phase == "climb":
            try:    h = float(self.tello.get_height())
            except: h = 80
            if h < self.RETURN_ALT - 10:
                return [0, 0, 20, 0]
            else:
                self._phase = "fly"
                self._t     = time.time()
                return [0,0,0,0]

        # 2. 朝起飛點飛行
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

        # 3. 懸停
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
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.tello.set_speed(50)
        print(f"🔋 電量: {self.tello.get_battery()}%")

        # 子系統
        self.midas       = MidASCruiser()
        self.forward     = ForwardTracker()
        self.circle      = CircleScanner()
        self.qr_scanner  = QRScanner()
        self.tracker     = FlightTracker()
        self.rviz_bridge = RvizBridge()
        self.return_home = ReturnHomeController(self.tello, self.tracker)
        self.inspector   = AisleInspector(
            self.tello, self.midas, self.qr_scanner, self.tracker)

        # 任務模式
        raw_mode = CFG.mission_mode
        if str(raw_mode) == "1":
            self.mission_mode = MissionMode.MODE1
        elif str(raw_mode) == "2":
            self.mission_mode = MissionMode.MODE2
        else:
            self.mission_mode = MissionMode.MODE2   # auto 預設 Mode2
        print(f"🎯 任務模式: {'環繞巡檢(1)' if self.mission_mode==MissionMode.MODE1 else '走道巡檢(2)'}")

        # 狀態
        self.current_state   = (DroneState.MIDAS if self.mission_mode == MissionMode.MODE1
                                else DroneState.CLIMB)
        self.state_start_t   = time.time()
        self.manual_mode     = False
        self.running         = True
        
        # 🟢 新增安全鎖：確保只在成功起飛後才發送自動 RC 指令
        self.is_flying       = False   
        
        self._scanned_popup  = 0.0

        # 電量監控
        self._last_bat_check       = 0.0
        self._low_battery_triggered = False

        # 高度控制（Mode 1 MIDAS）
        self._alt_next_t = float("inf")
        self._alt_ud_cmd = 0

        pygame.init()
        pygame.display.set_mode((300, 200))
        pygame.display.set_caption("Tello Mission Control v9.4")

        # 邊緣偵測：避免長按 T/L 鍵重複觸發
        self._prev_t_key = False
        self._prev_l_key = False

    # ── 鍵盤控制 ──────────────────────────────────────────
    def get_keyboard_control(self):
        lr = fb = ud = yv = 0
        manual_active = quit_flag = takeoff_cmd = land_cmd = False
        force_state   = None
        switch_mode   = None

        pygame.event.pump()
        keys = pygame.key.get_pressed()
        SPD  = 5

        if keys[pygame.K_w]:      ud = SPD;   manual_active = True
        if keys[pygame.K_s]:      ud = -SPD;  manual_active = True
        if keys[pygame.K_a]:      yv = -SPD;  manual_active = True
        if keys[pygame.K_d]:      yv = SPD;   manual_active = True
        if keys[pygame.K_UP]:     fb = SPD;   manual_active = True
        if keys[pygame.K_DOWN]:   fb = -SPD;  manual_active = True
        if keys[pygame.K_LEFT]:   lr = -SPD;  manual_active = True
        if keys[pygame.K_RIGHT]:  lr = SPD;   manual_active = True
        if keys[pygame.K_SPACE]:  lr=fb=ud=yv=0; manual_active=True
        if keys[pygame.K_t]:      takeoff_cmd = True
        if keys[pygame.K_l]:      land_cmd    = True
        if keys[pygame.K_ESCAPE]: quit_flag   = True
        # F1/F2 切換模式
        if keys[pygame.K_F1]:     switch_mode = MissionMode.MODE1
        if keys[pygame.K_F2]:     switch_mode = MissionMode.MODE2
        # 數字鍵強制狀態（Mode 1）
        if keys[pygame.K_1]:  force_state = DroneState.MIDAS
        if keys[pygame.K_2]:  force_state = DroneState.FORWARD
        if keys[pygame.K_3]:  force_state = DroneState.CIRCLE
        if keys[pygame.K_4]:  force_state = DroneState.QR_SCAN

        return (manual_active, lr, fb, ud, yv,
                quit_flag, force_state,
                bool(keys[pygame.K_t]), bool(keys[pygame.K_l]),
                switch_mode)

    def change_state(self, new_state, qr_bbox=None):
        old = self.current_state
        self.current_state = new_state
        self.state_start_t = time.time()
        if new_state == DroneState.RETURN_HOME: self.return_home.start()
        if new_state == DroneState.FORWARD:     self.forward.start()
        elif new_state == DroneState.CIRCLE:    self.circle.start()
        elif new_state == DroneState.QR_SCAN:   self.qr_scanner.start(qr_bbox)
        print(f"\n🔄 狀態切換: {old} → {new_state}")

    def _get_random_alt(self) -> int:
        cfg = MIDAS_CFG
        now = time.time()
        if now >= self._alt_next_t:
            self._alt_next_t = now + cfg.get("alt_change_interval_sec", 3.0)
            try:    h = float(self.tello.get_height())
            except: h = 100.0
            if h <= cfg.get("alt_min_cm", 50):
                self._alt_ud_cmd = cfg.get("alt_speed", 12)
            elif h >= cfg.get("alt_max_cm", 180):
                self._alt_ud_cmd = -cfg.get("alt_speed", 12)
            else:
                c = random.randint(0, 2)
                alt_sp = cfg.get("alt_speed", 12)
                self._alt_ud_cmd = alt_sp if c==0 else (-alt_sp if c==1 else 0)
        return self._alt_ud_cmd

    def _check_battery(self):
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

    # ── 主迴圈 ────────────────────────────────────────────
    def run(self):
        print("\n" + "="*55)
        print("Tello 任務控制器 v9.4")
        print("F1=環繞模式  F2=走道巡檢模式")
        print("T=起飛  L=降落  方向鍵=移動  WSAD=升降轉  ESC=停止")
        print("="*55)

        frame_reader     = self.tello.get_frame_read()
        last_ctrl_t      = time.time()

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

                # 邊緣偵測：只在「剛按下」那一幀觸發，避免長按重複起飛/降落
                takeoff_cmd = t_held and not self._prev_t_key
                land_cmd    = l_held and not self._prev_l_key
                self._prev_t_key = t_held
                self._prev_l_key = l_held

                self.tracker.update(self.tello, is_manual=manual_active)
                self.rviz_bridge.send(self.tracker)

                if quit_flag:
                    print("使用者中斷")
                    break

                # ── 起飛指令 (保護機制：地面狀態才能起飛) ──
                if takeoff_cmd and not self.is_flying:
                    print("🛸 手動起飛...")
                    try:
                        self.tello.takeoff()
                        self.is_flying = True  # 起飛成功才解除鎖定
                    except Exception as e:
                        print(f"❌ 起飛失敗: {e}")
                        continue
                    
                    stab = CFG.get("takeoff", "stabilize_wait_sec", default=2.5)
                    print(f"⏳ 等待起飛穩定 {stab} 秒...")
                    time.sleep(stab)
                    
                    self.tracker.reset_pose()
                    self._alt_next_t = time.time() + 5.0
                    self._alt_ud_cmd = 0
                    if self.mission_mode == MissionMode.MODE2:
                        self.inspector.reset()
                        self.current_state = DroneState.CLIMB
                    else:
                        self.current_state = DroneState.MIDAS
                    print("✅ 起飛完成，正式進入自動巡邏")

                # ── 降落指令 (保護機制：飛行狀態才能手動降落) ──
                if land_cmd and self.is_flying:
                    self._alt_ud_cmd = 0
                    self._alt_next_t = float("inf")
                    self.is_flying = False     # 將狀態鎖死，停止 RC 發送
                    self.tello.send_rc_control(0,0,0,0)
                    time.sleep(0.3)
                    self.tello.land()

                if switch_mode is not None and switch_mode != self.mission_mode:
                    self.mission_mode = switch_mode
                    label = "環繞巡檢(1)" if switch_mode==MissionMode.MODE1 else "走道巡檢(2)"
                    print(f"🔀 切換模式 → {label}")
                    if switch_mode == MissionMode.MODE1:
                        self.change_state(DroneState.MIDAS)
                    else:
                        self.inspector.reset()
                        self.current_state = DroneState.CLIMB

                if force_state:
                    self.change_state(force_state)

                # ────────────────────────────────────────
                #  自動/手動控制 (安全鎖：僅在 is_flying=True 時送出 RC 指令)
                # ────────────────────────────────────────
                if self.is_flying:
                    if not manual_active:
                        control_cmd = [0, 0, 0, 0]

                        # ── Mode 1：環繞巡檢 ────────────────
                        if self.mission_mode == MissionMode.MODE1:
                            control_cmd = self._run_mode1(frame)

                        # ── Mode 2：走道巡檢 ────────────────
                        elif self.mission_mode == MissionMode.MODE2:
                            control_cmd, frame = self._run_mode2(frame)

                        # ── 低電量/完成任務 回航 ────────────
                        if self.current_state == DroneState.RETURN_HOME:
                            control_cmd = self.return_home.get_rc()
                            self.rviz_bridge.set_returning(True)
                            # 如果自動降落程序已啟動，將飛行狀態解除
                            if self.return_home.is_landing():
                                self.is_flying = False

                        now = time.time()
                        if now - last_ctrl_t >= CONTROL_INTERVAL:
                            self.tello.send_rc_control(*control_cmd)
                            last_ctrl_t = now

                    else:
                        self.tello.send_rc_control(lr, fb, ud, yv)
                        cv2.putText(frame, "MANUAL MODE", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    # 地面待機狀態 (不送任何 RC 指令)
                    cv2.putText(frame, "STANDBY (Press T to Takeoff)", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                # ── HUD ────────────────────────────────
                cv2.putText(frame,
                    f"State:{self.current_state}  Mode:{'1-Circle' if self.mission_mode==1 else '2-Aisle'}",
                    (10, FRAME_H-60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                cv2.putText(frame, f"Bat:{self.tello.get_battery()}%",
                    (FRAME_W-130, FRAME_H-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, "T:Takeoff  L:Land  F1/F2:Mode",
                    (10, FRAME_H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

                if time.time() < self._scanned_popup:
                    h_, w_ = frame.shape[:2]
                    ov = frame.copy()
                    cv2.rectangle(ov,(w_//2-180,h_//2-60),(w_//2+180,h_//2+60),(0,200,0),-1)
                    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
                    cv2.putText(frame,"SCANNED !",(w_//2-130,h_//2+20),
                                cv2.FONT_HERSHEY_DUPLEX,1.8,(255,255,255),3)

                cv2.imshow("Tello Mission Control v9.4", frame)
                if cv2.waitKey(1) == 27:
                    break

        except Exception as e:
            print(f"❌ 錯誤: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.cleanup()

    # ── Mode 1 邏輯（環繞巡檢）───────────────────────────
    def _run_mode1(self, frame) -> list:
        ctrl = [0,0,0,0]

        if self.current_state == DroneState.MIDAS:
            depth_norm, center_depth = self.midas.process_frame(frame)
            fbv, yv = self.midas.get_control(center_depth, time.time())
            ud_m    = self._get_random_alt()
            ctrl    = [0, fbv, ud_m, yv]

            depth_disp = cv2.applyColorMap(
                (depth_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("Depth Map", depth_disp)

            results = self.forward.model(frame, conf=box_conf, verbose=False)
            if results[0].boxes:
                valid = [b for b in results[0].boxes
                         if self.forward._is_valid_box(*map(int,b.xyxy[0]))]
                if valid:
                    best = max(valid, key=lambda b:(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
                    area = (best.xyxy[0][2]-best.xyxy[0][0])*(best.xyxy[0][3]-best.xyxy[0][1])
                    if area > MIDAS_CFG.get("target_found_area", 10000):
                        self.change_state(DroneState.FORWARD)

            frame[:] = self.midas.draw_overlay(frame, center_depth, fbv, yv)

        elif self.current_state == DroneState.FORWARD:
            lr, fb, ud, yaw, bbox, area, reached = self.forward.process_frame(frame)
            ctrl = [lr, fb, ud, yaw]
            if bbox:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,"MODE: FORWARD",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            if reached:
                self.tello.send_rc_control(0,0,0,0); time.sleep(1)
                self.change_state(DroneState.CIRCLE)
            elif self.forward.should_abort() or self.forward.is_timeout():
                self.change_state(DroneState.MIDAS)

        elif self.current_state == DroneState.CIRCLE:
            lr, fb, ud, yaw, bbox, qr_det, qr_bbox = self.circle.process_frame(frame)
            ctrl = [lr, fb, ud, yaw]
            if bbox:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            if qr_det and qr_bbox:
                qx1,qy1,qx2,qy2 = qr_bbox
                cv2.rectangle(frame,(qx1,qy1),(qx2,qy2),(255,255,0),3)
            cv2.putText(frame,"MODE: CIRCLE SCAN",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
            if qr_det and qr_bbox and self.circle.is_complete():
                self.tello.send_rc_control(0,0,0,0); time.sleep(1)
                self.change_state(DroneState.QR_SCAN, qr_bbox)
            elif self.circle.should_abort() or self.circle.is_timeout():
                self.change_state(DroneState.MIDAS)

        elif self.current_state == DroneState.QR_SCAN:
            lr, fb, ud, yaw, bbox, area, reached, decoded, data = \
                self.qr_scanner.process_frame(frame)
            ctrl = [lr, fb, ud, yaw]
            if bbox:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),3)
            if decoded and data:
                cv2.putText(frame,f"SCANNED:{data}",(10,120),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                self._scanned_popup = time.time()+3.0
            cv2.putText(frame,"MODE: QR SCAN",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
            if self.qr_scanner.is_complete() or self.qr_scanner.should_abort():
                self.tello.send_rc_control(0,0,50,0); time.sleep(1)
                self.change_state(DroneState.MIDAS)

        return ctrl

    # ── Mode 2 邏輯（走道巡檢）───────────────────────────
    def _run_mode2(self, frame) -> tuple:
        if self.current_state == DroneState.RETURN_HOME:
            return [0,0,0,0], frame

        # MiDaS 每幀都跑（供 inspector 用）
        depth_norm, center_depth = self.midas.process_frame(frame)
        depth_disp = cv2.applyColorMap(
            (depth_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("Depth Map", depth_disp)

        lr, fb, ud, yaw, status, done = \
            self.inspector.process(frame, depth_norm, center_depth)

        if done:
            print("🏁 全部掃完，自動返航")
            self.change_state(DroneState.RETURN_HOME)
            return [0,0,0,0], frame

        cv2.putText(frame, f"AISLE: {status}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 2)
        cv2.putText(frame, f"Depth:{center_depth:.3f}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)

        return [lr, fb, ud, yaw], frame

    # ── 清理 ──────────────────────────────────────────────
    def cleanup(self):
        print("\n🧹 清理中...")
        self.rviz_bridge.close()
        self.tello.send_rc_control(0,0,0,0)
        time.sleep(0.5)
        self.tracker.save_path_csv()
        self.tello.streamoff()
        pygame.quit()
        cv2.destroyAllWindows()
        print("✅ 程式結束")

# ──────────────────────────────────────────────────────────
#  程式入口
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    controller = TelloMissionController()
    controller.run()