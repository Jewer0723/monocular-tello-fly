"""
su4.7_dashboard.py  –  Tello 雙模式任務控制系統（三區塊單一操作視窗版）
=============================================
功能 1（Mode 1）: 環繞巡檢  — MIDAS巡航 → FORWARD接近 → CIRCLE環繞 → QR_SCAN掃碼
功能 2（Mode 2）: 走道巡檢  — 起飛後直接 roll左掃描QR(5個) → MiDaS判定靠近對向面板
                             → 左轉180度 → 繼續roll左掃描(5個) → 繞到第二走道 → 重複 → 回航降落

修改紀錄:
  [su4.1] 移除起飛爬升 (CLIMB) 邏輯，起飛後直接進入 ROLL_SCAN。
  [su4.1] 移除右偏回朔 (RESCAN_RIGHT) 邏輯，若掃描丟失則直接繼續向左掃描。
  [su4.1] 導入 MiDaS 定距平移邏輯：SCAN 階段不再依賴 YOLO 面積決定前後距離。
  [su4.1-yaml] 將飛行速度、控制參數、模型路徑、任務參數集中到 mission_command.yaml。
  [su4.3-no-height] 保留原 su4.3 掃描流程，只移除高度計定高控制：
          - ROLL_SCAN 階段 ud 固定為 0，不再讀取/修正高度。
          - 發現 QR 目標時仍使用 QRScanner 的 lr/yaw，前後距離仍由 MiDaS base_fb 維持。
  [su4.7-dashboard] 將即時影像、MiDaS 深度圖、任務／鍵盤控制資訊整合為單一 Pygame 視窗。
          - 移除 OpenCV 的 Tello Mission Control 與 Depth Map 分離視窗。
          - 支援調整視窗大小，版面比例可由 mission_command.yaml 的 dashboard 區塊設定。
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
    def box_conf(self):  return self.get("models", "box_conf", default=0.50)
    @property
    def qr_conf(self):   return self.get("models", "qr_conf",  default=0.65)

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
DASHBOARD_CFG = CFG.section("dashboard")

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
            payload = json.dumps({"x": round(tracker.x, 1), "y": round(tracker.y, 1),"z": round(tracker.z, 1), "yaw": round(tracker.yaw, 1), "home": [tracker.home[0], tracker.home[2]], "returning": self._returning}).encode()
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
#  單一視窗三區塊操作儀表板
#  左側：即時影像；右上：MiDaS 深度圖；右下：任務／控制資訊
# ──────────────────────────────────────────────────────────
class MissionDashboard:
    BG = (15, 19, 26)
    PANEL_BG = (24, 30, 40)
    PANEL_BORDER = (58, 70, 88)
    TEXT = (230, 235, 242)
    MUTED = (155, 166, 182)
    ACCENT = (53, 190, 255)
    GREEN = (72, 210, 134)
    ORANGE = (255, 184, 77)

    def __init__(self):
        pygame.init()
        self.min_width = int(DASHBOARD_CFG.get("min_width", 960))
        self.min_height = int(DASHBOARD_CFG.get("min_height", 600))
        width = max(self.min_width, int(DASHBOARD_CFG.get("width", 1280)))
        height = max(self.min_height, int(DASHBOARD_CFG.get("height", 720)))
        self.left_ratio = float(DASHBOARD_CFG.get("left_ratio", 0.68))
        self.depth_ratio = float(DASHBOARD_CFG.get("depth_ratio", 0.47))
        self.max_fps = max(10, int(DASHBOARD_CFG.get("max_fps", 30)))

        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Tello Inspection Dashboard")
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont("consolas", 24, bold=True)
        self.panel_font = pygame.font.SysFont("consolas", 18, bold=True)
        self.text_font = pygame.font.SysFont("consolas", 15)
        self.small_font = pygame.font.SysFont("consolas", 13)

    def handle_events(self) -> bool:
        """處理單一 Pygame 視窗事件；回傳是否要求結束。"""
        quit_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.VIDEORESIZE:
                width = max(self.min_width, int(event.w))
                height = max(self.min_height, int(event.h))
                self.screen = pygame.display.set_mode(
                    (width, height), pygame.RESIZABLE
                )
        return quit_requested

    @staticmethod
    def _content_rect(rect: pygame.Rect, title_height: int = 34) -> pygame.Rect:
        return pygame.Rect(
            rect.x + 8,
            rect.y + title_height,
            max(1, rect.w - 16),
            max(1, rect.h - title_height - 8),
        )

    def _draw_panel(self, rect: pygame.Rect, title: str):
        pygame.draw.rect(self.screen, self.PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(
            self.screen, self.PANEL_BORDER, rect, width=1, border_radius=8
        )
        label = self.panel_font.render(title, True, self.ACCENT)
        self.screen.blit(label, (rect.x + 12, rect.y + 8))

    def _blit_bgr(self, image_bgr, target: pygame.Rect):
        if image_bgr is None or image_bgr.size == 0:
            placeholder = self.text_font.render("Waiting for image...", True, self.MUTED)
            self.screen.blit(placeholder, placeholder.get_rect(center=target.center))
            return

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        src_h, src_w = rgb.shape[:2]
        scale = min(target.w / float(src_w), target.h / float(src_h))
        dst_w = max(1, int(src_w * scale))
        dst_h = max(1, int(src_h * scale))
        surface = pygame.image.frombuffer(rgb.tobytes(), (src_w, src_h), "RGB")
        surface = pygame.transform.smoothscale(surface, (dst_w, dst_h))
        x = target.x + (target.w - dst_w) // 2
        y = target.y + (target.h - dst_h) // 2
        self.screen.blit(surface, (x, y))

    def _fit_text(self, text: str, font, max_width: int) -> str:
        text = str(text)
        if font.size(text)[0] <= max_width:
            return text
        suffix = "..."
        while text and font.size(text + suffix)[0] > max_width:
            text = text[:-1]
        return text + suffix

    def _draw_status(self, rect: pygame.Rect, info: dict):
        content = self._content_rect(rect)
        x, y = content.x + 4, content.y + 3
        line_h = 18
        max_w = max(1, content.w - 8)

        flight_color = self.GREEN if info.get("is_flying") else self.ORANGE
        lines = [
            (f"FLIGHT  : {info.get('flight_label', '--')}", flight_color),
            (f"MODE    : {info.get('mode_label', '--')}", self.TEXT),
            (f"STATE   : {info.get('state', '--')}", self.TEXT),
            (f"AISLE   : {info.get('aisle', '--')}   FACE: {info.get('face', '--')}   QR: {info.get('qr', '--')}", self.TEXT),
            (f"BATTERY : {info.get('battery', '--')}%   HEIGHT: {info.get('height', '--')} cm", self.TEXT),
            (f"DEPTH   : {info.get('depth', '--')}   TARGET: {info.get('target_depth', '--')}", self.TEXT),
            (f"POSE    : X {info.get('x', '--')}   Y {info.get('y', '--')}   Z {info.get('z', '--')} cm", self.TEXT),
            (f"YAW     : {info.get('yaw', '--')} deg", self.TEXT),
            (f"RC CMD  : LR {info.get('lr', 0):+d}  FB {info.get('fb', 0):+d}  UD {info.get('ud', 0):+d}  YAW {info.get('rc_yaw', 0):+d}", self.TEXT),
            (f"HYBRID  : {info.get('hybrid', '--')}   AUTO COUNT: {info.get('auto_count', 0)}", self.TEXT),
        ]

        for text_value, color in lines:
            if y + line_h > content.bottom:
                break
            text_value = self._fit_text(text_value, self.text_font, max_w)
            self.screen.blit(self.text_font.render(text_value, True, color), (x, y))
            y += line_h

        help_lines = [
            "T Takeoff | L Land | SPACE Hover/Stop",
            "ARROWS Move | W/S Up/Down | A/D Yaw",
            "F1 Circle | F2 Aisle | 1..5 State | ESC Quit",
        ]
        required_h = len(help_lines) * 16 + 8
        help_y = max(y + 4, content.bottom - required_h)
        if help_y < content.bottom:
            pygame.draw.line(
                self.screen,
                self.PANEL_BORDER,
                (content.x, help_y - 5),
                (content.right, help_y - 5),
                1,
            )
        for help_text in help_lines:
            if help_y + 15 > content.bottom:
                break
            help_text = self._fit_text(help_text, self.small_font, max_w)
            self.screen.blit(self.small_font.render(help_text, True, self.MUTED), (x, help_y))
            help_y += 16

    def render(self, camera_bgr, depth_bgr, info: dict):
        self.screen.fill(self.BG)
        width, height = self.screen.get_size()
        pad = 12
        header_h = 48
        content_h = max(1, height - header_h - pad)
        available_w = max(1, width - pad * 3)
        left_w = int(available_w * self.left_ratio)
        left_w = max(1, min(left_w, available_w - 1))
        right_w = max(1, available_w - left_w)

        # 🔴 修改配置：將左側切割為上方攝影機 (約 80%) 與下方 CSV 區塊 (約 20%)
        csv_h = int((content_h - pad) * 0.20)
        camera_h = content_h - pad - csv_h

        camera_rect = pygame.Rect(pad, header_h, left_w, camera_h)
        csv_rect = pygame.Rect(pad, camera_rect.bottom + pad, left_w, csv_h)
        right_x = camera_rect.right + pad
        depth_h = int((content_h - pad) * self.depth_ratio)
        depth_h = max(1, min(depth_h, content_h - pad - 1))
        depth_rect = pygame.Rect(right_x, header_h, right_w, depth_h)
        status_rect = pygame.Rect(
            right_x,
            depth_rect.bottom + pad,
            right_w,
            max(1, content_h - depth_h - pad),
        )

        title = self.title_font.render("TELLO INSPECTION DASHBOARD", True, self.TEXT)
        self.screen.blit(title, (pad, 11))
        status_text = "FLYING" if info.get("is_flying") else "STANDBY"
        status_color = self.GREEN if info.get("is_flying") else self.ORANGE
        status_surface = self.panel_font.render(status_text, True, status_color)
        self.screen.blit(status_surface, (width - pad - status_surface.get_width(), 15))

        self._draw_panel(camera_rect, "1  LIVE CAMERA / DETECTION")
        self._draw_panel(depth_rect, "2  MIDAS DEPTH MAP")
        self._draw_panel(status_rect, "3  MISSION / MANUAL CONTROL")
        self._draw_panel(csv_rect, "4  SCANNED QR CODES")  # 🔴 新增
        self._blit_bgr(camera_bgr, self._content_rect(camera_rect))
        self._blit_bgr(depth_bgr, self._content_rect(depth_rect))
        self._draw_status(status_rect, info)

        # 🔴 新增：繪製 CSV 掃描紀錄
        csv_content = self._content_rect(csv_rect)
        cx, cy = csv_content.x + 4, csv_content.y + 3
        line_h = 18
        max_cw = max(1, csv_content.w - 8)

        scanned_list = info.get("scanned_codes", [])
        # 只取面板容納得下的筆數，並反轉順序讓「最新掃描」顯示在最上方
        max_items = max(1, int(csv_content.h // line_h))
        display_items = scanned_list[-max_items:]
        display_items.reverse()

        for row in display_items:
            if cy + line_h > csv_content.bottom:
                break

            # 🔴 修改格式：直接讀取原始儲存的時間字串 (包含年月日與時間)
            time_str = row[0] if len(row) > 0 else ""
            data_str = row[1] if len(row) > 1 else ""

            # 組合字串格式：年月日 時間 資料名稱
            text_str = self._fit_text(f"{time_str} {data_str}", self.text_font, max_cw)
            self.screen.blit(self.text_font.render(text_str, True, self.GREEN), (cx, cy))
            cy += line_h

        pygame.display.flip()
        self.clock.tick(self.max_fps)

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
        self.scanned_list = []  # 🔴 新增：用來依序儲存 [Time, Data] 的列表
        self.scan_count = self.last_scan_time = self.consecutive_failures = 0
        self.scan_complete = False
        self.csv_file = QR_CFG.get("csv_file", "scanned_codes.csv")
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
                        if len(row) >= 2:
                            self.scanned_set.add(row[1])
                            self.scanned_list.append(row)  # 🔴 新增：載入歷史資料
            except Exception: pass
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Time", "Data"])

    def set_context_provider(self, fn):
        self.context_provider = fn

    def _record_decode(self, data, bbox=None, area=0):
        is_dup = data in self.scanned_set

        # 1. 取得當下時間字串
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 2. 🔴 修改點：無論是否為重複條碼，都加入 scanned_list，讓儀表板「立刻」跳出最新資料
        self.scanned_list.append([time_str, data])

        # 3. CSV 存檔邏輯照舊：只有「不重複」的條碼才會寫入檔案與掃描集合
        if not is_dup:
            self.scanned_set.add(data)
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([time_str, data])
            print(f"✅ 新條碼已記錄至 CSV: {data}")
        else:
            print(f"🔄 掃到重複條碼 (僅更新儀表板顯示): {data}")

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
        self._target_count = INSP_CFG.get("qr_target_count", 7)
        self._roll_l       = INSP_CFG.get("roll_scan_speed", -15)
        
        self._panel_depth  = INSP_CFG.get("panel_approach_depth", 0.30)
        self._turn_speed   = INSP_CFG.get("turn_180_speed", 35)
        self._turn_tol     = INSP_CFG.get("turn_tolerance_deg", 8)

        # 距離維持配置 (透過 MiDaS PID 控制 fb)
        self._maintain_dist = INSP_CFG.get("maintain_distance_enabled", True)
        self._target_depth  = INSP_CFG.get("target_depth", 0.30)
        self._depth_tol     = INSP_CFG.get("depth_tolerance", 0.03)
        self._depth_kp      = INSP_CFG.get("depth_kp", 100)
        self._max_fb        = INSP_CFG.get("max_fb_speed", 20)

        # QR 視覺輔助定高：不使用高度計，只在看見 QR 時依 bbox 上下位置微調 ud。
        self._qr_height_enabled = INSP_CFG.get("qr_height_enabled", True)
        self._qr_height_deadzone = INSP_CFG.get("qr_height_deadzone", 45)
        self._qr_height_kp = INSP_CFG.get("qr_height_kp", 0.12)
        self._qr_height_max_ud = INSP_CFG.get("qr_height_max_ud", 10)

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
            qr_lr, qr_fb, qr_ud, qr_yaw, bbox, area, reached, decoded, data = \
                self.qr_scanner.process_frame(frame)

            # MiDaS 距離維持 PID 控制 (fb)
            base_fb = 0
            if self._maintain_dist:
                err_depth = self._target_depth - center_depth
                if abs(err_depth) > self._depth_tol:
                    base_fb = int(err_depth * self._depth_kp)
                    base_fb = max(-self._max_fb, min(self._max_fb, base_fb))

            # 預設不上下修正；只有看見 QR bbox 時才用 QR 視覺輔助定高。
            base_ud = 0

            # 預設維持向左飄，並借用 base_fb 穩定距離；ud 預設 0。
            lr, fb, ud, yaw = self._roll_l, base_fb, base_ud, 0

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, f"Locked", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 發現 QR 時，左右/旋轉交給 QRScanner 鎖定
                # 前後(fb) 依然強制維持 MiDaS 的 base_fb
                # 垂直(ud) 改用 QR bbox 的畫面上下位置做「弱修正」，不再使用高度計。
                qr_height_ud = self._calc_qr_height_ud(bbox)
                lr, ud, yaw = qr_lr, qr_height_ud, qr_yaw

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

            if self.qr_scanner.should_abort() or self.qr_scanner.is_timeout():
                # 移除右轉回掃，直接重新開始向左掃
                self.qr_scanner.start()
                lr, fb, ud, yaw = self._roll_l, base_fb, base_ud, 0

            if self._qr_count >= self._target_count:
                if self._face == 1:
                    print("✅ 第一面掃完，準備轉向 180 度")
                    self._state = "TURN_OPPOSITE"
                    self._start_relative_turn(-180)

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
    ARRIVE_CM = RETURN_CFG.get("arrive_radius_cm", 20)
    HOVER_SEC = RETURN_CFG.get("hover_sec", 2.0)
    SPEED = RETURN_CFG.get("fly_speed", 50)
    DESCEND_SPD = RETURN_CFG.get("descend_speed", -10)
    TARGET_H_CM = RETURN_CFG.get("target_height_cm", 50)
    YAW_SPEED = RETURN_CFG.get("yaw_speed", 40)
    RETURN_ALT = RETURN_CFG.get("return_altitude_cm", 150)

    def __init__(self, tello, tracker):
        self.tello = tello
        self.tracker = tracker
        self._phase = "idle"
        self._t = 0.0
        self._last_yaw_err = 0
        self._yaw_int = 0
        self._min_dist = float('inf')  # 記錄回航過程中的最短距離 (抓越過點用)
        self._start_dist = 0.0  # 記錄開始回航時的總距離

    def start(self):
        dist = self.tracker.distance_to_home()
        self._start_dist = dist
        self._min_dist = dist
        print(f"[ReturnHome] 啟動，距起飛點估算距離={dist:.0f}cm")
        self._phase = "fly"
        self._t = time.time()
        self._yaw_int = 0

    def get_rc(self) -> list:
        if self._phase == "idle":
            return [0, 0, 0, 0]

        if self._phase == "climb":
            try:
                h = float(self.tello.get_height())
            except:
                h = 80
            if h < self.RETURN_ALT - 10:
                return [0, 0, RETURN_CFG.get("climb_speed", 20), 0]
            else:
                self._phase = "fly"
                self._t = time.time()
                return [0, 0, 0, 0]

        if self._phase == "fly":
            dx = self.tracker.home[0] - self.tracker.x
            dz = self.tracker.home[2] - self.tracker.z

            # [修正] 移除原本錯誤的 * 2.0 倍率，取真實計算距離
            dist = math.sqrt(dx ** 2 + dz ** 2)

            # 更新歷史最短距離
            if dist < self._min_dist:
                self._min_dist = dist

            # 條件 1：正常進入降落半徑 (完美狀況)
            if dist <= self.ARRIVE_CM:
                print(f"[ReturnHome] 抵達起飛點半徑內 (誤差={dist:.0f}cm)，準備懸停降落")
                self._phase = "hover"
                self._t = time.time()
                return [0, 0, 0, 0]

            # 條件 2：軌跡飄移防護 (Overshoot Detection)
            # 只要距離比歷史最小值增加了 40cm，且已經飛了一段距離，代表已越過真實最近點！
            if dist > self._min_dist + 10 and dist < self._start_dist * 0.8:
                print(f"[ReturnHome] 軌跡飄移保護！已越過最近點(最低 {self._min_dist:.0f}cm)，強制降落")
                self._phase = "hover"
                self._t = time.time()
                return [0, 0, 0, 0]

            tgt_yaw = math.degrees(math.atan2(dx, dz)) if (abs(dx) > 0.1 or abs(dz) > 0.1) else 0
            yaw_err = tgt_yaw - self.tracker.yaw
            while yaw_err > 180:  yaw_err -= 360
            while yaw_err < -180: yaw_err += 360
            p = 0.8 * yaw_err
            self._yaw_int += yaw_err * 0.01
            self._yaw_int = max(-100, min(100, self._yaw_int))
            d = 0.1 * (yaw_err - self._last_yaw_err)
            yaw_cmd = int(max(-self.YAW_SPEED, min(self.YAW_SPEED, p + 0.05 * self._yaw_int + d)))
            self._last_yaw_err = yaw_err

            if abs(yaw_err) < 30:
                # [修正] 提升回航速度，將 /8 改為 /3，且保底速度提高至 20
                fb_v = min(self.SPEED, max(20, int(dist / 3)))
            else:
                fb_v = 0

            try:
                cur_h = float(self.tello.get_height())
            except:
                cur_h = 80
            ud_v = self.DESCEND_SPD if cur_h > self.TARGET_H_CM else 0
            return [0, fb_v, ud_v, yaw_cmd]

        if self._phase == "hover":
            if time.time() - self._t >= self.HOVER_SEC:
                print("[ReturnHome] 降落")
                self._phase = "land"
                self.tello.land()
            return [0, 0, 0, 0]

        return [0, 0, 0, 0]

    def is_active(self):
        return self._phase != "idle"

    def is_landing(self):
        return self._phase == "land"

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

        # 三區塊統一操作頁面：不再建立額外的 OpenCV / Pygame 視窗。
        self.dashboard = MissionDashboard()
        self._latest_depth_disp = None
        self._latest_center_depth = None
        self._dashboard_battery = "--"
        self._dashboard_height = "--"
        self._dashboard_telemetry_t = 0.0

        self.qr_scanner.set_context_provider(self._get_scan_context)

    def get_keyboard_control(self):
        lr = fb = ud = yv = 0
        manual_active = quit_flag = takeoff_cmd = land_cmd = False
        force_state   = None
        switch_mode   = None

        quit_flag = self.dashboard.handle_events()
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
        if keys[pygame.K_5]:      force_state = DroneState.RETURN_HOME

        return (manual_active, lr, fb, ud, yv,
                quit_flag, force_state, takeoff_cmd, land_cmd, switch_mode)

    def _update_depth_view(self, depth_norm, center_depth):
        """更新右上角深度圖快取，供單一儀表板顯示。"""
        depth_u8 = np.clip(depth_norm * 255, 0, 255).astype(np.uint8)
        depth_disp = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        h, w = depth_disp.shape[:2]
        cv2.rectangle(
            depth_disp,
            (w // 3, h // 3),
            (2 * w // 3, 2 * h // 3),
            (255, 255, 255),
            2,
        )
        cv2.putText(
            depth_disp,
            f"CENTER {center_depth:.3f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
        )
        self._latest_depth_disp = depth_disp
        self._latest_center_depth = float(center_depth)

    def _refresh_dashboard_telemetry(self):
        """限制遙測查詢頻率，避免每個影格都向 Tello 送出查詢。"""
        now = time.time()
        if now - self._dashboard_telemetry_t < 0.5:
            return
        self._dashboard_telemetry_t = now
        try:
            self._dashboard_battery = int(self.tello.get_battery())
        except Exception:
            pass
        try:
            self._dashboard_height = int(self.tello.get_height())
        except Exception:
            pass

    def _get_dashboard_info(self, control_cmd, manual_active: bool) -> dict:
        self._refresh_dashboard_telemetry()
        aisle_info = self.inspector.get_hud_info()
        lr, fb, ud, rc_yaw = [int(v) for v in control_cmd]
        mode_label = (
            "1 - CIRCLE INSPECTION"
            if self.mission_mode == MissionMode.MODE1
            else "2 - AISLE INSPECTION"
        )
        control_label = "MANUAL" if manual_active else "AUTO"
        flight_label = (
            f"FLYING / {control_label}" if self.is_flying else "STANDBY"
        )
        depth_text = (
            f"{self._latest_center_depth:.3f}"
            if self._latest_center_depth is not None
            else "--"
        )
        return {
            "is_flying": self.is_flying,
            "flight_label": flight_label,
            "mode_label": mode_label,
            "state": self.current_state,
            "aisle": aisle_info.get("aisle_no", "--"),
            "face": aisle_info.get("face_no", "--"),
            "qr": f"{aisle_info.get('qr_count', 0)}/{aisle_info.get('target_count', 0)}",
            "battery": self._dashboard_battery,
            "height": self._dashboard_height,
            "depth": depth_text,
            "target_depth": f"{float(INSP_CFG.get('target_depth', 0.30)):.3f}",
            "x": f"{self.tracker.x:.1f}",
            "y": f"{self.tracker.y:.1f}",
            "z": f"{self.tracker.z:.1f}",
            "yaw": f"{self.tracker.yaw:.1f}",
            "lr": lr,
            "fb": fb,
            "ud": ud,
            "rc_yaw": rc_yaw,
            "hybrid": "ON" if self.hybrid_enabled else "OFF",
            "auto_count": self._auto_switch_count,
            "scanned_codes": self.qr_scanner.scanned_list  # 🔴 新增：傳遞掃描紀錄
        }

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
        if bat <= LOWBAT_CFG.get("threshold_pct", 10) and not self._low_battery_triggered:
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
                control_cmd = [0, 0, 0, 0]

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
                        stab = CFG.get("takeoff", "stabilize_wait_sec", default=1.5)
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
                        control_cmd = [lr, fb, ud, yv]
                        self.tello.send_rc_control(*control_cmd)
                else:
                    cv2.putText(frame, "STANDBY (Press T to Takeoff)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                self.tracker.draw_minimap(frame)

                if time.time() < self._scanned_popup:
                    h_, w_ = frame.shape[:2]
                    ov = frame.copy()
                    cv2.rectangle(ov, (w_ // 2 - 180, h_ // 2 - 60), (w_ // 2 + 180, h_ // 2 + 60), (0, 200, 0), -1)
                    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, "SCANNED !", (w_ // 2 - 130, h_ // 2 + 20),
                                cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 3)

                dashboard_info = self._get_dashboard_info(
                    control_cmd, manual_active
                )
                self.dashboard.render(
                    frame, self._latest_depth_disp, dashboard_info
                )

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

            self._update_depth_view(depth_norm, center_depth)

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

                # 🔴 新增：在環繞飛行 (CIRCLE) 期間，也即時嘗試解碼
                now = time.time()
                if not hasattr(self, '_circle_last_decode_t'):
                    self._circle_last_decode_t = 0

                # 每 0.3 秒才執行一次解碼，避免運算負載過高導致畫面卡頓
                if now - self._circle_last_decode_t > 0.3:
                    ok, data = decode_qr_from_frame(frame, qr_bbox)
                    if ok and data:
                        # 呼叫你剛才修改過的 _record_decode，直接更新到儀表板與 CSV
                        self.qr_scanner._record_decode(data, qr_bbox, 0)
                        # 觸發畫面中央的 "SCANNED !" 綠色彈跳特效
                        self._scanned_popup = time.time() + 3.0
                    self._circle_last_decode_t = now

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
        self._update_depth_view(depth_norm, center_depth)

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
