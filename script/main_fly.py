import torch
import cv2
import time
import os
import csv
from datetime import datetime
from collections import deque
from djitellopy import Tello
from ultralytics import YOLO
from pyzbar import pyzbar
import manual_control
import midas_utils
import right_turn_utils

# =============================
# 參數設定
# =============================

FRAME_W, FRAME_H = 640, 480

# Forward / Circle PID
KP_YAW = 0.25
KP_UPDOWN = 0.25
KP_FORWARD = 0.0006

TARGET_AREA = 120000
AREA_TOLERANCE = 15000
DEADZONE = 20

MAX_SPEED = 20
ORBIT_SPEED = 10

# 避障參數
OBSTACLE_THRESHOLD = 0.35
CLEAR_THRESHOLD = 0.25
TURN_DURATION = 1.5

BASE_FORWARD = 20
TURN_SPEED = 40

# =============================
# FSM 定義
# =============================

class State:
    SEARCH = 0
    FORWARD = 1
    CIRCLE = 2

state = State.SEARCH

# =============================
# 初始化 YOLO
# =============================

model = YOLO("../model/box1.pt")

# =============================
# 初始化 MiDaS
# =============================

device = "cuda" if torch.cuda.is_available() else "cpu"

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device).eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# =============================
# 初始化 Tello
# =============================

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())
tello.streamon()
frame_reader = tello.get_frame_read()

controller = right_turn_utils.RightTurnAvoider()

# =============================
# CSV 初始化
# =============================

CSV_FILE = "scanned_codes.csv"
scanned_set = set()

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Data"])

# =============================
# 工具函式
# =============================

def clamp(val, minv, maxv):
    return max(minv, min(maxv, val))

def detect_target(frame):
    results = model(frame, conf=0.6, verbose=False)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None, None

    best_box = max(
        results[0].boxes,
        key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) *
                      (b.xyxy[0][3] - b.xyxy[0][1])
    )

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    area = (x2 - x1) * (y2 - y1)

    return (x1, y1, x2, y2, cx, cy), area, results

def compute_depth(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

    center_val, _, _ = midas_utils.get_depth_regions(depth_norm)
    return center_val

def safety_layer(fb, yaw, depth):
    if depth > OBSTACLE_THRESHOLD:
        print("⚠ Obstacle detected")
        fb = 0
        yaw = TURN_SPEED
    return fb, yaw

# =============================
# 主迴圈
# =============================

print("Mission Start")

while True:

    frame = frame_reader.frame
    if frame is None:
        continue

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    # =============================
    # 手動優先
    # =============================

    manual, lr, fb, ud, yv, quit_flag = manual_control.get_keyboard_control(tello)
    if quit_flag:
        break

    if manual:
        tello.send_rc_control(lr, fb, ud, yv)
        cv2.imshow("Mission", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # =============================
    # 感知
    # =============================

    bbox, area, _ = detect_target(frame)
    depth = compute_depth(frame)

    lr = fb = ud = yaw = 0

    # =============================
    # FSM 控制
    # =============================

    if state == State.SEARCH:
        yaw = 40  # 原地旋轉搜尋
        if bbox is not None:
            state = State.FORWARD

    elif state == State.FORWARD:
        if bbox is None:
            state = State.SEARCH
        elif area >= TARGET_AREA:
            state = State.CIRCLE
        else:
            x1, y1, x2, y2, cx, cy = bbox
            error_x = cx - FRAME_W // 2
            error_y = cy - FRAME_H // 2
            error_area = TARGET_AREA - area

            if abs(error_x) > DEADZONE:
                if abs(error_x) > 120:
                    lr = clamp(int(KP_YAW * error_x), -MAX_SPEED, MAX_SPEED)
                else:
                    yaw = clamp(int(KP_YAW * error_x), -MAX_SPEED, MAX_SPEED)

            if abs(error_y) > DEADZONE:
                ud = clamp(int(-KP_UPDOWN * error_y), -MAX_SPEED, MAX_SPEED)

            if abs(error_area) > AREA_TOLERANCE:
                fb = clamp(int(KP_FORWARD * error_area), -MAX_SPEED, MAX_SPEED)

    elif state == State.CIRCLE:
        if bbox is None:
            state = State.SEARCH
        else:
            x1, y1, x2, y2, cx, cy = bbox
            error_x = cx - FRAME_W // 2
            error_y = cy - FRAME_H // 2

            if abs(error_x) > DEADZONE:
                yaw = clamp(int(KP_YAW * error_x), -MAX_SPEED, MAX_SPEED)

            if abs(error_y) > DEADZONE:
                ud = clamp(int(-KP_UPDOWN * error_y), -MAX_SPEED, MAX_SPEED)

            lr = ORBIT_SPEED

            # QR 掃描
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            barcodes = pyzbar.decode(gray)

            for barcode in barcodes:
                data = barcode.data.decode("utf-8")
                if data not in scanned_set:
                    scanned_set.add(data)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, data])
                    print("Saved:", data)

    # =============================
    # Safety Layer
    # =============================

    fb, yaw = safety_layer(fb, yaw, depth)

    # =============================
    # 發送控制
    # =============================

    tello.send_rc_control(lr, fb, ud, yaw)

    cv2.putText(frame, f"STATE: {state}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Mission", frame)

    if cv2.waitKey(1) == 27:
        break

# =============================
# 結束
# =============================

tello.send_rc_control(0,0,0,0)
time.sleep(0.5)
tello.land()
tello.streamoff()
cv2.destroyAllWindows()