from djitellopy import Tello
from ultralytics import YOLO
import cv2
from pyzbar import pyzbar
import manual_control
import csv
import os
from datetime import datetime

# ======================
# 參數設定
# ======================

FRAME_W, FRAME_H = 640, 480

KP_YAW = 0.25
KP_UPDOWN = 0.25
KP_FORWARD = 0.0006

TARGET_AREA = 120000
AREA_TOLERANCE = 15000
DEADZONE = 20

MAX_SPEED = 20
ORBIT_SPEED = 10

def clamp(val, minv, maxv):
    return max(minv, min(maxv, val))

# ======================
# 初始化
# ======================

model = YOLO("../model/box4.pt")

tello = Tello()
tello.connect()
tello.streamon()

print("Battery:", tello.get_battery())

CSV_FILE = "scanned_codes.csv"

# 已掃描過的資料集合（避免重複）
scanned_set = set()

# 如果檔案不存在就建立並寫入標題
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Data"])

# ======================
# 主迴圈
# ======================

while True:
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    # 預設自動控制值
    yaw = 0
    up_down = 0
    forward = 0
    left_right = 0

    # ======================
    # YOLO 偵測
    # ======================

    results = model(frame, conf=0.6, verbose=False)

    if results[0].boxes is not None and len(results[0].boxes) > 0:

        boxes = results[0].boxes

        best_box = max(
            boxes,
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) *
                          (b.xyxy[0][3] - b.xyxy[0][1])
        )

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        bbox_cx = (x1 + x2) // 2
        bbox_cy = (y1 + y2) // 2
        bbox_area = (x2 - x1) * (y2 - y1)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # 誤差
        error_x = bbox_cx - FRAME_W // 2
        error_y = bbox_cy - FRAME_H // 2
        error_area = TARGET_AREA - bbox_area

        # yaw
        if abs(error_x) > DEADZONE:
            yaw = clamp(int(KP_YAW * error_x), -MAX_SPEED, MAX_SPEED)

        # 上下
        if abs(error_y) > DEADZONE:
            up_down = clamp(int(-KP_UPDOWN * error_y), -MAX_SPEED, MAX_SPEED)

        # 前後
        if abs(error_area) > AREA_TOLERANCE:
            forward = clamp(int(KP_FORWARD * error_area), -MAX_SPEED, MAX_SPEED)

        # 右環繞
        left_right = ORBIT_SPEED

        # QR 掃描
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)

        for barcode in barcodes:
            x, y, w, h = barcode.rect
            data = barcode.data.decode("utf-8")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, data, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # ⭐ 如果是新條碼才寫入
            if data not in scanned_set:
                scanned_set.add(data)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(CSV_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, data])

                print("✅ 已寫入 CSV:", data)

    # ======================
    # ⭐ 手動接管區
    # ======================

    manual_active, lr_m, fb_m, ud_m, yv_m, quit_flag = \
        manual_control.get_keyboard_control(tello)

    if quit_flag:
        break

    if manual_active:
        # 👉 人工優先
        tello.send_rc_control(lr_m, fb_m, ud_m, yv_m)

        cv2.putText(frame, "MANUAL MODE", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        # 👉 自動模式
        tello.send_rc_control(left_right, forward, up_down, yaw)

        cv2.putText(frame, "AUTO ORBIT MODE", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Orbit + Manual Backup", frame)

    if cv2.waitKey(1) == 27:
        break


# ======================
# 結束
# ======================

tello.land()
tello.streamoff()
cv2.destroyAllWindows()