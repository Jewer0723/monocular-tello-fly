from djitellopy import Tello
from ultralytics import YOLO
import cv2
from ..manual import manual_control

# =======================
# 參數設定
# =======================
FRAME_W, FRAME_H = 640, 480

TARGET_AREA = 120000        # 到達包裹的面積閥值（需依實際調）
AREA_TOLERANCE = 15000

KP_YAW = 0.25
KP_UPDOWN = 0.25
KP_FORWARD = 0.0006

MAX_SPEED = 20
DEADZONE = 20

# =======================
# 初始化 YOLO
# =======================
model = YOLO("../../model/box1.pt")

# =======================
# 初始化 Tello
# =======================
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())
tello.streamon()

manual_mode = True

def clamp(val, minv, maxv):
    return max(min(val, maxv), minv)

# =======================
# 主迴圈
# =======================
while True:
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    # ========= 手動控制（最高優先權） =========
    manual_active, lr, fb, ud, yv, exit_flag = manual_control.get_keyboard_control(tello)

    if exit_flag:
        tello.land()
        break

    # 預設控制值
    yaw = 0
    up_down = 0
    forward = 0
    left_right = 0

    # ========= 手動模式 =========
    if manual_active:
        left_right = lr
        forward = fb
        up_down = ud
        yaw = yv

        mode_text = "MANUAL"

    # ========= 自動追蹤模式 =========
    else:
        results = model(frame, conf=0.6, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # 選擇面積最大的 bbox
            best_box = max(
                boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) *
                              (b.xyxy[0][3] - b.xyxy[0][1])
            )

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2
            bbox_area = (x2 - x1) * (y2 - y1)

            # 視覺化
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (bbox_cx, bbox_cy), 5, (0,0,255), -1)

            error_x = bbox_cx - FRAME_W // 2
            error_y = bbox_cy - FRAME_H // 2
            error_area = TARGET_AREA - bbox_area

            if abs(error_x) > DEADZONE:
                yaw = clamp(int(KP_YAW * error_x), -MAX_SPEED, MAX_SPEED)

            if abs(error_y) > DEADZONE:
                up_down = clamp(int(-KP_UPDOWN * error_y), -MAX_SPEED, MAX_SPEED)

            if abs(error_area) > AREA_TOLERANCE:
                forward = clamp(int(KP_FORWARD * error_area), -MAX_SPEED, MAX_SPEED)

        mode_text = "AUTO"

    # ========= 發送指令 =========
    tello.send_rc_control(left_right, forward, up_down, yaw)

    cv2.putText(frame, mode_text, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Tello YOLO Tracker", frame)

    if cv2.waitKey(1) == 27:
        tello.land()
        break

# =======================
# 收尾
# =======================
tello.land()
tello.streamoff()
cv2.destroyAllWindows()