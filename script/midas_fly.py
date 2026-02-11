import torch
import cv2
import numpy as np
from djitellopy import Tello
import time
from collections import deque
import manual_control
import cv2_utils
import midas_utils
import right_turn_utils

# =====================
# 裝置設定
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =====================
# 飛行參數（修正版）
# =====================
BASE_FORWARD = 20  # 前進速度
TURN_SPEED = 40  # 轉向速度

# =====================
# 關鍵修正：深度值理解
# =====================
OBSTACLE_THRESHOLD = 0.35   # 遇到障礙物的閾值（高於此值表示太近）
CLEAR_THRESHOLD = 0.25      # 安全前進的閾值（低於此值表示安全）
TURN_DURATION = 1.5         # 每次轉向的持續時間（秒）

SMOOTHING_WINDOW = 5
CONTROL_INTERVAL = 0.1

FRAME_W, FRAME_H = 640, 480

# =====================
# 載入 MiDaS
# =====================
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device).eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# =====================
# 初始化 Tello
# =====================
tello = Tello()
tello.connect()
print("電池電量:", tello.get_battery(), "%")

tello.streamon()
frame_reader = tello.get_frame_read()

tello.set_speed(50)

tello.send_rc_control(0, 0, 0, 0)
time.sleep(0.5)

# =====================
# 平滑佇列
# =====================
center_queue = deque(maxlen=SMOOTHING_WINDOW)
left_queue = deque(maxlen=SMOOTHING_WINDOW)
right_queue = deque(maxlen=SMOOTHING_WINDOW)

# =====================
# 主迴圈
# =====================
controller = right_turn_utils.RightTurnAvoider()

print("開始簡單右轉避障巡航...")
print("按 ESC 鍵結束")
print(f"障礙閾值: {OBSTACLE_THRESHOLD:.2f}")
print(f"安全閾值: {CLEAR_THRESHOLD:.2f}")
print(f"轉向時間: {TURN_DURATION}秒")
print("規則: 遇到障礙物 → 停止並向右轉 → 安全後繼續前進")

last_control_time = time.time()
frame_count = 0

try:
    while True:
        # 讀取幀
        frame = frame_reader.frame
        if frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_count += 1

        # ---------- MiDaS 深度計算 ----------
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

        # 使用原始MiDaS輸出：高值=近，低值=遠
        depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

        # ---------- 獲取區域深度 ----------
        center_val, left_val, right_val = midas_utils.get_depth_regions(depth_norm)

        # 平滑處理
        center_queue.append(center_val)
        left_queue.append(left_val)
        right_queue.append(right_val)

        center_avg = np.mean(center_queue)

        # ---------- 狀態更新和控制決策 ----------
        state = controller.update_state(center_avg, OBSTACLE_THRESHOLD, TURN_DURATION, CLEAR_THRESHOLD)
        fbv, yv = controller.get_control(state, BASE_FORWARD, TURN_SPEED)

        # 限制速度範圍
        fbv = int(np.clip(fbv, 0, 30))  # 只前進，不後退
        yv = int(np.clip(yv, 0, 50))  # 只向右轉

        # ---------- 發送控制指令 ----------
        manual, lr_k, fb_k, ud_k, yv_k, quit_flag = manual_control.get_keyboard_control(tello)
        if quit_flag:
            break

        if manual:
            tello.send_rc_control(lr_k, fb_k, ud_k, yv_k)
            last_control_time = time.time()
        else :
            current_time = time.time()
            if current_time - last_control_time >= CONTROL_INTERVAL:
                tello.send_rc_control(0, fbv, 0, yv)  # 沒有滾轉，沒有高度控制
                last_control_time = time.time()

        # ---------- 視覺化 ----------
        # 深度圖
        depth_display = cv2.applyColorMap(
            (depth_norm * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # 原始畫面疊加層
        frame_overlay = cv2_utils.draw_simple_overlay(
            frame, center_avg, state, fbv, yv, controller.obstacle_count, OBSTACLE_THRESHOLD, CLEAR_THRESHOLD
        )

        cv2.imshow("Tello Camera", frame_overlay)
        cv2.imshow("Depth Map", depth_display)

        # 每30幀顯示一次深度信息
        if frame_count % 30 == 0:
            print(f"fps {frame_count}: depth={center_avg:.3f}, "
                  f"status={state}, FB={fbv}, Yaw={yv}")

        # ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            print("使用者中斷...")
            break

        time.sleep(0.03)

except KeyboardInterrupt:
    print("鍵盤中斷...")

except Exception as e:
    print(f"錯誤: {e}")
    import traceback

    traceback.print_exc()

finally:
    print("降落...")
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()
    print("完成")
    print(f"總共遇到 {controller.obstacle_count} 次障礙物")