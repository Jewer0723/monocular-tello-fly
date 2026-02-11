import cv2
import numpy as np

# =====================
# 視覺化函數
# =====================
def draw_simple_overlay(frame, center_depth, state, fbv, yv, obstacle_count, OBSTACLE_THRESHOLD, CLEAR_THRESHOLD):
    """繪製簡單疊加層"""
    h, w = frame.shape[:2]

    # 狀態顏色
    if state == "TURNING":
        color = (0, 165, 255)  # 橙色
        status_text = "TURNING RIGHT"
    else:
        if center_depth > OBSTACLE_THRESHOLD:
            color = (0, 0, 255)  # 紅色
            status_text = "OBSTACLE!"
        elif center_depth > CLEAR_THRESHOLD:
            color = (0, 255, 255)  # 黃色
            status_text = "CAUTION"
        else:
            color = (0, 255, 0)  # 綠色
            status_text = "CLEAR"

    # 繪製中心區域
    cv2.rectangle(frame, (w // 3, h // 3), (2 * w // 3, 2 * h // 3), color, 2)

    # 顯示狀態信息
    y_offset = 30
    cv2.putText(frame, f"status: {status_text}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"depth: {center_depth:.3f}", (10, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"control: FB={fbv:2d} Yaw={yv:2d}", (10, y_offset + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"danger count: {obstacle_count}", (10, y_offset + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    # 深度條
    bar_w = 200
    bar_h = 20
    bar_x = w - bar_w - 20
    bar_y = 30

    # 當前深度
    depth_pos = int(np.clip(center_depth, 0, 1) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + depth_pos, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

    # 閾值標記
    cv2.line(frame, (bar_x + int(OBSTACLE_THRESHOLD * bar_w), bar_y - 5),
             (bar_x + int(OBSTACLE_THRESHOLD * bar_w), bar_y + bar_h + 5),
             (255, 0, 0), 2)

    cv2.line(frame, (bar_x + int(CLEAR_THRESHOLD * bar_w), bar_y - 5),
             (bar_x + int(CLEAR_THRESHOLD * bar_w), bar_y + bar_h + 5),
             (0, 255, 0), 2)

    cv2.putText(frame, "danger", (bar_x + int(OBSTACLE_THRESHOLD * bar_w) - 20, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.putText(frame, "safe", (bar_x + int(CLEAR_THRESHOLD * bar_w) - 20, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 在轉向時顯示箭頭
    if state == "TURNING":
        # 繪製右轉箭頭
        arrow_center = (w // 2, h // 2)
        arrow_size = 50

        # 箭頭線
        cv2.arrowedLine(frame,
                        (arrow_center[0] - arrow_size, arrow_center[1]),
                        (arrow_center[0] + arrow_size, arrow_center[1]),
                        (0, 165, 255), 3, tipLength=0.3)

        cv2.putText(frame, "RIGHT TURN", (arrow_center[0] - 60, arrow_center[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    return frame