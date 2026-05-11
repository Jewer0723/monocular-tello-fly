import cv2
from djitellopy import Tello
from ultralytics import YOLO

# 1. 初始化 YOLO 模型 (建議使用最新的 YOLOv11 或 YOLOv8)
# 如果是第一次運行，系統會自動下載權重文件
model = YOLO('../model/box2.pt')

# 2. 設定偵測信心值 (Confidence Threshold)
# 數值介於 0.0 到 1.0 之間，越高代表越嚴格
CONF_THRESHOLD = 0.5

# 3. 初始化並連接 Tello
tello = Tello()
tello.connect()
tello.streamon()  # 開啟影像串流

print(f"剩餘電量: {tello.get_battery()}%")

try:
    while True:
        # 4. 獲取當前畫面
        frame = tello.get_frame_read().frame

        # 5. 進行 YOLO 物件偵測
        # conf 參數即為你想要調整的信心值
        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)

        # 6. 在影像上繪製結果
        # plot() 會自動畫出框框與標籤
        annotated_frame = results[0].plot()

        # 顯示畫面
        cv2.imshow("Tello YOLO Test", annotated_frame)

        # 按 'q' 退出循環
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 7. 安全關閉
    tello.streamoff()
    cv2.destroyAllWindows()
