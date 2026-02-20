import cv2
from djitellopy import Tello
from ultralytics import YOLO

# 1. 載入模型
model = YOLO('../model/barcode1.pt')

# 2. 自動獲取指定類別的 ID (對應模型內部的索引)
target_names = ['cardboard box', 'barcode', 'qr code']
target_ids = [k for k, v in model.names.items() if v in target_names]

print(f"偵測目標: {target_names} (對應 ID: {target_ids})")

# 3. 初始化 Tello
me = Tello()
me.connect()
me.streamon()

try:
    while True:
        frame = me.get_frame_read().frame
        if frame is None: continue

        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 4. 使用 classes 參數進行過濾
        # 只有在 target_ids 列表中的類別會被偵測與處理
        results = model(frame, verbose=False)

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                detections.append({
                    "area": area,
                    "coords": (x1, y1, x2, y2),
                    "conf": float(box.conf[0]),
                    "cls": int(box.cls[0])
                })

        # 5. 顯示結果
        if detections:
            # 使用 min 函數搭配 lambda 找出 area 最小的項目
            smallest_item = min(detections, key=lambda x: x['area'])

            x1, y1, x2, y2 = smallest_item["coords"]
            label = f"SMALLEST: {model.names[smallest_item['cls']]} {smallest_item['conf']:.2f}"

            # 繪製該最小目標
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 紅色框強調
            cv2.putText(display_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 在畫面角落顯示該面積數值供參考
            cv2.putText(display_frame, f"Min Area: {smallest_item['area']}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Smallest Target Only", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    me.streamoff()
    cv2.destroyAllWindows()