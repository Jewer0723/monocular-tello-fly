from djitellopy import Tello
import cv2
from pyzbar import pyzbar

FRAME_W, FRAME_H = 640, 480

tello = Tello()
tello.connect()
tello.streamon()

while True:
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    barcodes = pyzbar.decode(frame)

    for barcode in barcodes:
        x, y, w, h = barcode.rect
        data = barcode.data.decode("utf-8")

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, data, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Tello QR/Barcode", frame)

    if cv2.waitKey(1) == 27:
        break

tello.streamoff()
cv2.destroyAllWindows()