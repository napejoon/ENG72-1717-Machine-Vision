import cv2


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"คลิกที่ Pixel: X={x}, Y={y}")

cap = cv2.VideoCapture(1) # หรือเปลี่ยนเป็น 0 ถ้าไม่เจอ
if not cap.isOpened():
    print("เปิดกล้องไม่สำเร็จ")
    exit()

cv2.namedWindow("Calibration Board")
cv2.setMouseCallback("Calibration Board", click_event)

while True:
    ret, frame = cap.read()
    cv2.imshow("Calibration Board", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()