import cv2
import numpy as np
from pydobot import Dobot
import time
import speech_recognition as sr
import keyboard

# --- 1. การเตรียมข้อมูล Calibration (จาก Code A) ---
pts_camera = np.array([
    [204, 141], [525, 138], [230, 360], [535, 389]
], dtype="float32")

pts_robot = np.array([
    [220, 44], [229, -88], [109, 26], [109, -81]
], dtype="float32")

matrix = cv2.getPerspectiveTransform(pts_camera, pts_robot)


def convert_to_robot(cx, cy):
    point = np.array([[[cx, cy]]], dtype="float32")
    transformed = cv2.perspectiveTransform(point, matrix)
    rx = transformed[0][0][0]
    ry = transformed[0][0][1]
    rx = rx + 5
    ry = ry - 0  # ค่า Offset ที่ปรับจูน
    return rx, ry


# --- 2. ตั้งค่าระบบเสียง ---
r = sr.Recognizer()
mic = sr.Microphone()

# --- 3. ตั้งค่าหุ่นยนต์และตำแหน่ง Drop/Wait ---
WAIT_X, WAIT_Y, WAIT_Z = 90, -100, 50

try:
    device = Dobot(port='COM13', verbose=False)
    print(f">> Robot Ready! กำลังไปจุด Standby...")
    device.move_to(WAIT_X, WAIT_Y, WAIT_Z, 0, wait=True)
except Exception as e:
    print(f">> เชื่อมต่อหุ่นยนต์ไม่ได้: {e}")
    exit()

cap = cv2.VideoCapture(1)
target_color = None
status_msg = "Ready: Press 'g' to Order"

print("\n=== ระบบพร้อมทำงานแบบวนลูป ===")
print("1. คลิกที่หน้าต่างกล้อง")
print("2. กด 'g' เพื่อพูดชื่อสี (เหลือง, แดง, น้ำเงิน)")
print("3. กด 'q' เพื่อออกจากโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret: break

    # แสดงสถานะบนหน้าจอเฟรมกล้อง
    cv2.putText(frame, f"Status: {status_msg}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- ส่วนที่ 1: รับคำสั่งเสียง (กด g เพื่อสั่ง) ---
    if keyboard.is_pressed('g'):
        status_msg = "Listening..."
        cv2.imshow('Dobot Control Center', frame)  # อัปเดตหน้าจอให้เห็นว่ากำลังฟัง

        with mic as source:
            print("\n[ฟังเสียง...] พูดชื่อสีได้เลย (เหลือง/แดง/น้ำเงิน)")
            audio = r.listen(source, phrase_time_limit=3)
        try:
            text = r.recognize_google(audio, language='th-TH')
            print("ข้อความ: " + text)
            if text == "เหลือง":
                status_msg = "Target: Yellow"
                target_color = "yellow"

            elif text == "แดง":
                status_msg = "Target: Red"
                target_color = "red"

            elif text == "น้ำเงิน":
                status_msg = "Target: Blue"
                target_color = "blue"
            #################################################
            # รับค่าเสียง แปลงเป็นข้อความ เขียน if elif กำหนด        #
            # status_msg = "Target: ?????????"              #
            # target_color = "????????                      #
            #################################################

        except:
            status_msg = "Speech not clear"
            target_color = None

        time.sleep(0.5)  # หน่วงเวลานิดหน่อยให้เห็นสถานะ

    # --- ส่วนที่ 2: การประมวลผล Vision และสั่งงานหุ่นยนต์ ---
    if target_color:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ตั้งค่า Range สี HSV
        if target_color == "yellow":
            lower, upper = np.array([15, 100, 100]), np.array([35, 255, 255])
        elif target_color == "red":
            # สีแดงครอบคลุม 2 ช่วงใน HSV
            m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)
        elif target_color == "blue":
            lower, upper = np.array([100, 100, 100]), np.array([130, 255, 255])

        if target_color != "red":
            mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                M = cv2.moments(c)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                rx, ry = convert_to_robot(cx, cy)
                print(f">> เจอสี{target_color}! กำลังไปหยิบ...")

                # ลำดับการทำงาน
                device.move_to(rx, ry, 50, 0, wait=True)
                device.move_to(rx, ry, -55, 0, wait=True)
                device.suck(True)
                time.sleep(0.5)
                device.move_to(rx, ry, 50, 0, wait=True)

                # นำไปวางที่จุด Standby (90, -100, 50)
                device.move_to(WAIT_X, WAIT_Y, 50, 0, wait=True)
                device.move_to(WAIT_X, WAIT_Y, WAIT_Z, 0, wait=True)
                device.suck(False)
                time.sleep(0.5)
                device.move_to(WAIT_X, WAIT_Y, 50, 0, wait=True)

                print(">> วางของเรียบร้อย กลับมารอรับคำสั่งใหม่")
                status_msg = "Done! Press 'g' for next"
            else:
                status_msg = "Object too small"
        else:
            status_msg = f"Cannot see {target_color}"
            print(f">> ไม่เห็นวัตถุสี{target_color}")

        target_color = None  # วนกลับมารับคำสั่งใหม่

    cv2.imshow('Dobot Control Center', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
device.close()