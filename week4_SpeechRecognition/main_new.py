import cv2
import numpy as np
from pydobot import Dobot

# --- 1. ตั้งค่าการเชื่อมต่อ Dobot ---
port = "COM10"  # ตรวจสอบพอร์ตใน Device Manager
device = Dobot(port=port, verbose=False)

# ตำแหน่ง Home (x, y, z, r)
home_x, home_y, home_z, home_r = 200, -150, 50, 0
device.move_to(home_x, home_y, home_z, home_r, wait=True)


# --- 2. ฟังก์ชันแปลงพิกเซลเป็นพิกัดหุ่นยนต์ ---
def get_robot_coordinates(cx, cy):
    # สูตรนี้ต้องปรับ (Calibrate) ตามระยะการวางกล้องจริง
    # ตัวอย่าง: robot_x = Offset + (pixel_y * Scale)
    rx = 100 + (cy * 0.5)
    ry = -150 + (cx * 0.5)
    return rx, ry


# --- 3. เริ่มการทำงานของกล้อง ---
cap = cv2.VideoCapture(1)

target_mode = None  # สถานะ: None, 'yellow', 'silver'

print("--- ระบบพร้อมทำงาน ---")
print("กด 'y' : หาและหยิบสีเหลือง")
print("กด 's' : หาและหยิบสีเงิน")
print("กด 'q' : ออกจากโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ดักจับปุ่มกด
    key = cv2.waitKey(1) & 0xFF
    if key == ord('y'):
        target_mode = 'yellow'
        print("\n[Command] กำลังค้นหาสีเหลือง...")
    elif key == ord('s'):
        target_mode = 'silver'
        print("\n[Command] กำลังค้นหาสีเงิน...")
    elif key == ord('q'):
        break

    # ประมวลผลเมื่อเลือกโหมด
    if target_mode:
        if target_mode == 'yellow':
            lower, upper = np.array([20, 100, 100]), np.array([30, 255, 255])
            color_label = (0, 255, 255)
        else:  # silver
            lower, upper = np.array([0, 0, 150]), np.array([180, 50, 255])
            color_label = (200, 200, 200)

        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                # คำนวณพิกัดกล้อง (Pixel)
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # คำนวณพิกัดหุ่นยนต์ (mm)
                rx, ry = get_robot_coordinates(cx, cy)

                # --- แสดงผลพิกัดทั้งสองระบบ ---
                print("-" * 30)
                print(f"Target Detected: {target_mode.upper()}")
                print(f"Camera (Pixel): CX={cx}, CY={cy}")
                print(f"Robot  (mm):    RX={rx:.2f}, RY={ry:.2f}")
                print("-" * 30)

                # วาดวงกลมและพิกัดลงบนหน้าจอกล้อง
                cv2.circle(frame, (cx, cy), 7, color_label, -1)
                cv2.putText(frame, f"Cam: {cx},{cy}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Robot: {rx:.1f},{ry:.1f}", (cx + 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # --- ขั้นตอนการเคลื่อนที่หยิบ ---
                try:
                    # 1. เล็งเหนือวัตถุ
                    device.move_to(rx, ry, 50, 0, wait=True)
                    # 2. ลงไปหยิบ (ปรับค่า Z ตามความสูงโต๊ะจริง)
                    device.move_to(rx, ry, -70, 0, wait=True)
                    device.suck(True)
                    # 3. ยกขึ้น
                    device.move_to(rx, ry, 50, 0, wait=True)
                    # 4. วางลงที่เดิม
                    device.move_to(rx, ry, -65, 0, wait=True)
                    device.suck(False)
                    # 5. กลับจุด Home
                    device.move_to(home_x, home_y, home_z, home_r, wait=True)
                    print(">> ทำงานสำเร็จ: วางวัตถุเรียบร้อย")
                except Exception as e:
                    print(f"!! ข้อผิดพลาด: {e}")

                # เคลียร์สถานะหลังทำงานเสร็จ เพื่อรอคำสั่งถัดไป
                target_mode = None
                break

    # แสดงภาพ Real-time
    cv2.imshow('Robot Vision System', frame)

# ปิดระบบ
cap.release()
cv2.destroyAllWindows()
device.suck(False)
device.close()