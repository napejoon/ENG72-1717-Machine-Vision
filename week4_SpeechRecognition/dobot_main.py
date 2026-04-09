import cv2
import numpy as np
from pydobot import Dobot
from pydobot.message import Message
import time

# ==========================================
# 1. การตั้งค่าระบบ (CONFIG)
# ==========================================
COM_PORT = 'COM10'  # ตรวจสอบ Port ใน Device Manager อีกครั้ง
SAFE_Z = 50  # ความสูงปลอดภัย (ยกแขนสูง)
PICK_Z = -40  # ความสูงตอนจิ้มลงไปดูด (ปรับตามความหนาวัตถุ)

BOX_POS = {'x': 145, 'y': 200}

# --- ส่วนการแปลงพิกัด (Calibration) ---
# ค่าเหล่านี้ต้องปรับตามความสูงกล้อง 50 ซม. ของคุณ
PIXEL_TO_MM_RATIO = 0.60  # 1 pixel = 0.65 mm (โดยประมาณ)
OFFSET_X = 50  # ระยะห่างจากฐานหุ่นยนต์ถึงขอบล่างของภาพ
OFFSET_Y = 50  # ระยะเบี่ยงเบนซ้าย-ขวา

# ==========================================
# 2. ตั้งค่าสี HSV (ปรับปรุงใหม่ตามสภาพแสงจริง)
# ==========================================
color_ranges = {
    'yellow': {
        'lower': np.array([10, 80, 80]),   # ขยายขอบเขตลงมาครอบคลุมสีเหลืองอมส้ม
        'upper': np.array([45, 255, 255]) 
    },
    'black': {
        'lower': np.array([0, 0, 0]),
        'upper': np.array([180, 255, 90])  # เพิ่ม V เป็น 90 เผื่อสีดำสะท้อนแสงไฟ
    },
    'silver': {
        'lower': np.array([0, 0, 70]),     # ดัน V ต่ำสุดขึ้นมาเพื่อหนีเงาดำ
        'upper': np.array([180, 60, 180])  # ล็อก V สูงสุดไว้ที่ไม่เกิน 180 เพื่อไม่ให้ไปจับโดน "กระดาษสีขาว"
    }
}

# ==========================================
# 3. ฟังก์ชันประมวลผลภาพและการคำนวณ (Vision & Logic)
# ==========================================
def find_object(frame, color_name):
    if color_name not in color_ranges:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    lower = color_ranges[color_name]['lower']
    upper = color_ranges[color_name]['upper']
    mask = cv2.inRange(blurred, lower, upper)

    # ทำความสะอาด Mask (ลบ Noise)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow("Debug Mask (White = Object found)", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy, c
    return None

def pixel_to_robot(px, py):
    # แปลงพิกัดภาพเป็นมิลลิเมตร
    rx = (py * PIXEL_TO_MM_RATIO) + OFFSET_X
    ry = (px * PIXEL_TO_MM_RATIO) + OFFSET_Y
    return round(rx, 1), round(ry, 1)

# 🚨 เพิ่มฟังก์ชันตรวจสอบระยะการทำงาน (Boundary Check)
def is_within_workspace(x, y, z):
    # คำนวณระยะห่างจากฐาน (Base)
    distance_from_base = np.sqrt(x**2 + y**2)
    
    # ระยะปลอดภัยของ Dobot (รัศมี X, Y) ไม่ควรใกล้ฐานเกิน 150mm และไม่ไกลเกิน 320mm
    if distance_from_base < 150 or distance_from_base > 320:
        return False
    # ป้องกันการพุ่งชนพื้นโต๊ะ
    if z < -60: 
        return False
        
    return True

def auto_home(device):
    print("🏠 กำลัง Set Home อัตโนมัติ (หุ่นยนต์จะขยับแขน กรุณารอ 20 วินาที)...")
    try:
        msg = Message()
        msg.id = 31  
        msg.ctrl = 0x03
        msg.params = bytearray([])
        device._send_command(msg)
    except Exception as e:
        pass 
    
    time.sleep(20) 
    print("✅ Set Home เสร็จเรียบร้อย! พร้อมทำงาน")

# ==========================================
# 4. ส่วนรันโปรแกรมหลัก (Main Process)
# ==========================================
def main():
    # --- เริ่มต้นเชื่อมต่อหุ่นยนต์ ---
    try:
        print("🤖 กำลังเชื่อมต่อ DOBOT...")
        device = Dobot(port=COM_PORT, verbose=False)
        print("✅ เชื่อมต่อสำเร็จ! (ไฟเขียว)")
        device.move_to(200, 0, SAFE_Z, 0, wait=True)
    except Exception as e:
        print(f"❌ เชื่อมต่อหุ่นยนต์ไม่ได้: {e}")
        return
    
    auto_home(device)

    device.move_to(150, -200, 50, 0, wait=True)

    # --- เริ่มต้นกล้อง ---
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ เปิดกล้องไม่ได้")
        device.close()
        return

    print("\n--- พร้อมรับคำสั่ง ---")
    print("คำสั่งที่รองรับ: yellow, black, silver (พิมพ์ 'q' เพื่อเลิก)")

    try:
        while True:
            # โชว์ภาพสด
            ret, frame = cap.read()
            if not ret: break

            cv2.putText(frame, "Waiting for command in Console...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Robot Vision", frame)
            cv2.waitKey(1)

            # รับคำสั่งผ่าน PyCharm Console
            color_cmd = input("\nใส่สีที่ต้องการหยิบ > ").lower().strip()

            if color_cmd == 'q':
                break

            # จับภาพใหม่เพื่อประมวลผล
            ret, frame = cap.read()
            result = find_object(frame, color_cmd)

            if result:
                px, py, cnt = result
                rx, ry = pixel_to_robot(px, py)

                # วาดวงกลมยืนยันในจอ
                cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)
                cv2.circle(frame, (px, py), 7, (0, 255, 0), -1)
                cv2.imshow("Detection Result", frame)
                cv2.waitKey(1000)

                print(f"🎯 พบวัตถุ {color_cmd} | พิกัดหุ่นยนต์: X={rx}, Y={ry}")

                # 🚨 ตรวจสอบว่าพิกัดอยู่ในระยะทำการหรือไม่ก่อนสั่งขยับ
                if not is_within_workspace(rx, ry, PICK_Z):
                    print(f"🚫 ยกเลิกการหยิบ: พิกัดเป้าหมาย (X:{rx}, Y:{ry}) อยู่นอกระยะแขนหุ่นยนต์!")
                    print("💡 คำแนะนำ: ลองขยับวัตถุให้เข้ามาตรงกลาง หรือปรับสมการ OFFSET ในโค้ดดูครับ")
                    continue  # วนลูปกลับไปรอรับคำสั่งใหม่ทันที โดยที่หุ่นยนต์ไม่ขยับและไม่ค้าง

                # --- ลำดับการเคลื่อนที่ ---
                try:
                    device.move_to(rx, ry, SAFE_Z, 0, wait=True)  # ไปเหนือวัตถุ
                    device.move_to(rx, ry, PICK_Z, 0, wait=True)  # ลงไปจิ้ม
                    device.suck(True)  # เปิดลมดูด
                    time.sleep(0.5)
                    device.move_to(rx, ry, SAFE_Z, 0, wait=True)  # ยกของขึ้น

                    # ไปที่กล่อง
                    device.move_to(BOX_POS['x'], BOX_POS['y'], SAFE_Z, 0, wait=True)
                    device.suck(False)  # วางของ
                    time.sleep(0.5)

                    print("✅ สำเร็จ! กำลังกลับจุดพัก...")
                    device.move_to(200, 0, SAFE_Z, 0, wait=True)
                except Exception as e:
                    print(f"⚠️ พบปัญหาขณะสั่งหุ่นยนต์ขยับ: {e}")

            else:
                print(f"⚠️ ไม่พบวัตถุสี '{color_cmd}' ในหน้าจอ")

    except KeyboardInterrupt:
        print("\nหยุดการทำงาน...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        device.suck(False)
        device.close()
        print("🔌 ปิดการเชื่อมต่อเรียบร้อย")

if __name__ == "__main__":
    main()