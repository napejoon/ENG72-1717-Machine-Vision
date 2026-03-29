import os
import cv2
import torch
import threading  # นำเข้า Threading เพื่อแยกการทำงาน
from groundingdino.util.inference import load_model, load_image, predict, annotate

# CPU process
DEVICE = "cuda"
CONFIG_PATH = "ENG35-1717-Machine-Vision/week3_grounding_dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "ENG35-1717-Machine-Vision/week3_grounding_dino/GroundingDINO/weights/groundingdino_swint_ogc.pth"

print("="*50)
print("Interactive Real-time Grounding DINO (Smooth Threaded)")
print("="*50)

print("\nLoading model into CPU (Please wait)... ")
model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

TEMP_IMAGE_PATH = "temp_capture.jpg"
TEXT_PROMPT = ""

# --- ตัวแปรสำหรับ Threading ---
latest_boxes = None
latest_logits = None
latest_phrases = None
is_processing = False  # ตัวเช็คว่า AI กำลังคิดอยู่ไหม

def run_dino_thread(frame_for_ai, prompt):
    """ฟังก์ชันนี้จะทำงานอยู่เบื้องหลัง เพื่อไม่ให้กล้องกระตุก"""
    global latest_boxes, latest_logits, latest_phrases, is_processing
    
    cv2.imwrite(TEMP_IMAGE_PATH, frame_for_ai)
    image_source, image = load_image(TEMP_IMAGE_PATH)

    # ให้ AI ทำนายผล (ใช้เวลา 1-3 วินาที)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )
    
    # อัปเดตกรอบใหม่ และบอกว่า AI ว่างแล้ว
    latest_boxes = boxes
    latest_logits = logits
    latest_phrases = phrases
    is_processing = False

print("\n" + "="*50)
print("📷 Camera is OPEN!")
print("🟢 Press 's' on your keyboard to type what you want to find.")
print("🔴 Press 'q' on your keyboard to quit the program.")
print("="*50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1024, 720))
    display_frame = frame.copy()

    if TEXT_PROMPT != "":
        # 1. ถ้า AI "ว่าง" ให้โยนภาพเฟรมปัจจุบันไปให้ AI คิดเบื้องหลัง
        if not is_processing:
            is_processing = True
            # สั่งให้ทำงานเบื้องหลัง
            thread = threading.Thread(target=run_dino_thread, args=(frame.copy(), TEXT_PROMPT))
            thread.daemon = True
            thread.start()

        # 2. ถ้าระหว่างนี้ AI เคยตีกรอบมาแล้ว ให้เอากรอบล่าสุดมาวาดโชว์บนกล้องไปพลางๆ
        if latest_boxes is not None:
            # 1. เราต้องสลับสีหลอกมันก่อน ส่งเป็น RGB เข้าไป
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. ฟังก์ชัน annotate จะไปแอบสลับสีกลับเป็น BGR ให้เราเอง (ออกมาสีถูกต้องพอดีเป๊ะ!)
            display_frame = annotate(image_source=frame_rgb, boxes=latest_boxes, logits=latest_logits, phrases=latest_phrases)

    cv2.imshow('Live Object Detection', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        user_prompt = input("\n🔍 What do you want to find? (Type here and press Enter): ")
        if user_prompt.strip():
            TEXT_PROMPT = user_prompt.lower().strip()
            if not TEXT_PROMPT.endswith("."):
                TEXT_PROMPT += " ."
            print(f"✅ Now finding: '{TEXT_PROMPT}'")
            # รีเซ็ตกรอบเก่าทิ้งเวลาเปลี่ยนคำค้นหา
            latest_boxes = None 
        else:
            TEXT_PROMPT = ""
            latest_boxes = None
            print("❌ Search cleared. Back to normal camera.")

cap.release()
cv2.destroyAllWindows()
if os.path.exists(TEMP_IMAGE_PATH):
    os.remove(TEMP_IMAGE_PATH)