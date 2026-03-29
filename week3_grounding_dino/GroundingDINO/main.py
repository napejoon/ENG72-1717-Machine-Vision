import os
import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate

# CPU process
DEVICE = "cpu"
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

print("="*50)
print("Real-time Object Detection with Grounding DINO")
print("="*50)

# 1. ถามคำค้นหาก่อนเริ่ม
user_prompt = input("What are u finding: ")
if not user_prompt.strip():
    print("You did not answer. Program is shutting down.")
    exit()

TEXT_PROMPT = user_prompt.lower().strip()
if not TEXT_PROMPT.endswith("."):
    TEXT_PROMPT += " ."

# 2. โหลดโมเดล
print("\nLoading model into CPU (Please wait)... ")
model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# 3. เปิดกล้อง
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

TEMP_IMAGE_PATH = "temp_capture.jpg"
print(f"\nStart Real-time finding: '{TEXT_PROMPT}'. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 4. เซฟเฟรมปัจจุบันเป็นไฟล์ชั่วคราวเพื่อให้ load_image อ่านได้
    cv2.imwrite(TEMP_IMAGE_PATH, frame)
    
    # 5. โหลดภาพเข้า Grounding DINO
    image_source, image = load_image(TEMP_IMAGE_PATH)

    # 6. ให้โมเดลทำนาย (ตรงนี้จะกินเวลาบน CPU ทำให้ภาพดูกระตุก)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    # 7. วาดกรอบและข้อความลงบนภาพ
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # 8. **สำคัญมาก:** แปลงสีจาก RGB (ของ DINO) กลับเป็น BGR (ของ OpenCV)
    # ถ้าไม่แปลง สีในกล้องจะเพี้ยน (เช่น สีแดงกลายเป็นสีน้ำเงิน)
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # 9. แสดงผลภาพที่ตีกรอบแล้วแบบ Real-time
    cv2.imshow('Real-time Grounding DINO', annotated_frame_bgr)

    # กด 'q' เพื่อหยุดการทำงาน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# คืนค่ากล้องและลบไฟล์ชั่วคราวทิ้งเมื่อปิดโปรแกรม
cap.release()
cv2.destroyAllWindows()
if os.path.exists(TEMP_IMAGE_PATH):
    os.remove(TEMP_IMAGE_PATH)