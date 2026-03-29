import cv2
import numpy as np

def nothing(x):
    pass

# โหลดภาพ
img = cv2.imread('156.jpg')
if img is None:
    print("ไม่พบไฟล์รูปภาพ")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_smoothed = cv2.medianBlur(gray, 3)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_boosted = clahe.apply(gray_smoothed)

h, w = gray.shape

# หาจุดศูนย์กลางจาน
blur_dish = cv2.GaussianBlur(gray, (35, 35), 0)
dish_circles = cv2.HoughCircles(
    blur_dish, cv2.HOUGH_GRADIENT, dp=1, minDist=h//2,
    param1=50, param2=30, minRadius=int(w * 0.25), maxRadius=int(w * 0.45)
)

if dish_circles is not None:
    dish_circles = np.round(dish_circles[0, :]).astype("int")
    dish_x, dish_y, base_dish_r = dish_circles[0]
else:
    dish_x, dish_y, base_dish_r = w // 2, h // 2, int(w * 0.35)

cv2.namedWindow('Tuning Dashboard')
cv2.resizeWindow('Tuning Dashboard', 600, 300)

cv2.createTrackbar('1. Mask Shrink', 'Tuning Dashboard', 46, 120, nothing) 
cv2.createTrackbar('2. Threshold', 'Tuning Dashboard', 40, 255, nothing) 
cv2.createTrackbar('3. Peak Separate', 'Tuning Dashboard', 40, 100, nothing) 
cv2.createTrackbar('4. Min Area', 'Tuning Dashboard', 1, 50, nothing) 
# 🔴 เอา Kernel Size กลับมาให้แล้วครับ!
cv2.createTrackbar('5. Kernel Size', 'Tuning Dashboard', 145, 255, nothing) 

while True:
    output = img.copy()
    
    shrink_val = cv2.getTrackbarPos('1. Mask Shrink', 'Tuning Dashboard')
    thresh_val = cv2.getTrackbarPos('2. Threshold', 'Tuning Dashboard')
    peak_val = cv2.getTrackbarPos('3. Peak Separate', 'Tuning Dashboard') / 100.0
    min_area_val = cv2.getTrackbarPos('4. Min Area', 'Tuning Dashboard')
    
    k_size_raw = cv2.getTrackbarPos('5. Kernel Size', 'Tuning Dashboard')
    k_size = max(3, k_size_raw)
    if k_size % 2 == 0: k_size += 1

    mask = np.zeros_like(gray)
    current_r = max(10, base_dish_r - shrink_val)
    cv2.circle(mask, (dish_x, dish_y), current_r, 255, -1)
    # วาดกรอบบนรูป
    cv2.circle(output, (dish_x, dish_y), current_r, (255, 255, 0), 2)
    masked_gray = cv2.bitwise_and(gray_boosted, gray_boosted, mask=mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tophat = cv2.morphologyEx(masked_gray, cv2.MORPH_TOPHAT, kernel)
    
    # หน้าต่างใหม่: สร้างก้อนสีขาว
    _, binary = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Distance Transform ผ่าก้อน
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    _, peaks = cv2.threshold(dist_transform, peak_val * dist_transform.max(), 255, 0)
    peaks = np.uint8(peaks)
    
    contours, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bacteria_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area_val:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output, (cX, cY), 5, (0, 255, 0), 30)
                cv2.circle(output, (cX, cY), 1, (0, 0, 255), -30)
                bacteria_count += 1

    cv2.putText(output, f"Total: {bacteria_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    def resize_for_display(image, height=650):
        h, w = image.shape[:2]
        new_w = int(w * (height / h))
        return cv2.resize(image, (new_w, height))

    cv2.imshow("1. Binary View (Must see solid white clumps)", resize_for_display(binary)) 
    cv2.imshow("2. Peak View (Separated Dots)", resize_for_display(peaks)) 
    cv2.imshow("3. Final Result", resize_for_display(output))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()