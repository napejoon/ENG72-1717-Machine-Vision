import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

IMAGE_SIZE = 128


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def detect_and_localize_anomaly(image_path, model_path, anomaly_threshold=0.0015, pixel_threshold=0.1):
    # โหลดโมเดล
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # เตรียมภาพ
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)

    # ให้โมเดลสร้างภาพใหม่
    with torch.no_grad():
        reconstructed = model(img_tensor)

    # 4. คำนวณ Error รวม (ตัดสิน OK / NG)
    mse_loss = nn.MSELoss()
    error = mse_loss(reconstructed, img_tensor).item()
    is_ng = error > anomaly_threshold
    status = "NG" if is_ng else "OK"

    # หาตำแหน่งที่เสีย
    # คำนวณความต่างสัมบูรณ์ระหว่างภาพจริงกับภาพสร้างใหม่
    diff_tensor = torch.abs(img_tensor - reconstructed)
    diff_map = diff_tensor.squeeze().numpy()

    # เตรียมภาพต้นฉบับสำหรับวาดกรอบ
    original_img_8u = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
    result_img = cv2.cvtColor(original_img_8u, cv2.COLOR_GRAY2RGB)

    # สร้าง Heatmap เพื่อแสดงจุดที่ต่างกัน
    # จุดที่ค่าต่างกันมากจะออกสีแดง/ส้ม
    diff_map_normalized = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    diff_map_8u = (diff_map_normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_map_8u, cv2.COLORMAP_JET)

    if is_ng:
        # ใช้ Threshold สร้าง Binary Mask ตัดเฉพาะจุดที่มีค่าความต่างเกิน pixel_threshold
        _, binary_mask = cv2.threshold(diff_map, pixel_threshold, 1.0, cv2.THRESH_BINARY)
        binary_mask_8u = (binary_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # กรอง Noiseทิ้ง (กำหนดพื้นที่ขั้นต่ำ เช่น > 5 พิกเซล)
            if cv2.contourArea(contour) > 5:
                x, y, w, h = cv2.boundingRect(contour)
                # วาดกรอบสีแดง
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # พล็อตผลลัพธ์แบบ 4 รูปติดกัน
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title(f"Original ({status})\nScore: {error:.4f}")
    plt.imshow(original_img_8u, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Reconstructed")
    plt.imshow(reconstructed.squeeze().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Difference Heatmap")
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Defect Location")
    plt.imshow(result_img)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_file = 'anomaly_model_bottle.pth'

    # ทดสอบภาพปกติ
    detect_and_localize_anomaly('D:/ENG72 1717_ws/bottle/datasets/MVTecAD/bottle/test/broken_large/010.png', model_file)

