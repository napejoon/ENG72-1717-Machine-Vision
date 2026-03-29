import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

IMAGE_SIZE = 128

# โครงสร้างโมเดล Autoencoder (เหมือนเดิม)
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

def evaluate_multiple_thresholds(test_dir, model_path, thresholds_list):
    # 1. โหลดโมเดล
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    mse_loss = nn.MSELoss()
    
    y_true = []
    errors = [] 

    print("กำลังสแกนรูปภาพและคำนวณ Error (ขั้นตอนนี้ทำแค่รอบเดียว)...")
    
    # 2. อ่านภาพและเก็บค่า Error ทั้งหมด
    for folder_name in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder_name)
        if not os.path.isdir(folder_path): continue
            
        true_label = 0 if folder_name == 'good' else 1
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                reconstructed = model(img_tensor)
                
            error = mse_loss(reconstructed, img_tensor).item()
            
            y_true.append(true_label)
            errors.append(error)

    y_true = np.array(y_true)
    errors = np.array(errors)
    print("สแกนเสร็จสิ้น! เริ่มทำการเปรียบเทียบ Thresholds\n")

    # 3. สร้างตารางสรุปผลแบบ Text
    print(f"{'Threshold':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)

    # 4. เตรียมพล็อต Confusion Matrix หลายๆ รูปในหน้าต่างเดียว
    # สร้าง Grid กราฟ ให้พอดีกับจำนวน Threshold (เช่น 6 ค่า จะได้ 2 แถว 3 คอลัมน์)
    num_thresholds = len(thresholds_list)
    cols = 3
    rows = (num_thresholds + cols - 1) // cols 
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() # ทำให้เรียกใช้แกนง่ายขึ้น

    # 5. วนลูปทดสอบแต่ละ Threshold
    for i, t in enumerate(thresholds_list):
        # ถ่ายทอดตรรกะ: ถ้า Error > t ให้เป็น 1 (NG), ถ้าไม่ใช่เป็น 0 (OK)
        y_pred = (errors > t).astype(int)

        # คำนวณ Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        # ปริ้นท์ข้อมูลลงตาราง Text
        print(f"{t:<12.6f} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f}")

        # พล็อต Confusion Matrix
        if i < len(axes):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['OK', 'NG'], yticklabels=['OK', 'NG'])
            axes[i].set_title(f'Threshold: {t}')
            axes[i].set_ylabel('Actual')
            axes[i].set_xlabel('Predicted')

    # ลบกรอบกราฟที่ว่างเปล่าทิ้ง (กรณีจำนวน Threshold ไม่เต็ม Grid)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_file = 'ENG35-1717-Machine-Vision/week2_bottle/anomaly_model_tile.pth'
    test_directory = 'ENG35-1717-Machine-Vision/week2_bottle/datasets/MVTecAD/tile/test'
    
    # กำหนดลิสต์ของค่า Threshold ที่คุณต้องการทดสอบ (ใส่ 5-6 ค่าได้เลย)
    # หมายเหตุ: ลองปรับตัวเลขตรงนี้ให้สอดคล้องกับค่า Error ของโมเดลคุณ
    my_thresholds = [0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025]
    
    evaluate_multiple_thresholds(test_directory, model_file, my_thresholds)