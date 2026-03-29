import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# กำหนด Hyperparameters และการแปลงภาพ
BATCH_SIZE = 16
EPOCHS = 500
LEARNING_RATE = 1e-3
IMAGE_SIZE = 128
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# สร้างโครงสร้างโมเดล Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model():
    # โหลดข้อมูล (ดึงเฉพาะโฟลเดอร์ train/good)
    train_dataset = datasets.ImageFolder(root='ENG35-1717-Machine-Vision/week2_bottle/datasets/datasets/MVTecAD/tile/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("เริ่มการเทรนโมเดล...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for data in train_loader:
            img, _ = data

            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.6f}")

    # บันทึกโมเดล
    torch.save(model.state_dict(), 'anomaly_model_tile.pth')
    print("เทรนเสร็จสิ้นและบันทึกโมเดลเรียบร้อยแล้ว!")


if __name__ == "__main__":
    train_model()