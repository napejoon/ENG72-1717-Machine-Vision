import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_SIZE = 128


# ===============================
# Model
# ===============================
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


# ===============================
# Predict image
# ===============================
def predict_image(img_path, model, transform, threshold):

    img = Image.open(img_path)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        reconstructed = model(img_tensor)

    mse = nn.MSELoss()
    error = mse(reconstructed, img_tensor).item()

    pred = 1 if error > threshold else 0

    return pred, error


# ===============================
# Confusion Matrix Plot
# ===============================
def plot_confusion_matrix(TP, TN, FP, FN):

    matrix = [[TN, FP],
              [FN, TP]]

    plt.figure(figsize=(5,5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred OK","Pred NG"],
        yticklabels=["Actual OK","Actual NG"]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")

    plt.show()


# ===============================
# Error Distribution Plot
# ===============================
def plot_error_distribution(good_errors, defect_errors):

    plt.figure(figsize=(8,5))

    plt.hist(good_errors,
             bins=20,
             alpha=0.6,
             label="Good",
             color="blue")

    plt.hist(defect_errors,
             bins=20,
             alpha=0.6,
             label="Defect",
             color="red")

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")

    plt.title("Error Distribution (Good vs Defect)")

    plt.legend()

    plt.show()


# ===============================
# Evaluate Dataset
# ===============================
def evaluate_dataset(dataset_path, model_path, threshold=0.0002):

    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    good_errors = []
    defect_errors = []

    for folder in os.listdir(dataset_path):

        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        print("Testing folder:", folder)

        # ground truth
        if folder == "good":
            gt = 0
        else:
            gt = 1

        for img_name in os.listdir(folder_path):

            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(folder_path, img_name)

            pred, error = predict_image(img_path, model, transform, threshold)

            # save error distribution
            if gt == 0:
                good_errors.append(error)
            else:
                defect_errors.append(error)

            # confusion matrix
            if gt == 1 and pred == 1:
                TP += 1

            elif gt == 0 and pred == 0:
                TN += 1

            elif gt == 0 and pred == 1:
                FP += 1

            elif gt == 1 and pred == 0:
                FN += 1


    # ===============================
    # Metrics
    # ===============================

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    print("\n===== RESULT =====")

    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)

    print("\nAccuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)


    # ===============================
    # Plot graphs
    # ===============================

    plot_confusion_matrix(TP, TN, FP, FN)

    plot_error_distribution(good_errors, defect_errors)


# ===============================
# Main
# ===============================
if __name__ == "__main__":

    model_file = "anomaly_model_wood.pth"

    dataset_path = "C:/Users/natin/Documents/vision/Train/datasets/MVTecAD/wood/test"

    evaluate_dataset(dataset_path, model_file)