import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import copy

# Custom model that combines image and metadata
class CheXpertModel(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXpertModel, self).__init__()
        # Load pretrained MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Get the number of features from MobileNetV2
        num_features = self.mobilenet.last_channel
        
        # Create new classifier that combines image features with metadata
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 4, 512),  # +4 for metadata features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Remove the original classifier
        self.mobilenet.classifier = nn.Identity()

    def forward(self, x, metadata):
        # Get image features
        features = self.mobilenet.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Combine with metadata
        combined = torch.cat([features, metadata], dim=1)
        
        # Pass through classifier
        return self.classifier(combined)

# -----------------------------
# Dataset Class
# -----------------------------
class CheXpertDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.image_paths = dataframe['Path'].values
        
        # Extract metadata
        self.ages = dataframe['Age'].values
        self.sexes = dataframe['Sex'].values
        self.views = dataframe['Frontal/Lateral'].values
        self.techniques = dataframe['AP/PA'].values
        
        # Fill NaNs for disease labels and ensure labels are in range [0, 1]
        self.dataframe.iloc[:, 5:] = self.dataframe.iloc[:, 5:].fillna(0).clip(0, 1)
        self.labels = self.dataframe.iloc[:, 5:].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224))  # fallback blank image

        if self.transform:
            image = self.transform(image)

        # Prepare metadata
        age = torch.tensor([float(self.ages[idx]) / 100.0])  # Normalize age
        sex = torch.tensor([1.0 if self.sexes[idx].lower() == 'male' else 0.0])
        view = torch.tensor([1.0 if self.views[idx] == 'Lateral' else 0.0])
        technique = torch.tensor([1.0 if self.techniques[idx] == 'AP' else 0.0])
        
        metadata = torch.cat([age, sex, view, technique])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, metadata, label

# -----------------------------
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduced size for faster training
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------
# Take only 10% of the dataframe and split into train/val
# -----------------------------
small_df = test_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
train_df = small_df.sample(frac=0.8, random_state=42)
val_df = small_df.drop(train_df.index)

train_dataset = CheXpertDataset(train_df, transform=train_transform)
val_dataset = CheXpertDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# -----------------------------
# Load Pretrained MobileNetV2
# -----------------------------
mobilenet = models.mobilenet_v2(pretrained=True)

# Replace the classifier to match our number of classes (14)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 14)

# -----------------------------
# Set Device, Loss, Optimizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(mobilenet.parameters(), lr=1e-4)

# -----------------------------
# Early Stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=3, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        model.load_state_dict(self.best_model)

# -----------------------------
# Training Function
# -----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=4, patience=3):
    early_stopping = EarlyStopping(patience=patience, delta=0.01)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, metadata, labels) in enumerate(train_loader):
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)

            if torch.isnan(loss) or loss.item() < 0:
                print(f"Invalid loss detected: {loss.item()}. Stopping training.")
                return

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Validation Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    early_stopping.load_best_model(model)

# -----------------------------
# Train & Save
# -----------------------------
model = CheXpertModel(num_classes=14)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=4, patience=3)
torch.save(model.state_dict(), 'mobilenet_chexpert_best.pth')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, roc_auc_score
from PIL import Image

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilenet = models.mobilenet_v2(pretrained=False)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 14)  # 14 CheXpert classes
mobilenet.load_state_dict(torch.load(r"C:\Users\abdur\Documents\Data_Mining\mobilenet_chexpert_best.pth", map_location=device))
mobilenet = mobilenet.to(device)

# -----------------------------
# Prepare Test DataLoader
# -----------------------------
  # Replace with actual test CSV
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = CheXpertDataset(trained_df, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# -----------------------------
# Evaluate Model
# -----------------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    return all_targets, all_outputs

all_targets, all_outputs = evaluate_model(mobilenet, test_loader, device)

# -----------------------------
# Classification Report
# -----------------------------
threshold = 0.5
binary_predictions = (torch.sigmoid(torch.tensor(all_outputs)) > threshold).numpy()
report = classification_report(all_targets, binary_predictions, zero_division=0, target_names=trained_df.columns[5:])
print("Classification Report:")
print(report)

# -----------------------------
# AUC Scores
# -----------------------------
auc_scores = {}
for i, label in enumerate(trained_df.columns[5:]):
    try:
        auc_scores[label] = roc_auc_score(all_targets[:, i], all_outputs[:, i])
    except ValueError:
        auc_scores[label] = float('nan')

try:
    micro_auc = roc_auc_score(all_targets.ravel(), all_outputs.ravel(), average="micro")
except ValueError:
    micro_auc = float('nan')

print("\nAUC Scores:")
for label, auc in auc_scores.items():
    print(f"{label:25}: {auc:.3f}")
print(f"\nMicro-average AUC: {micro_auc:.3f}")

# -----------------------------
# Visualization
# -----------------------------
def visualize_predictions(test_df, y_pred, y_true, num_samples=3, random_seed=42):
    pathologies = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax',
        'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]
    np.random.seed(random_seed)
    y_pred_probs = torch.sigmoid(torch.tensor(y_pred)).numpy()
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    for i in range(num_samples):
        idx = np.random.randint(len(test_df))
        sample = test_df.iloc[idx]
        img_path = sample['Path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not load image at {img_path}")
            continue
        image = cv2.resize(image, (256, 256))

        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"X-ray: {img_path.split('/')[-1]}", fontsize=10)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.axis('off')

        pred_values = [str(y_pred_binary[idx][i]) for i in range(len(pathologies))]
        true_values = [str(int(y_true[idx][i])) for i in range(len(pathologies))]

        cell_colors = []
        for p, t in zip(y_pred_binary[idx], y_true[idx]):
            if p == t:
                cell_colors.append(['lightgreen']*3)
            else:
                cell_colors.append(['lightcoral']*3)

        table = plt.table(
            cellText=np.array([pathologies, pred_values, true_values]).T,
            colLabels=['Pathology', 'Predicted', 'True'],
            cellColours=cell_colors,
            loc='center',
            colWidths=[0.4, 0.3, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title("Model Predictions vs Ground Truth", pad=20)
        plt.tight_layout()
        plt.show()

# Visualize predictions
visualize_predictions(trained_df, all_outputs, all_targets, num_samples=5)