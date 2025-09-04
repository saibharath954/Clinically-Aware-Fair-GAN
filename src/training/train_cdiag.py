import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd
from tqdm import tqdm
import os
import yaml  # <-- Import YAML

# Import our custom dataset class
from src.data.dataset import MIMICCXRClassifierDataset # <-- Updated class name
import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- ⚙️ Configuration Loading ---
# Load configuration from the YAML file
with open('configs/train_cdiag.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Determine device, falling back to CPU if CUDA is not available or specified
# if CONFIG['DEVICE'] == 'cuda' and not torch.cuda.is_available():
#     print("CUDA not available, falling back to CPU.")
#     DEVICE = 'cpu'
# else:
#     DEVICE = CONFIG['DEVICE']
def get_device(config_device: str = "cpu"):
    import torch
    if config_device == "cuda":
        try:
            if torch.cuda.is_available():
                _ = torch.cuda.get_device_name(0)
                return "cuda"
            else:
                print("⚠️ CUDA not available. Falling back to CPU.")
        except Exception as e:
            print(f"⚠️ CUDA init failed: {e}. Falling back to CPU.")
    return "cpu"
DEVICE = get_device(CONFIG.get('DEVICE', 'cpu'))
print(f"Using device: {DEVICE}")

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Function to train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def main():
    """Main training and saving function."""
    print(f"Using device: {CONFIG['DEVICE']}")
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

    # --- 🧠 Data Loading & Augmentation ---
    # Define transformations for the training data
    # ResNet was trained on ImageNet, so we use its normalization stats
    train_transform = A.Compose([
        A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    df = pd.read_csv(CONFIG['CSV_PATH'])
    # For this development phase, we'll use the whole dataset as 'train'
    # In a full project, you'd split this into train/val sets
    train_dataset = MIMICCXRClassifierDataset(df, CONFIG['IMAGE_DIR'], transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=4 # Adjust based on your system
    )

    # --- 🚀 Model Setup ---
    print("Loading pre-trained ResNet-50 model...")
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Replace the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Output is 1 for binary classification (logit)
    model.to(CONFIG['DEVICE'])

    # --- 🏋️ Training ---
    criterion = nn.BCEWithLogitsLoss() # Stable for binary classification
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])

    best_loss = float('inf')

    for epoch in range(CONFIG['EPOCHS']):
        print(f"--- Epoch {epoch+1}/{CONFIG['EPOCHS']} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['DEVICE'])
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Save the model if it has the best loss so far
        # In a real project, you'd evaluate on a separate validation set
        if train_loss < best_loss:
            best_loss = train_loss
            model_path = os.path.join(CONFIG['OUTPUT_DIR'], CONFIG['MODEL_NAME'])
            torch.save(model.state_dict(), model_path)
            print(f"✨ Model saved to {model_path}")
            
    print("\n✅ Training of Cdiag complete!")

if __name__ == "__main__":
    main()