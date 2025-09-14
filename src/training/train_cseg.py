import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp

# The dataset classes are already defined in Cell 4

# --- ⚙️ Setup ---
# The CONFIG dictionary is already loaded from the previous cell
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
print(f"Using device: {DEVICE}")

# --- 🏋️‍♀️ Loss, Metrics, and Training Functions ---

# The combined loss is robust for segmentation tasks
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = nn.BCELoss()(inputs, targets)
        
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

def dice_score(preds, targets, smooth=1e-6):
    """Calculates Dice score for a batch."""
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float() # Binarize
    
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_dice_score = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            total_dice_score += dice_score(outputs, masks).item()
    
    val_loss = running_loss / len(dataloader.dataset)
    val_dice = total_dice_score / len(dataloader) # Avg dice score per batch
    return val_loss, val_dice

# --- 🚀 Main Execution ---
def run_training():
    # --- Data Loading & Augmentation ---
    # Augmentations for segmentation must be geometric/color-based, not ones that remove pixels
    transform = A.Compose([
        A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.Normalize(mean=(0.5,), std=(0.5,)), # Simple normalization for grayscale
        ToTensorV2(),
    ])
    
    train_df = pd.read_csv(CONFIG['TRAIN_CSV_PATH'])
    val_df = pd.read_csv(CONFIG['VAL_CSV_PATH'])
    
    train_dataset = MIMICXRSegmentationDataset(train_df, CONFIG['IMAGE_DIR'], CONFIG['MASK_DIR'], transform=transform)
    val_dataset = MIMICXRSegmentationDataset(val_df, CONFIG['IMAGE_DIR'], CONFIG['MASK_DIR'], transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'])

    # --- Model Setup ---
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
    model.to(DEVICE)
    
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    
    best_val_dice = 0.0
    
    # --- Training Loop ---
    for epoch in range(CONFIG['EPOCHS']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['EPOCHS']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice Score: {val_dice:.4f}")
        
        # We save the model with the highest Dice Score
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            model_path = os.path.join(CONFIG['OUTPUT_DIR'], CONFIG['MODEL_NAME'])
            torch.save(model.state_dict(), model_path)
            print(f"✨ New best model saved to {model_path} (Dice: {val_dice:.4f})")
            
    print("\n✅ Training of Cseg complete!")

# Run the training process
run_training()