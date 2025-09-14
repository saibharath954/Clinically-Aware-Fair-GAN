import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- ⚙️ Configuration ---
CSV_PATH = 'data/splits/master_subset_2k.csv'
IMAGE_DIR = 'data/mimic-cxr-jpg-2.0.0/files/' # This path is now set for your JPG structure
MASK_DIR = 'data/masks/'
IMG_SIZE = 256

def preprocess_jpg(image_path, transform):
    """
    Loads a JPG image, converts to 3-channel RGB, and applies robust 
    transformations suitable for an ImageNet-pre-trained model.
    """
    # Load image and ensure it's in RGB format
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Apply Albumentations (Resize, Normalize for ImageNet, ToTensor)
    if transform:
        augmented = transform(image=image_np)
        image_tensor = augmented['image']
    
    # Add the batch dimension (B, C, H, W)
    return image_tensor.unsqueeze(0)

def main():
    """
    Main function to generate and save segmentation masks from JPG files.
    """
    # --- 🧠 Model Loading (Updated for 3-channel input) ---
    print("Loading pre-trained U-Net model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The U-Net's encoder (resnet34) was pre-trained on 3-channel ImageNet images.
    # We MUST use in_channels=3 for it to work correctly.
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3, # <-- Key Change: Model expects a 3-channel input
        classes=1,
    )
    model.to(device)
    model.eval()

    # --- 🖼️ Data Transformation Pipeline ---
    # This pipeline correctly resizes, normalizes using ImageNet stats, and converts to a tensor.
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # --- 💾 Data Handling ---
    print(f"Loading image list from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    os.makedirs(MASK_DIR, exist_ok=True)
    print(f"Masks will be saved to {MASK_DIR}")

    # --- 🔄 Main Loop ---
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Masks"):
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = row['dicom_id'] # This is used as the unique filename
        
        # Construct the path to the JPG file
        jpg_path = os.path.join(
            IMAGE_DIR, f'p{subject_id[:2]}', f'p{subject_id}', f's{study_id}', f'{dicom_id}.jpg'
        )
        
        if not os.path.exists(jpg_path):
            continue

        try:
            # 1. Preprocess the JPG using the robust pipeline
            image_tensor = preprocess_jpg(jpg_path, transform).to(device)

            # 2. Perform inference
            with torch.no_grad():
                predicted_mask = model(image_tensor)

            # 3. Post-process the output mask
            probabilities = torch.sigmoid(predicted_mask).squeeze()
            binary_mask = (probabilities > 0.5).cpu().numpy()

            # 4. Convert to an image and save as PNG
            mask_to_save = (binary_mask * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_to_save)
            
            output_path = os.path.join(MASK_DIR, f"{dicom_id}.png")
            mask_image.save(output_path)

        except Exception as e:
            print(f"❌ Error processing {jpg_path}: {e}")

    print("\n✅ Mask generation complete!")

if __name__ == "__main__":
    main()