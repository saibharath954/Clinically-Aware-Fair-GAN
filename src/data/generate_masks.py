# src/data/generate_masks.py

import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp
from skimage.transform import resize

# --- ⚙️ Configuration (Updated) ---
# Update these paths to match your project structure
CSV_PATH = 'data/splits/master_subset_2k.csv'
# This path now points to the root of your JPG file structure
IMAGE_DIR = 'data/mimic-cxr-jpg-2.0.0/files/' 
MASK_DIR = 'data/masks/'
IMG_SIZE = 256

def preprocess_image(jpg_path, img_size):
    """
    Loads a JPG image, converts to grayscale, normalizes it, resizes it, 
    and converts it to a PyTorch tensor suitable for the model.
    """
    # Load JPG file as grayscale
    image_pil = Image.open(jpg_path).convert('L')
    image = np.array(image_pil, dtype=np.float32)

    # Normalize to [0, 1]
    min_val, max_val = image.min(), image.max()
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    
    # Resize image
    image_resized = resize(image, (img_size, img_size), anti_aliasing=True)

    # Convert to tensor and add batch/channel dimensions (B, C, H, W)
    tensor = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0)
    return tensor

def main():
    """
    Main function to generate and save segmentation masks.
    """
    # --- 🧠 Model Loading ---
    print("Loading pre-trained U-Net model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
    model.to(device)
    model.eval()

    # --- 💾 Data Handling ---
    print(f"Loading image list from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    os.makedirs(MASK_DIR, exist_ok=True)
    print(f"Masks will be saved to {MASK_DIR}")

    # --- 🔄 Main Loop ---
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Masks"):
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = row['dicom_id']
        
        # Construct the full JPG path (Updated)
        # It now points to a .jpg file in your specified directory
        image_path = os.path.join(
            IMAGE_DIR, 
            f'p{subject_id[:2]}', 
            f'p{subject_id}', 
            f's{study_id}', 
            f'{dicom_id}.jpg'  # <-- The extension is now .jpg
        )
        
        if not os.path.exists(image_path):
            print(f"⚠️ Warning: Image not found, skipping: {image_path}")
            continue

        try:
            image_tensor = preprocess_image(image_path, IMG_SIZE).to(device)

            with torch.no_grad():
                predicted_mask = model(image_tensor)

            probabilities = torch.sigmoid(predicted_mask).squeeze()
            binary_mask = (probabilities > 0.5).cpu().numpy()

            mask_to_save = (binary_mask * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_to_save)
            
            output_path = os.path.join(MASK_DIR, f"{dicom_id}.png")
            mask_image.save(output_path)

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

    print("\n✅ Mask generation complete!")

if __name__ == "__main__":
    main()