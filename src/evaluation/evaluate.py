import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.utils as vutils
import pandas as pd
from tqdm import tqdm
import os
import yaml
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

# We need to import our project's models and datasets
from src.models.generator import Generator
from src.data.dataset import MIMICCXRClassifierDataset

def generate_synthetic_data(config, device):
    """Loads a trained generator and creates a new synthetic dataset."""
    print("---  bước 1: Generating Synthetic Dataset ---")
    os.makedirs(config['SYNTHETIC_DATA_DIR'], exist_ok=True)

    # Load Generator
    netG = Generator(config['LATENT_DIM'], config['CHANNELS']).to(device)
    netG.load_state_dict(torch.load(config['GENERATOR_CHECKPOINT'], map_location=device))
    netG.eval()

    # Create balanced labels for the synthetic data
    num_positive = config['NUM_SYNTHETIC_IMAGES'] // 2
    labels = np.array([1] * num_positive + [0] * (config['NUM_SYNTHETIC_IMAGES'] - num_positive))
    np.random.shuffle(labels)

    image_ids = []
    generated_labels = []
    
    with torch.no_grad():
        for i in tqdm(range(0, config['NUM_SYNTHETIC_IMAGES'], config['GENERATION_BATCH_SIZE']), desc="Generating Images"):
            batch_size = min(config['GENERATION_BATCH_SIZE'], config['NUM_SYNTHETIC_IMAGES'] - i)
            noise = torch.randn(batch_size, config['LATENT_DIM'], 1, 1, device=device)
            fake_imgs = netG(noise)

            for j in range(batch_size):
                img_idx = i + j
                image_id = f"synth_{img_idx:05d}.jpg"
                vutils.save_image(fake_imgs[j], os.path.join(config['SYNTHETIC_DATA_DIR'], image_id), normalize=True)
                image_ids.append(image_id)
                generated_labels.append(labels[img_idx])
    
    # Save the labels to a CSV file
    synthetic_df = pd.DataFrame({'image_id': image_ids, 'Pneumonia': generated_labels})
    synthetic_df.to_csv(config['SYNTHETIC_CSV_PATH'], index=False)
    print(f"✅ Generated {config['NUM_SYNTHETIC_IMAGES']} synthetic images and saved labels to {config['SYNTHETIC_CSV_PATH']}")
    return synthetic_df

def train_downstream_classifier(config, device):
    """Trains a new classifier from scratch on the synthetic dataset."""
    print("\n--- bước 2: Training Downstream Classifier on Synthetic Data ---")
    os.makedirs(config['CLASSIFIER_OUTPUT_DIR'], exist_ok=True)

    transform = A.Compose([
        A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # We need a small, modified Dataset class for the synthetic data
    class SyntheticDataset(Dataset):
        def __init__(self, df, image_dir, transform):
            self.df = df
            self.image_dir = image_dir
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_path = os.path.join(self.image_dir, row['image_id'])
            image = np.array(Image.open(image_path).convert("RGB"))
            label = torch.tensor(row['Pneumonia'], dtype=torch.float32)
            if self.transform:
                image = self.transform(image=image)['image']
            return image, label.unsqueeze(0)

    synth_df = pd.read_csv(config['SYNTHETIC_CSV_PATH'])
    train_dataset = SyntheticDataset(synth_df, config['SYNTHETIC_DATA_DIR'], transform)
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['NUM_WORKERS'])
    
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    for epoch in range(config['EPOCHS']):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['EPOCHS']}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), os.path.join(config['CLASSIFIER_OUTPUT_DIR'], config['CLASSIFIER_MODEL_NAME']))
    print("✅ Downstream classifier training complete.")
    return model

def evaluate_on_real_data(classifier, config, device):
    """Evaluates the synthetically-trained classifier on the real test set."""
    print("\n--- bước 3: Evaluating on Real Test Data ---")

    transform = A.Compose([
        A.Resize(config['IMG_SIZE'], config['IMG_SIZE']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    test_df = pd.read_csv(config['REAL_DATA_CSV_TEST'])
    # The original dataset needs a different class that knows the MIMIC path structure
    class RealTestDataset(MIMICCXRClassifierDataset):
        def __getitem__(self, idx):
            image, label = super().__getitem__(idx)
            race = self.df.iloc[idx]['race_group']
            return image, label, race

    test_dataset = RealTestDataset(test_df, config['IMAGE_DIR_REAL'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'])
    
    classifier.eval()
    all_preds, all_labels, all_races = [], [], []

    with torch.no_grad():
        for images, labels, races in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_races.extend(races)

    # --- Calculate Metrics ---
    df_results = pd.DataFrame({'label': all_labels, 'pred_prob': all_preds, 'race': all_races})
    df_results['prediction'] = (df_results['pred_prob'] > 0.5).astype(int)

    # Utility Metrics
    auc = roc_auc_score(df_results['label'], df_results['pred_prob'])
    accuracy = accuracy_score(df_results['label'], df_results['prediction'])
    f1 = f1_score(df_results['label'], df_results['prediction'])
    
    # Fairness Metrics (Equal Opportunity Difference)
    tpr_per_group = {}
    for group in df_results['race'].unique():
        group_df = df_results[df_results['race'] == group]
        tn, fp, fn, tp = confusion_matrix(group_df['label'], group_df['prediction'], labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tpr_per_group[group] = tpr
    
    eod = max(tpr_per_group.values()) - min(tpr_per_group.values())

    # --- Print Report ---
    print("\n--- 📊 Evaluation Report ---")
    print(f"Trained on {config['NUM_SYNTHETIC_IMAGES']} synthetic images. Evaluated on {len(df_results)} real test images.")
    print("\n## 🎯 Overall Performance (Utility)")
    print(f"**AUC:** {auc:.4f}")
    print(f"**Accuracy:** {accuracy:.4f}")
    print(f"**F1-Score:** {f1:.4f}")

    print("\n## ⚖️ Fairness Performance")
    print("True Positive Rate (TPR) by Group:")
    for group, tpr in tpr_per_group.items():
        print(f"  - {group}: {tpr:.4f}")
    print(f"**Equal Opportunity Difference (Max TPR - Min TPR): {eod:.4f}**")
    print("\n--- Evaluation Complete ---")


def main():
    """Main function to run the entire evaluation pipeline."""
    with open('configs/evaluate.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['DEVICE'] if torch.cuda.is_available() else 'cpu'

    # STEP 1
    generate_synthetic_data(config, device)
    
    # STEP 2
    trained_classifier = train_downstream_classifier(config, device)
    
    # STEP 3
    evaluate_on_real_data(trained_classifier, config, device)

if __name__ == '__main__':
    main()