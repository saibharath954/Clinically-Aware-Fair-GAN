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
import segmentation_models_pytorch as smp
import hashlib
import shutil

from data.dataset import MIMICCXR_GANDataset
from training.train_gan import CONFIG
from models.generator import Generator
from models.discriminator import Discriminator

# --- Utility Functions ---
def weights_init(m):
    """Custom weights initialization called on netG and netD."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def gradient_penalty(critic, real, fake, device):
    """Calculates the gradient penalty for WGAN-GP."""
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    # Create interpolated images WITH gradient tracking
    interpolated_images = real * alpha + fake * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Compute gradients
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

def is_valid_checkpoint(file_path):
    """Check if a checkpoint file is valid."""
    if not os.path.exists(file_path):
        return False

    try:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        required_keys = ['epoch', 'netG_state_dict', 'netD_state_dict',
                        'optimizerG_state_dict', 'optimizerD_state_dict',
                        'G_losses', 'D_losses']

        if not all(key in checkpoint for key in required_keys):
            return False

        return True
    except Exception as e:
        return False

def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, G_losses, D_losses, path):
    """Save training checkpoint with integrity verification."""
    try:
        temp_path = path + ".temp"

        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses
        }, temp_path)

        if is_valid_checkpoint(temp_path):
            if os.path.exists(path):
                os.remove(path)
            shutil.move(temp_path, path)
            print(f"✅ Checkpoint saved successfully for epoch {epoch}")
            return True
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def load_checkpoint(path, netG, netD, optimizerG, optimizerD, device):
    """Load training checkpoint with error handling."""
    if not os.path.exists(path):
        return 0, [], []

    if not is_valid_checkpoint(path):
        corrupted_path = path + ".corrupted"
        if os.path.exists(path):
            shutil.move(path, corrupted_path)
        return 0, [], []

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        print(f"✅ Checkpoint loaded successfully from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['G_losses'], checkpoint['D_losses']

    except Exception as e:
        corrupted_path = path + ".corrupted"
        if os.path.exists(path):
            shutil.move(path, corrupted_path)
        return 0, [], []

# --- 🚀 Main Training Execution ---
def run_gan_training(CONFIG):
    DEVICE = CONFIG['DEVICE']
    os.makedirs(CONFIG['IMAGE_OUTPUT_DIR'], exist_ok=True)
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    transform = A.Compose([
        A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
    train_df = pd.read_csv(CONFIG['TRAIN_CSV_PATH'])
    train_dataset = MIMICCXR_GANDataset(train_df, CONFIG['IMAGE_DIR'], transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True,
                           num_workers=CONFIG['NUM_WORKERS'])

    # --- Initialize Models ---
    netG = Generator(CONFIG['LATENT_DIM'], CONFIG['CHANNELS']).to(DEVICE)
    netD = Discriminator(CONFIG['CHANNELS']).to(DEVICE)

    # --- Load Frozen Critics ---
    Cdiag = models.resnet50()
    Cdiag.fc = nn.Linear(Cdiag.fc.in_features, 1)
    Cdiag.load_state_dict(torch.load(CONFIG['CDIAG_PATH'], map_location=DEVICE))
    Cdiag.eval()
    for param in Cdiag.parameters():
        param.requires_grad = False
    Cdiag.to(DEVICE)

    Cseg = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
    Cseg.load_state_dict(torch.load(CONFIG['CSEG_PATH'], map_location=DEVICE))
    Cseg.eval()
    for param in Cseg.parameters():
        param.requires_grad = False
    Cseg.to(DEVICE)

    # --- Optimizers ---
    optimizerD = optim.Adam(netD.parameters(), lr=CONFIG['D_LR'], betas=(CONFIG['B1'], CONFIG['B2']))
    optimizerG = optim.Adam(netG.parameters(), lr=CONFIG['G_LR'], betas=(CONFIG['B1'], CONFIG['B2']))

    # --- Load from Epoch 60 ---
    checkpoint_path = f"{CONFIG['OUTPUT_DIR']}/training_checkpoint.pth"
    netG_weights_path = f"{CONFIG['OUTPUT_DIR']}/netG_epoch_60.pth"
    netD_weights_path = f"{CONFIG['OUTPUT_DIR']}/netD_epoch_60.pth"

    if os.path.exists(checkpoint_path) and is_valid_checkpoint(checkpoint_path):
        start_epoch, G_losses, D_losses = load_checkpoint(checkpoint_path, netG, netD, optimizerG, optimizerD, DEVICE)
        print(f"Resuming from epoch {start_epoch}")
    elif os.path.exists(netG_weights_path) and os.path.exists(netD_weights_path):
        print("📦 Loading weights from epoch 60")
        netG.load_state_dict(torch.load(netG_weights_path, map_location=DEVICE))
        netD.load_state_dict(torch.load(netD_weights_path, map_location=DEVICE))
        start_epoch = 60
        G_losses, D_losses = [], []
    else:
        print("🚀 Starting from scratch")
        netG.apply(weights_init)
        netD.apply(weights_init)
        start_epoch = 0
        G_losses, D_losses = [], []

    print(f"✅ Starting from epoch {start_epoch}")

    fixed_noise = torch.randn(64, CONFIG['LATENT_DIM'], 1, 1, device=DEVICE)

    # --- Training Variables ---
    best_wasserstein = float('inf')
    lr_decay_counter = 0

    # --- 🏟️ The Grand Training Loop ---
    print("Starting Training Loop...")

    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        try:
            epoch_g_losses = []
            epoch_d_losses = []
            epoch_gp_values = []
            epoch_wasserstein_values = []

            for i, (real_imgs, labels, races) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")):
                real_imgs, labels, races = real_imgs.to(DEVICE), labels.to(DEVICE), races.to(DEVICE)

                # ---------------------------
                # Train Discriminator (Critic)
                # ---------------------------
                netD.zero_grad()

                # Generate fake images
                noise = torch.randn(real_imgs.size(0), CONFIG['LATENT_DIM'], 1, 1, device=DEVICE)
                with torch.no_grad():
                    fake_imgs = netG(noise)

                # Real images forward pass
                critic_real = netD(real_imgs).reshape(-1)
                # Fake images forward pass
                critic_fake = netD(fake_imgs).reshape(-1)

                # Calculate Wasserstein distance
                wasserstein_dist = torch.mean(critic_real) - torch.mean(critic_fake)
                epoch_wasserstein_values.append(wasserstein_dist.item())

                # Gradient penalty
                gp = gradient_penalty(netD, real_imgs, fake_imgs, DEVICE)

                # Discriminator loss
                loss_critic = -wasserstein_dist + CONFIG['LAMBDA_GP'] * gp
                loss_critic.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizerD.step()

                # Update generator every N_CRITIC steps
                if i % CONFIG['N_CRITIC'] == 0:
                    # ---------------------
                    # Train Generator
                    # ---------------------
                    netG.zero_grad()

                    # Generate fresh fake images
                    noise = torch.randn(real_imgs.size(0), CONFIG['LATENT_DIM'], 1, 1, device=DEVICE)
                    fake_imgs = netG(noise)

                    # 1. Adversarial Loss
                    critic_output = netD(fake_imgs)
                    adversarial_loss = -torch.mean(critic_output)

                    # 2. Fairness Loss
                    with torch.no_grad():
                        diag_output = Cdiag(fake_imgs)
                        diag_preds = torch.sigmoid(diag_output).squeeze()

                    fairness_loss = 0.0
                    tpr_per_group = []
                    for j in range(races.shape[1]):
                        group_mask = races[:, j] == 1
                        if group_mask.sum() > 0:
                            tpr_num = torch.sum(diag_preds[group_mask] * labels[group_mask])
                            tpr_den = torch.sum(labels[group_mask]) + 1e-6
                            tpr_per_group.append(tpr_num / tpr_den)

                    if len(tpr_per_group) > 1:
                        tpr_tensor = torch.stack(tpr_per_group)
                        if not torch.isnan(tpr_tensor).any():
                            fairness_loss = torch.std(tpr_tensor)  # Use standard deviation for better fairness

                    # 3. Clinical Loss
                    with torch.no_grad():
                        seg_masks = torch.sigmoid(Cseg(fake_imgs))
                        clinical_loss = torch.mean((1 - seg_masks) ** 2)

                    # Total Generator Loss
                    loss_gen = (adversarial_loss +
                               CONFIG['LAMBDA_FAIR'] * fairness_loss +
                               CONFIG['LAMBDA_CLINIC'] * clinical_loss)

                    loss_gen.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
                    optimizerG.step()

                    # Store losses for logging
                    G_losses.append(loss_gen.item())
                    D_losses.append(loss_critic.item())
                    epoch_g_losses.append(loss_gen.item())
                    epoch_d_losses.append(loss_critic.item())
                    epoch_gp_values.append(gp.item())

                # --- Logging ---
                if i % CONFIG['LOG_FREQ'] == 0:
                    avg_g_loss = np.mean(epoch_g_losses[-10:]) if epoch_g_losses else 0
                    avg_d_loss = np.mean(epoch_d_losses[-10:]) if epoch_d_losses else 0
                    avg_gp = np.mean(epoch_gp_values[-10:]) if epoch_gp_values else 0
                    avg_wasserstein = np.mean(epoch_wasserstein_values[-10:]) if epoch_wasserstein_values else 0

                    print(f"[{epoch+1}/{CONFIG['EPOCHS']}][{i}/{len(dataloader)}] "
                          f"Loss_D: {avg_d_loss:.4f} "
                          f"Loss_G: {avg_g_loss:.4f} "
                          f"GP: {avg_gp:.4f} "
                          f"W_dist: {avg_wasserstein:.4f}")

                # --- Save checkpoint every 20 batches ---
                if i % 20 == 0:
                    save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, G_losses, D_losses, checkpoint_path)

            # --- Validation & Monitoring ---
            # Calculate epoch statistics
            epoch_avg_g_loss = np.mean(epoch_g_losses) if epoch_g_losses else 0
            epoch_avg_d_loss = np.mean(epoch_d_losses) if epoch_d_losses else 0
            epoch_avg_wasserstein = np.mean(epoch_wasserstein_values) if epoch_wasserstein_values else 0

            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Avg G Loss: {epoch_avg_g_loss:.4f}")
            print(f"   Avg D Loss: {epoch_avg_d_loss:.4f}")
            print(f"   Avg Wasserstein: {epoch_avg_wasserstein:.4f}")

            # Learning rate scheduling
            if epoch_avg_wasserstein < best_wasserstein:
                best_wasserstein = epoch_avg_wasserstein
                lr_decay_counter = 0
            else:
                lr_decay_counter += 1
                if lr_decay_counter >= 5:  # Patience of 5 epochs
                    for param_group in optimizerG.param_groups:
                        param_group['lr'] *= 0.8
                    for param_group in optimizerD.param_groups:
                        param_group['lr'] *= 0.8
                    lr_decay_counter = 0
                    print(f"📉 Learning rate reduced to: {optimizerG.param_groups[0]['lr']}")

            # --- Save Images ---
            if (epoch + 1) % CONFIG['SAVE_IMG_FREQ'] == 0:
                with torch.no_grad():
                    fake_grid = netG(fixed_noise).detach().cpu()
                vutils.save_image(fake_grid, f"{CONFIG['IMAGE_OUTPUT_DIR']}/epoch_{epoch+1}.png",
                                 normalize=True, nrow=8)
                print(f"🖼️  Saved generated images for epoch {epoch+1}")

            # --- Save model checkpoints ---
            if (epoch + 1) % CONFIG.get('SAVE_MODEL_FREQ', 10) == 0:
                torch.save(netG.state_dict(), f"{CONFIG['OUTPUT_DIR']}/netG_epoch_{epoch+1}.pth")
                torch.save(netD.state_dict(), f"{CONFIG['OUTPUT_DIR']}/netD_epoch_{epoch+1}.pth")
                print(f"💾 Saved model weights for epoch {epoch+1}")

            # --- Save checkpoint at end of epoch ---
            save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, G_losses, D_losses, checkpoint_path)
            print(f"💾 Epoch {epoch+1} completed. Checkpoint saved.")

        except Exception as e:
            print(f"❌ Error occurred during epoch {epoch+1}: {e}")
            emergency_path = f"{CONFIG['OUTPUT_DIR']}/emergency_checkpoint_epoch_{epoch}.pth"
            save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, G_losses, D_losses, emergency_path)
            print(f"💾 Emergency checkpoint saved: {emergency_path}")
            continue

    # Final cleanup and model saving
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    torch.save(netG.state_dict(), f"{CONFIG['OUTPUT_DIR']}/netG_final.pth")
    torch.save(netD.state_dict(), f"{CONFIG['OUTPUT_DIR']}/netD_final.pth")
    print("💾 Saved final models")
    print("✅ GAN Training Complete!")

    return netG, netD, G_losses, D_losses

# Run training
generator, discriminator, G_losses, D_losses = run_gan_training(CONFIG)