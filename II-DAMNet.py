import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from models.pspnet import PSPNet
# Import de l'architecture PSPNet originale
from models.pspnet import PSPNet

# ==============================================================================
# CONFIGURATION DES HYPERPARAMÈTRES ET DOSSIERS
# ==============================================================================
DATASET_ROOT = "dataset_hybride_strict"

TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_GT_DIR    = os.path.join(DATASET_ROOT, "train", "labels")

VAL_IMAGE_DIR   = os.path.join(DATASET_ROOT, "val", "images")
VAL_GT_DIR      = os.path.join(DATASET_ROOT, "val", "labels")

# Paramètres d'entraînement
BATCH_SIZE = 12       
EPOCHS = 100
LEARNING_RATE = 0.001 
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# --- CONFIGURATION 4 CLASSES ---
NUM_CLASSES = 4
IGNORE_INDEX = 255   

SAVE_DIR = "checkpoints/II-DAMNet"
os.makedirs(SAVE_DIR, exist_ok=True)
# ==============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 1. FONCTION DE MAPPING PYTORCH CORRIGÉE (11 Classes -> 4 Classes)
# ==============================================================================

# ==============================================================================
# LE MAPPING CORRIGÉ BASÉ SUR LES INDICES REELS DE RESCUENET
# ==============================================================================
def map_mask_to_4_classes(mask_tensor):
    # 1. Force the entire canvas to default to Target Class 3 (Fond/Background).
    # This completely zeroes out ID:1 (Background) and ID:2 (Water) instantly!
    # It also sweeps vehicles (ID:7) and regular trees into the background.
    mapped = torch.full_like(mask_tensor, 3)
    
    # 2. Map your true targets explicitly using the shifted IDs from your image:
    
    # Target 0: Bâtiment Intact -> Shifted Class 3
    mapped[mask_tensor == 3] = 0
    
    # Target 1: Bâtiment Endommagé -> Shifted Classes 4, 5, and 6
    mapped[(mask_tensor == 4) | (mask_tensor == 5) | (mask_tensor == 6)] = 1
    
    # Target 2: Route Bloquée -> Shifted Class 9 (Assuming Road-Blocked shifted from 8 to 9)
    # We also include 8 just in case your dataset combines clear/blocked roads
    mapped[(mask_tensor == 8) | (mask_tensor == 9)] = 2
    
    # 3. Preserve the training ignore index
    mapped[mask_tensor == 255] = IGNORE_INDEX
    
    return mapped

# ==============================================================================
# 2. CLASSE DATASET
# ==============================================================================
class HybrideDataset4Classes(Dataset):
    def __init__(self, image_dir, gt_dir, img_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        gt_path = os.path.join(self.gt_dir, f"{base_name}_lab.png") 
        
        img_pil = Image.open(img_path).convert('RGB')
        gt_pil = Image.open(gt_path).convert('L')
        
        if self.img_transform:
            img_tensor = self.img_transform(img_pil)
        if self.mask_transform:
            gt_tensor = self.mask_transform(gt_pil)
            gt_tensor = torch.squeeze(gt_tensor).long() 
            gt_tensor = map_mask_to_4_classes(gt_tensor)

        return img_tensor, gt_tensor

# ==============================================================================
# 3. FONCTIONS UTILITAIRES ET DE VÉRIFICATION VISUELLE
# ==============================================================================
def calculate_miou(hist):
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou = np.zeros(NUM_CLASSES)
    valid_mask = union > 0
    iou[valid_mask] = (intersection[valid_mask] / union[valid_mask]) * 100
    miou = np.nanmean(iou) 
    return miou

def visually_verify_dataset(dataloader, num_samples=2):
    """
    Récupère un batch du dataloader et affiche les images RGB et les masques
    remappés pour une inspection visuelle rigoureuse avant entraînement.
    """
    print("\n[Vérification] Chargement d'un batch de validation visuelle...")
    images, masks = next(iter(dataloader))
    
    images = images.numpy()
    masks = masks.numpy()
    
    class_names = {0: '0: Intact', 1: '1: Damaged', 2: '2: Blocked', 3: '3: Background'}
    # Couleurs discrètes : Intact=Vert, Endommagé=Rouge, Route Bloquée=Jaune, Fond=Bleu/Gris
    custom_cmap = ListedColormap(['#2ca02c', '#d62728', '#bcbd22', '#1f77b4'])
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        mask = masks[i]
        
        # Dé-normalisation de l'image pour affichage standard
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_unnorm = img * std + mean
        img_unnorm = np.clip(img_unnorm, 0, 1)
        img_unnorm = np.transpose(img_unnorm, (1, 2, 0))
        
        print(f"Échantillon {i+1} -> Classes cibles trouvées : {np.unique(mask)}")
        for c in range(4):
            pixel_count = np.sum(mask == c)
            print(f"   Class {c} ({class_names[c]}) : {pixel_count:,} pixels")
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_unnorm)
        axes[0].set_title(f"Image UAV Échantillon {i+1}")
        axes[0].axis('off')
        
        im = axes[1].imshow(mask, cmap=custom_cmap, vmin=0, vmax=3)
        axes[1].set_title(f"Masque Remappé 4 Classes (Cible)")
        axes[1].axis('off')
        
        cbar = fig.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels([class_names[0], class_names[1], class_names[2], class_names[3]])
        
        plt.tight_layout()
        plt.show()
    print("[Vérification] Fin de l'inspection visuelle. Début du cycle d'entraînement.\n")

def plot_learning_curves(train_losses, val_losses, train_mious, val_mious, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r--', label='Val Loss', linewidth=2)
    plt.title('Loss d\'Entraînement et Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mious, 'b-', label='Train mIoU', linewidth=2)
    plt.plot(epochs, val_mious, 'r--', label='Val mIoU', linewidth=2)
    plt.title('mIoU d\'Entraînement et Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Mean IoU (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
def save_val_overlay_snapshot(model, dataloader, epoch, save_dir="checkpoints/val_overlays"):
    """
    Grabs a validation sample, overlays the model's current predictions 
    with transparency, and saves it to monitor learning progress.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Grab the first batch from validation
    images, masks = next(iter(dataloader))
    inputs, targets = images.to(device), masks.to(device)
    
    with torch.no_grad():
        outputs = model(inputs, targets)
        main_output = outputs[0] if isinstance(outputs, tuple) else outputs
        main_output = F.interpolate(main_output, size=targets.shape[-2:], mode='bilinear', align_corners=True)
        preds = main_output.max(1)[1].cpu().numpy()[0]  # Take the first sample prediction
        
    # Unpack the first image and target mask from the batch
    img = images[0].numpy()
    target = masks[0].numpy()
    
    # Reverse Image Normalization to make it viewable (RGB)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_unnorm = np.clip((img * std + mean), 0, 1)
    img_unnorm = np.transpose(img_unnorm, (1, 2, 0))
    
    # Define class names and matching color scheme (RGB formats)
    # Intact=Green, Damaged=Red, Blocked Road=Yellow, Background=Blue/None
    class_colors = {
        0: [0, 255, 0],    # Green
        1: [255, 0, 0],    # Red
        2: [255, 255, 0],  # Yellow
        3: [0, 0, 0]       # Background (keep dark/neutral)
    }
    
    # Create empty RGB color masks for Ground Truth and Prediction
    gt_overlay = np.zeros_like(img_unnorm)
    pred_overlay = np.zeros_like(img_unnorm)
    
    for c in range(4):
        gt_overlay[target == c] = np.array(class_colors[c]) / 255.0
        pred_overlay[preds == c] = np.array(class_colors[c]) / 255.0
        
    # Blend the color masks with the original unnormalized background image (Alpha=0.4)
    alpha = 0.4
    gt_blended = np.clip((1 - alpha) * img_unnorm + alpha * gt_overlay, 0, 1)
    pred_blended = np.clip((1 - alpha) * img_unnorm + alpha * pred_overlay, 0, 1)
    
    # Plot side-by-side: Raw Image, GT Overlay, Prediction Overlay
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_unnorm)
    axes[0].set_title("Original Validation Image")
    axes[0].axis('off')
    
    axes[1].imshow(gt_blended)
    axes[1].set_title("Corrected Ground Truth Overlay")
    axes[1].axis('off')
    
    axes[2].imshow(pred_blended)
    axes[2].set_title(f"Model Prediction Overlay (Epoch {epoch+1})")
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"overlay_epoch_{epoch+1}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[*] Visual progress overlay saved to: {save_path}")
# ==============================================================================
# 4. BOUCLE PRINCIPALE
# ==============================================================================
def main():
    print(f"--- DÉBUT DE L'ENTRAÎNEMENT HYBRIDE (FROM SCRATCH - 4 CLASSES) ---")

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.PILToTensor() 
    ])

    print("Chargement des Datasets...")
    train_dataset = HybrideDataset4Classes(TRAIN_IMAGE_DIR, TRAIN_GT_DIR, img_transform, mask_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    # --------------------------------------------------------------------------
    # AJOUT DE LA VÉRIFICATION VISUELLE DE SÉCURITÉ
    # --------------------------------------------------------------------------
    visually_verify_dataset(train_loader, num_samples=2)
    # --------------------------------------------------------------------------

    val_dataset = HybrideDataset4Classes(VAL_IMAGE_DIR, VAL_GT_DIR, img_transform, mask_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- ARCHITECTURE DU MODÈLE (Construction et Initialisation) ---
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device)

    # This is the correct instantiation line to match your import:
    model = PSPNet(layers=101, bins=(1, 2, 3, 6), dropout=0.1, classes=4, zoom_factor=8, pretrained=False, criterion=criterion)
    
    # Restructuration pour correspondre à votre architecture II-DAMNet
    model.layer0 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    model.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
    model.layer1[0].downsample[0] = nn.Conv2d(64, 256, kernel_size=1, bias=False)
    
    # Couche de classification finale pour 4 classes nettes
    model.cls = nn.Sequential(nn.Conv2d(4096, 4, kernel_size=1, bias=True))
    
    model = nn.DataParallel(model).to(device)
    
    print("✅ Modèle instancié avec succès. Entraînement avec des poids initialisés aléatoirement.")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    history_train_loss, history_train_miou = [], []
    history_val_loss, history_val_miou = [], []
    best_val_miou = 0.0

    for epoch in range(EPOCHS):
        # ==================== PHASE D'ENTRAÎNEMENT ====================
        model.train()
        train_loss = 0.0
        train_hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        for images, masks in pbar_train:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, masks)
            
            main_loss = outputs[1].mean()
            aux_loss = outputs[2].mean()
            loss = main_loss + 0.4 * aux_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            pred = outputs[0].cpu().numpy() 
            target = masks.cpu().numpy()
            
            valid_pixels = (target >= 0) & (target < NUM_CLASSES)
            train_hist += np.bincount(NUM_CLASSES * target[valid_pixels].astype(int) + pred[valid_pixels], 
                                      minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_miou = calculate_miou(train_hist)
        
        # ==================== PHASE DE VALIDATION ====================
        model.eval()
        val_loss = 0.0
        val_hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]")
            for images, masks in pbar_val:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images, masks)
                main_output = outputs[0] if isinstance(outputs, tuple) else outputs
                
                main_output = F.interpolate(main_output, size=masks.shape[-2:], mode='bilinear', align_corners=True)
                
                loss = criterion(main_output, masks)
                val_loss += loss.item()
                
                pred = main_output.max(1)[1].cpu().numpy()
                target = masks.cpu().numpy()
                valid_pixels = (target >= 0) & (target < NUM_CLASSES)
                val_hist += np.bincount(NUM_CLASSES * target[valid_pixels].astype(int) + pred[valid_pixels], 
                                        minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_miou = calculate_miou(val_hist)
        
        scheduler.step()
        
        history_train_loss.append(epoch_train_loss)
        history_train_miou.append(epoch_train_miou)
        history_val_loss.append(epoch_val_loss)
        history_val_miou.append(epoch_val_miou)
        
        print(f"\n--- Bilan Epoch {epoch+1} ---")
        print(f"Train | Loss: {epoch_train_loss:.4f} | mIoU: {epoch_train_miou:.2f}%")
        print(f"Val   | Loss: {epoch_val_loss:.4f} | mIoU: {epoch_val_miou:.2f}%")
        # ----------------------------------------------------------------------
        # CALL OVERLAY EVERY 5 EPOCHS
        # ----------------------------------------------------------------------
        if (epoch + 1) % 5 == 0:
            save_val_overlay_snapshot(model, val_loader, epoch)
        # ----------------------------------------------------------------------
        if epoch_val_miou > best_val_miou:
            best_val_miou = epoch_val_miou
            save_path = os.path.join(SAVE_DIR, "II-DAMNet.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[*] 🔥 Nouveau meilleur modèle sauvegardé : {save_path}")
            
        plot_learning_curves(history_train_loss, history_val_loss, history_train_miou, history_val_miou, 
                             os.path.join(SAVE_DIR, "training_curves_scratch_ad.png"))

if __name__ == '__main__':
    main()