import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F

# Import de l'architecture PSPNet originale
from models.pspnet import PSPNet

# ==============================================================================
# CONFIGURATION DYNAMIQUE (via argparse)
# ==============================================================================
parser = argparse.ArgumentParser(description="Évaluation du modèle hybride PSPNet")
parser.add_argument('--image_dir', type=str, required=True, help="Chemin vers le dossier des images de test")
parser.add_argument('--label_dir', type=str, required=True, help="Chemin vers le dossier des labels (Ground Truth) de test")
parser.add_argument('--model_path', type=str, 
                    default="checkpoints/II-DAMNet/II-DAMNet.pth", 
                    help="Chemin vers les poids du modèle (optionnel)")
parser.add_argument('--batch_size', type=int, default=32, help="Taille du batch")
args = parser.parse_args()

TEST_IMAGE_DIR = args.image_dir
TEST_GT_DIR    = args.label_dir
MODEL_PATH     = args.model_path
BATCH_SIZE     = args.batch_size

NUM_CLASSES = 4
IGNORE_INDEX = 255   

# Noms des classes pour l'affichage final
CLASS_NAMES = [
    "Bâtiment Intact", 
    "Bâtiment Endommagé", 
    "Route Bloquée", 
    "Fond (Background)"
]
# ==============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# 1. FONCTION DE MAPPING PYTORCH (11 Classes -> 4 Classes)
# ==============================================================================
def map_mask_to_4_classes(mask_tensor):
    mapped = torch.full_like(mask_tensor, IGNORE_INDEX)
    mapped[mask_tensor == 2] = 0                                       # Bâtiment Intact
    mapped[(mask_tensor == 3) | (mask_tensor == 4) | (mask_tensor == 5)] = 1 # Endommagé
    mapped[mask_tensor == 8] = 2                                       # Route Bloquée
    mapped[(mask_tensor == 0) | (mask_tensor == 1) | (mask_tensor == 6) | 
           (mask_tensor == 7) | (mask_tensor == 9) | (mask_tensor == 10)] = 3 # Fond
    return mapped

# ==============================================================================
# 2. CLASSE DATASET POUR LE TEST
# ==============================================================================
class TestDataset4Classes(Dataset):
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
        
        # 1. Obtenir le nom sans l'extension (.jpg, .png)
        base_name = os.path.splitext(img_name)[0]
        
        # 2. NETTOYAGE DES PRÉFIXES : On retire "kind_" ou "trad_night_" s'ils existent
        clean_name = base_name
        if clean_name.startswith("kind_"):
            clean_name = clean_name.replace("kind_", "", 1)
        elif clean_name.startswith("trad_night_"):
            clean_name = clean_name.replace("trad_night_", "", 1)
        
        # 3. Construire le chemin vers le label en utilisant le nom nettoyé
        gt_path = os.path.join(self.gt_dir, f"day_{clean_name}_lab.png") 
        
        img_pil = Image.open(img_path).convert('RGB')
        
        # Gestion de sécurité renforcée pour le debugging
        if os.path.exists(gt_path):
            gt_pil = Image.open(gt_path).convert('L')
        else:
            # S'il ne trouve pas le label, il crashera en vous disant exactement ce qu'il cherchait
            raise FileNotFoundError(f"Label introuvable: {gt_path} \n(Déduit à partir de l'image: {img_name})")
        
        if self.img_transform:
            img_tensor = self.img_transform(img_pil)
        if self.mask_transform:
            gt_tensor = self.mask_transform(gt_pil)
            gt_tensor = torch.squeeze(gt_tensor).long() 
            gt_tensor = map_mask_to_4_classes(gt_tensor)

        return img_tensor, gt_tensor

# ==============================================================================
# 3. BOUCLE D'ÉVALUATION PRINCIPALE
# ==============================================================================
def main():
    print(f"--- DÉBUT DE L'ÉVALUATION ---")
    print(f"Images : {TEST_IMAGE_DIR}")
    print(f"Labels : {TEST_GT_DIR}")
    print(f"Modèle : {MODEL_PATH}")

    # Normalisation stricte identique à l'entraînement
    img_transform = transforms.Compose([
        transforms.Resize((713, 713)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.PILToTensor() 
    ])

    test_dataset = TestDataset4Classes(TEST_IMAGE_DIR, TEST_GT_DIR, img_transform, mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Images trouvées dans le Test Set : {len(test_dataset)}")

    # --- ARCHITECTURE DU MODÈLE ---
    model = PSPNet(layers=101, bins=(1, 2, 3, 6), dropout=0.1, classes=4, zoom_factor=8, pretrained=False)
    
    model.layer0 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    model.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
    model.layer1[0].downsample[0] = nn.Conv2d(64, 256, kernel_size=1, bias=False)
    model.cls = nn.Sequential(nn.Conv2d(4096, 4, kernel_size=1, bias=True))
    
    # Gestion du chargement des poids
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '', 1) if k.startswith('module.') else k
        
        # ---> LA CORRECTION EST ICI : On ignore les poids du criterion <---
        if 'criterion' in name:
            continue
            
        new_state_dict[name] = v
        
    # On peut garder strict=True car on a filtré manuellement
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))

    with torch.no_grad():
        pbar_test = tqdm(test_loader, desc="Inférence Test")
        for images, masks in pbar_test:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            main_output = outputs[0] if isinstance(outputs, tuple) else outputs
            
            main_output = F.interpolate(main_output, size=masks.shape[-2:], mode='bilinear', align_corners=True)
            
            pred = main_output.max(1)[1].cpu().numpy()
            target = masks.cpu().numpy()
            
            valid_pixels = (target >= 0) & (target < NUM_CLASSES)
            hist += np.bincount(NUM_CLASSES * target[valid_pixels].astype(int) + pred[valid_pixels], 
                                minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)

    # --- CALCUL IoU ---
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou = np.zeros(NUM_CLASSES)
    
    valid_mask = union > 0
    iou[valid_mask] = (intersection[valid_mask] / union[valid_mask]) * 100
    
    miou = np.nanmean(iou)

    print("\n" + "="*50)
    print(" RÉSULTATS D'ÉVALUATION : TEST SET")
    print("="*50)
    for i, cls_name in enumerate(CLASS_NAMES):
        print(f"{cls_name.ljust(25)} : {iou[i]:.2f}%")
    print("-" * 50)
    print(f"{'Mean IoU (mIoU)'.ljust(25)} : {miou:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()