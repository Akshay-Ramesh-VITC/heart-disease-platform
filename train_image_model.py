import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import glob

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset Class
class CardiacSegmentationDataset(Dataset):
    def __init__(self, root_dir, img_size=256, transform=None):
        """
        Args:
            root_dir: Path to dataset root
            img_size: Target image size for resizing
            transform: Optional transforms
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        
        print(f"Loading dataset from: {root_dir}")
        
        # Common folder structures to check
        possible_structures = [
            # Structure 1: images/ and labels/ folders
            {'images': 'images', 'masks': 'labels'},
            # Structure 2: images/ and masks/ folders
            {'images': 'images', 'masks': 'masks'},
            {'images': 'image', 'masks': 'mask'},
            {'images': 'Images', 'masks': 'Masks'},
            {'images': 'Images', 'masks': 'Labels'},
            # Structure 3: train/images and train/masks
            {'images': 'train/images', 'masks': 'train/masks'},
            {'images': 'train/images', 'masks': 'train/labels'},
            # Structure 4: data/images and data/masks
            {'images': 'data/images', 'masks': 'data/masks'},
            {'images': 'data/images', 'masks': 'data/labels'},
            # Structure 5: flat structure with naming pattern
            {'images': '', 'masks': ''},
        ]
        
        # Try to find the correct structure
        for structure in possible_structures:
            img_path = os.path.join(root_dir, structure['images'])
            mask_path = os.path.join(root_dir, structure['masks'])
            
            if structure['images'] == '':
                # Flat structure - find files by pattern
                img_files = glob.glob(os.path.join(root_dir, '*.*'))
                img_files = [f for f in img_files if not ('mask' in f.lower() or 'label' in f.lower())]
            else:
                if os.path.exists(img_path):
                    img_files = glob.glob(os.path.join(img_path, '*.*'))
                else:
                    continue
            
            if len(img_files) > 0:
                print(f"Found {len(img_files)} images in: {img_path if structure['images'] != '' else root_dir}")
                
                # Debug: Check mask folder
                if structure['masks'] != '':
                    mask_folder = os.path.join(root_dir, structure['masks'])
                    if os.path.exists(mask_folder):
                        mask_files_in_folder = os.listdir(mask_folder)
                        print(f"Found {len(mask_files_in_folder)} files in mask folder: {mask_folder}")
                
                for img_file in img_files:
                    # Try to find corresponding mask
                    base_name = os.path.basename(img_file)
                    name_without_ext = os.path.splitext(base_name)[0]
                    img_ext = os.path.splitext(base_name)[1]
                    
                    # Common mask naming patterns
                    mask_patterns = [
                        # Same name, same extension
                        f"{name_without_ext}{img_ext}",
                        # Same name, different extensions
                        f"{name_without_ext}.png",
                        f"{name_without_ext}.jpg",
                        f"{name_without_ext}.jpeg",
                        # With suffix
                        f"{name_without_ext}_mask{img_ext}",
                        f"{name_without_ext}_label{img_ext}",
                        f"{name_without_ext}_mask.png",
                        f"{name_without_ext}_label.png",
                        # With prefix
                        f"mask_{name_without_ext}{img_ext}",
                        f"label_{name_without_ext}{img_ext}",
                    ]
                    
                    mask_file = None
                    for pattern in mask_patterns:
                        if structure['masks'] == '':
                            search_path = os.path.join(root_dir, pattern)
                        else:
                            search_path = os.path.join(root_dir, structure['masks'], pattern)
                        
                        if os.path.exists(search_path) and search_path != img_file:
                            mask_file = search_path
                            break
                    
                    if mask_file and os.path.exists(mask_file):
                        self.samples.append({
                            'image': img_file,
                            'mask': mask_file
                        })
                
                if len(self.samples) > 0:
                    print(f"Successfully matched {len(self.samples)} image-mask pairs")
                    break
                else:
                    print(f"Found images but couldn't match with masks in {structure['masks']}")
        
        print(f"\nTotal: {len(self.samples)} image-mask pairs loaded")
        
        if len(self.samples) == 0:
            print("\n⚠️ Dataset structure not recognized. Debugging info:")
            print(f"Root directory: {root_dir}")
            if os.path.exists(root_dir):
                print("Available folders/files:", os.listdir(root_dir))
                
                # Try to give helpful hints
                img_folder = os.path.join(root_dir, 'images')
                label_folder = os.path.join(root_dir, 'labels')
                
                if os.path.exists(img_folder) and os.path.exists(label_folder):
                    img_samples = os.listdir(img_folder)[:3]
                    label_samples = os.listdir(label_folder)[:3]
                    print(f"\nSample image files: {img_samples}")
                    print(f"Sample label files: {label_samples}")
                    print("\n💡 Tip: Make sure image and label files have matching names!")
            else:
                print("❌ Directory not found!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img = cv2.imread(self.samples[idx]['image'])
            if img is None:
                img = np.array(Image.open(self.samples[idx]['image']))
            
            # Load mask
            mask = cv2.imread(self.samples[idx]['mask'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.array(Image.open(self.samples[idx]['mask']).convert('L'))
            
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to grayscale if RGB
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Resize
            img = cv2.resize(img, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
            # Normalize image to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Normalize mask to class indices (0, 1, 2, ...)
            unique_values = np.unique(mask)
            
            # Create a mapping from pixel values to class indices
            mask_normalized = np.zeros_like(mask, dtype=np.uint8)
            for class_idx, pixel_value in enumerate(sorted(unique_values)):
                mask_normalized[mask == pixel_value] = class_idx
            
            # Ensure all values are valid
            max_val = mask_normalized.max()
            if max_val >= 255:  # Sanity check
                print(f"Warning: Sample {idx} has unusual mask values. Normalizing...")
                mask_normalized = (mask_normalized / max_val * 1).astype(np.uint8)
            
            # Convert to tensors
            img = torch.FloatTensor(img).unsqueeze(0)  # Add channel dimension
            mask = torch.LongTensor(mask_normalized)
            
            # Final safety check - clamp mask values
            mask = torch.clamp(mask, 0, 1)  # Since we only have 2 classes
            
            if self.transform:
                img = self.transform(img)
            
            return img, mask
        
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy sample
            return torch.zeros(1, self.img_size, self.img_size), torch.zeros(self.img_size, self.img_size, dtype=torch.long)


# U-Net Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 256)
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 128)
        self.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        return self.outc(x)


# Combined Loss: Dice + CrossEntropy
class CombinedLoss(nn.Module):
    def __init__(self, n_classes=4, weight_dice=0.5, weight_ce=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce_loss = nn.CrossEntropyLoss()
    
    def dice_loss(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(pred * target_one_hot, dims)
        cardinality = torch.sum(pred + target_one_hot, dims)
        
        dice = (2. * intersection + 1e-7) / (cardinality + 1e-7)
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.weight_dice * dice + self.weight_ce * ce


# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='Training')
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Validation'):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(loader)


# Main training script
def main():
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    IMG_SIZE = 256
    
    # Dataset path - adjust for Kaggle
    DATA_ROOT = '/kaggle/input/cardiac-semantic-segmentation-dataset/cardiac'
    
    # Create dataset
    full_dataset = CardiacSegmentationDataset(DATA_ROOT, img_size=IMG_SIZE)
    
    if len(full_dataset) == 0:
        print("\nERROR: No samples found! Please check dataset structure.")
        return
    
    # Determine number of classes by checking multiple samples
    print("\nAnalyzing dataset classes...")
    all_unique_values = set()
    
    # Check first 100 samples to get a good estimate
    check_count = min(100, len(full_dataset))
    for i in range(check_count):
        try:
            _, mask = full_dataset[i]
            unique_vals = torch.unique(mask).cpu().numpy()
            all_unique_values.update(unique_vals)
            
            if i == 0:
                print(f"Sample {i}: unique values = {unique_vals}, shape = {mask.shape}")
        except Exception as e:
            print(f"Error checking sample {i}: {e}")
            continue
    
    num_classes = len(all_unique_values)
    print(f"All unique mask values across {check_count} samples: {sorted(all_unique_values)}")
    print(f"Total number of classes: {num_classes}")
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model
    model = UNet(n_channels=1, n_classes=num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(n_classes=num_classes, weight_dice=0.7, weight_ce=0.3)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'num_classes': num_classes,
            }, 'best_cardiac_model.pth')
            print(f'✓ Saved best model with val loss: {val_loss:.4f}')
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'num_classes': num_classes,
    }, 'final_cardiac_model.pth')
    
    print('\n' + '='*50)
    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print('='*50)


if __name__ == '__main__':
    main()