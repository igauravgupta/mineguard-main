#!/usr/bin/env python3
"""
Training script for company-specific incident classification models
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class CompanyIncidentClassifier(nn.Module):
    """Company-specific incident classifier"""
    
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_data_transforms():
    """Get training and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description='Train company-specific incident classifier')
    parser.add_argument('company_id', type=str, help='Company ID (e.g., 00110)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    company_dir = base_dir / 'companies' / args.company_id
    config_path = company_dir / 'config.json'
    train_data_dir = company_dir / 'training_data' / 'train'
    models_dir = company_dir / 'models'
    
    # Verify paths
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return
    
    if not train_data_dir.exists():
        print(f"❌ Training data not found: {train_data_dir}")
        print(f"Please create training data folders in: {company_dir / 'training_data'}")
        return
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    incident_types = config['incident_types']
    num_classes = len(incident_types)
    
    print(f"\n{'='*60}")
    print(f"Training Model for Company: {config['company_name']} ({args.company_id})")
    print(f"{'='*60}")
    print(f"Incident Types ({num_classes}):")
    for i, incident in enumerate(incident_types, 1):
        print(f"  {i}. {incident}")
    print(f"{'='*60}\n")
    
    # Create models directory
    models_dir.mkdir(exist_ok=True)
    
    # Get data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(train_data_dir, transform=train_transform)
    
    # Check if we have images
    if len(full_dataset) == 0:
        print(f"❌ No images found in {train_data_dir}")
        print("Please add images to training_data/train/<incident_type>/ folders")
        return
    
    print(f"✓ Found {len(full_dataset)} training images")
    print(f"  Classes: {full_dataset.classes}")
    
    # Split dataset
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"  Train: {train_size} | Validation: {val_size}\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    model = CompanyIncidentClassifier(num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = models_dir / 'best_model.pth'
    
    print("Starting training...\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': full_dataset.classes,
                'company_id': args.company_id
            }, best_model_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {best_model_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
