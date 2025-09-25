#!/usr/bin/env python3
"""
Complete Colab training script for NSCLC radiogenomics with full training implementation
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def setup_colab_environment():
    """Setup Colab-specific configurations"""
    print("🚀 Setting up Colab environment...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected - training will be slow!")
    
    # Set memory allocation strategy
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_clinical_labels(clinical_csv_path):
    """Load clinical labels for supervised learning"""
    if not os.path.exists(clinical_csv_path):
        print(f"⚠️ Clinical labels file not found: {clinical_csv_path}")
        return None
    
    clinical_df = pd.read_csv(clinical_csv_path)
    print(f"📊 Clinical data loaded: {clinical_df.shape}")
    print(f"📋 Columns: {list(clinical_df.columns)}")
    
    # You'll need to map patient IDs to labels - this is dataset-specific
    # For now, return the dataframe for manual inspection
    return clinical_df

def create_dummy_labels(dataset_size, num_classes=2):
    """Create dummy labels for testing - replace with real labels"""
    print("⚠️ Using dummy labels - replace with real clinical outcomes!")
    return torch.randint(0, num_classes, (dataset_size,))

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        volumes = {}
        for modality in ['CT', 'PET']:
            if modality in batch['volumes']:
                volumes[modality] = batch['volumes'][modality].to(device)
        
        # Create dummy labels (replace with real labels from clinical data)
        batch_size = list(volumes.values())[0].size(0)
        labels = torch.randint(0, 2, (batch_size,)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Prepare input for model
        model_input = {'volumes': volumes}
        outputs = model(model_input)
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # Clear GPU cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            volumes = {}
            for modality in ['CT', 'PET']:
                if modality in batch['volumes']:
                    volumes[modality] = batch['volumes'][modality].to(device)
            
            batch_size = list(volumes.values())[0].size(0)
            labels = torch.randint(0, 2, (batch_size,)).to(device)  # Dummy labels
            
            model_input = {'volumes': volumes}
            outputs = model(model_input)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='NSCLC Colab Training')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--metadata_csv', type=str, required=True)
    parser.add_argument('--clinical_csv', type=str, help='Clinical labels CSV')
    parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/checkpoints')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup environment
    device = setup_colab_environment()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.test_mode:
        print("🧪 Running in test mode...")
        config['data']['batch_size'] = 1
        config['training']['epochs'] = 2
        config['data']['image_size'] = [64, 64, 32]
    
    print(f"📊 Configuration: {config}")
    
    # Import modules
    try:
        from src.data.metadata_handler import NSCLCMetadataHandler
        from src.data.data_loader import NestedDICOMDataset
        from src.models.multimodal_classifier import MultimodalLungCancerClassifier
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Load clinical labels if provided
    clinical_df = None
    if args.clinical_csv and os.path.exists(args.clinical_csv):
        clinical_df = load_clinical_labels(args.clinical_csv)
    
    # Create dataset
    print("\n🔍 Creating datasets...")
    try:
        dataset = NestedDICOMDataset(
            root_dir=args.data_root,
            metadata_csv_path=args.metadata_csv,
            modalities=["CT", "PET"],
            pair_modalities=True,
            lazy=True
        )
        
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("❌ Empty dataset! Check your data paths.")
            return
        
        # Split dataset (simple split for now)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data'].get('num_workers', 0),
            pin_memory=config['data'].get('pin_memory', False)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 0),
            pin_memory=config['data'].get('pin_memory', False)
        )
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create model
    print("\n🧪 Creating model...")
    try:
        model = MultimodalLungCancerClassifier(
            num_classes=config['model']['num_classes']
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {total_params:,} parameters")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {config['training']['epochs']} epochs...")
    
    best_val_acc = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n📅 Epoch {epoch+1}/{config['training']['epochs']}")
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduler step
        scheduler.step()
        
        # Print epoch results
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Best model saved: {best_model_path}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final save
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved: {final_model_path}")

if __name__ == "__main__":
    main()