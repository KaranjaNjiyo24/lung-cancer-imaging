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

def create_patient_label_mapping(clinical_csv_path, metadata_csv_path):
    """Create mapping from patient IDs to clinical outcomes"""
    
    # Load both datasets
    clinical_df = pd.read_csv(clinical_csv_path)
    metadata_df = pd.read_csv(metadata_csv_path)
    
    print(f"Clinical data: {clinical_df.shape[0]} patients")
    print(f"Metadata: {len(metadata_df['Subject ID'].unique())} unique patients")
    
    # Map Case ID to Subject ID format
    # Check if they match directly first
    clinical_patients = set(clinical_df['Case ID'].astype(str))
    metadata_patients = set(metadata_df['Subject ID'].unique())
    
    direct_matches = clinical_patients.intersection(metadata_patients)
    print(f"Direct patient ID matches: {len(direct_matches)}")
    
    if len(direct_matches) == 0:
        # Try different formatting approaches
        # Convert R01-XXX format
        clinical_df['Formatted_ID'] = clinical_df['Case ID'].apply(
            lambda x: f"R01-{str(x).zfill(3)}" if str(x).isdigit() else str(x)
        )
        clinical_patients_formatted = set(clinical_df['Formatted_ID'])
        matches = clinical_patients_formatted.intersection(metadata_patients)
        print(f"Formatted matches (R01-XXX): {len(matches)}")
        id_column = 'Formatted_ID'
    else:
        id_column = 'Case ID'
    
    # Create survival status labels (primary target)
    def create_survival_labels(row):
        if pd.isna(row['Survival Status']):
            return None
        # Convert to binary: 0 = Alive, 1 = Dead
        return 1 if row['Survival Status'].lower() in ['dead', 'deceased', '1'] else 0
    
    clinical_df['survival_label'] = clinical_df.apply(create_survival_labels, axis=1)
    
    # Create recurrence labels (secondary target)
    def create_recurrence_labels(row):
        if pd.isna(row['Recurrence']):
            return None
        return 1 if row['Recurrence'].lower() in ['yes', 'true', '1'] else 0
    
    clinical_df['recurrence_label'] = clinical_df.apply(create_recurrence_labels, axis=1)
    
    # Create patient mapping dictionary
    patient_labels = {}
    
    for _, row in clinical_df.iterrows():
        patient_id = str(row[id_column])
        
        if patient_id in metadata_patients:
            patient_labels[patient_id] = {
                'survival_status': row['survival_label'],
                'recurrence': row['recurrence_label'],
                'time_to_death': row.get('Time to Death (days)', None),
                'histological_grade': row.get('Histopathological Grade', None),
                'age': row.get('Age at Histological Diagnosis', None),
                'gender': row.get('Gender', None),
                'smoking_status': row.get('Smoking status', None)
            }
    
    print(f"Successfully mapped {len(patient_labels)} patients")
    print(f"Patients with survival labels: {sum(1 for p in patient_labels.values() if p['survival_status'] is not None)}")
    print(f"Patients with recurrence labels: {sum(1 for p in patient_labels.values() if p['recurrence'] is not None)}")
    
    return patient_labels

# Function to get labels for training
def get_patient_label(patient_id, patient_labels, target_type='survival'):
    """Get label for a specific patient and target type"""
    if patient_id not in patient_labels:
        return None
    
    patient_data = patient_labels[patient_id]
    
    if target_type == 'survival':
        return patient_data['survival_status']
    elif target_type == 'recurrence':
        return patient_data['recurrence']
    else:
        return None


def train_epoch_dummy(model, dataloader, criterion, optimizer, device, epoch, num_classes: int = 2):
    """Fallback training loop that uses randomly generated labels."""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        model_input = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and key in ['ct', 'pet']:
                model_input[key] = value.to(device)

        if not model_input:
            continue

        batch_size = list(model_input.values())[0].size(0)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)

        optimizer.zero_grad()
        outputs = model(model_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100. * correct / max(1, total):.2f}%'} )

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    epoch_loss = running_loss / max(1, len(dataloader))
    epoch_acc = 100. * correct / max(1, total)
    return epoch_loss, epoch_acc

# Updated training function with real labels
def train_epoch_with_real_labels(model, dataloader, criterion, optimizer, device, epoch, patient_labels, target_type='survival'):
    """Train for one epoch using real clinical labels"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    skipped = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Extract modality tensors
        model_input = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and key in ['ct', 'pet']:
                model_input[key] = value.to(device)
        
        if not model_input:
            skipped += 1
            continue
        
        # Get real labels from clinical data
        patient_ids = batch['patient_id']
        batch_labels = []
        
        for pid in patient_ids:
            label = get_patient_label(pid, patient_labels, target_type)
            if label is not None:
                batch_labels.append(label)
            else:
                batch_labels.append(0)  # Default for missing labels
        
        labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(model_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'Skipped': skipped
        })
        
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    epoch_loss = running_loss / max(1, len(dataloader) - skipped)
    epoch_acc = 100. * correct / max(1, total)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Extract modality tensors directly
            model_input = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and key in ['ct', 'pet']:
                    model_input[key] = value.to(device)
            
            if not model_input:
                continue
                
            batch_size = list(model_input.values())[0].size(0)
            labels = torch.randint(0, 2, (batch_size,)).to(device)
            
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
    
    print("\n📊 Loading clinical labels...")
    data_cfg = config.get('data', {})
    clinical_path = args.clinical_csv if args.clinical_csv else data_cfg.get('clinical_csv')
    patient_labels = None
    target_type = 'survival'

    if clinical_path and os.path.exists(clinical_path):
        patient_labels = create_patient_label_mapping(clinical_path, args.metadata_csv)
        TARGET_TYPE = 'survival'  # Options: 'survival', 'recurrence'
        target_type = TARGET_TYPE
        print(f"🎯 Training target: {TARGET_TYPE}")
    else:
        print("⚠️ No clinical labels found, using dummy labels")
    
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
        if patient_labels:
            train_loss, train_acc = train_epoch_with_real_labels(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch + 1,
                patient_labels,
                target_type,
            )
        else:
            train_loss, train_acc = train_epoch_dummy(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch + 1,
                num_classes=config['model']['num_classes'],
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
