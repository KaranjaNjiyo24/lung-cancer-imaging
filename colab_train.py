#!/usr/bin/env python3
"""
Colab-ready training script for NSCLC radiogenomics
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml


def setup_colab_environment():
    """Setup Colab-specific configurations"""
    print("🚀 Setting up Colab environment...")

    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        print("⚠️ No GPU detected - training will be slow!")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser(description="NSCLC Colab Training")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/checkpoints")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with small dataset")

    args = parser.parse_args()

    device = setup_colab_environment()

    config = load_config(args.config)

    if args.test_mode:
        print("🧪 Running in test mode...")
        config.setdefault("data", {})
        config.setdefault("training", {})
        config["data"]["batch_size"] = 1
        config["training"]["epochs"] = 2
        config["data"]["image_size"] = [64, 64, 32]

    print(f"📊 Configuration: {config}")

    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from src.data.data_loader import NestedDICOMDataset
        from src.data.metadata_handler import NSCLCMetadataHandler
        from src.models.multimodal_classifier import MultimodalLungCancerClassifier
        print("✅ All modules imported successfully")
    except ImportError as exc:
        print(f"❌ Import error: {exc}")
        return

    print("\n🔍 Testing data loading...")
    try:
        dataset = NestedDICOMDataset(
            root_dir=args.data_root,
            metadata_csv_path=args.metadata_csv,
            modalities=["CT", "PET"],
            pair_modalities=True,
            lazy=True,
        )
        print(f"✅ Dataset created: {len(dataset)} samples")

        if len(dataset) > 0:
            print("🧪 Testing first sample...")
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Patient: {sample['patient_id']}")

            volumes = sample["volumes"]
            for modality, volume in volumes.items():
                print(f"{modality}: {volume.shape}, {volume.dtype}")

    except Exception as exc:
        print(f"❌ Data loading failed: {exc}")
        import traceback

        traceback.print_exc()
        return

    print("\n🧪 Testing model creation...")
    try:
        model = MultimodalLungCancerClassifier(
            num_classes=config.get("model", {}).get("num_classes", 2)
        ).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {params:,} parameters")
    except Exception as exc:
        print(f"❌ Model creation failed: {exc}")
        import traceback

        traceback.print_exc()
        return

    print("\n🎉 Colab setup test completed successfully!")
    print("\nNext steps:")
    print("1. Fix any issues shown above")
    print("2. Run with --test_mode for 2-epoch training test")
    print("3. Scale up to full training")


if __name__ == "__main__":
    main()
