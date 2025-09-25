"""PyTorch dataset utilities for multimodal NSCLC imaging data."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dicom_processor import DICOMProcessor
from .metadata_handler import NSCLCMetadataHandler

try:  # pragma: no cover - optional import
    from torchvision.transforms import Compose
except ImportError:  # pragma: no cover
    Compose = None  # type: ignore


class NSCLCMultimodalDataset(Dataset):
    """Return CT/PET volumes (or placeholders) aligned by patient."""

    def __init__(
        self,
        data_root: str,
        metadata_handler: NSCLCMetadataHandler,
        transform: Optional[object] = None,
        target_size: Tuple[int, int, int] = (128, 128, 64),
        require_both_modalities: bool = False,
        label_field: Optional[str] = None,
        fallback_label: int = -1,
        augmentation_prob: float = 0.0,
        seed: Optional[int] = None,
        fill_missing_with_zeros: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_handler = metadata_handler
        if self.metadata_handler.metadata_df is None:
            self.metadata_handler.load_metadata()

        self.transform = transform
        self.target_size = target_size
        self.require_both_modalities = require_both_modalities
        self.label_field = label_field
        self.fallback_label = fallback_label
        self.augmentation_prob = augmentation_prob
        self.random = random.Random(seed)
        self.fill_missing_with_zeros = fill_missing_with_zeros

        self.dicom_processor = DICOMProcessor(target_spacing=(1.0, 1.0, 1.0))

        patients = self.metadata_handler.get_patient_ids()
        multimodal = set(self.metadata_handler.get_multimodal_patients())

        if require_both_modalities:
            candidates = [pid for pid in patients if pid in multimodal]
        else:
            candidates = patients

        self.patient_ids = [pid for pid in candidates if (self.data_root / pid).exists()]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.patient_ids)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, object]:
        patient_id = self.patient_ids[idx]
        file_map = self._find_patient_files(patient_id)

        ct_tensor, ct_available = self._load_modality(file_map.get("CT"), is_ct=True)
        pet_tensor, pet_available = self._load_modality(file_map.get("PET"), is_ct=False)

        if self.require_both_modalities and not (ct_available and pet_available):
            raise RuntimeError(
                f"Patient {patient_id} is missing a required modality (ct={ct_available}, pet={pet_available})."
            )

        metadata = file_map.get("metadata", {})
        label = self._extract_label(patient_id, metadata)

        sample = {
            "patient_id": patient_id,
            "ct": ct_tensor,
            "pet": pet_tensor,
            "ct_available": ct_available,
            "pet_available": pet_available,
            "metadata": metadata,
            "label": label,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        if self.augmentation_prob > 0 and self.random.random() < self.augmentation_prob:
            sample = self._random_flip(sample)

        return sample

    # ------------------------------------------------------------------
    def _load_modality(self, series_path: Optional[str], is_ct: bool) -> Tuple[Optional[torch.Tensor], bool]:
        if series_path is None:
            if self.fill_missing_with_zeros:
                zeros = torch.zeros((1, *self.target_size), dtype=torch.float32)
                return zeros, False
            return None, False

        volume = self.dicom_processor.load_dicom_series(series_path)
        if volume is None:
            if self.fill_missing_with_zeros:
                zeros = torch.zeros((1, *self.target_size), dtype=torch.float32)
                return zeros, False
            return None, False

        volume = self.dicom_processor.preprocess_ct(volume) if is_ct else self.dicom_processor.preprocess_pet(volume)
        volume = self.dicom_processor.resize_volume(volume, self.target_size)
        tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0)
        return tensor, True

    # ------------------------------------------------------------------
    def _find_patient_files(self, patient_id: str) -> Dict[str, object]:
        patient_dir = self.data_root / patient_id
        result: Dict[str, object] = {"metadata": {}}

        if not patient_dir.exists():
            return result

        session_dirs = sorted([p for p in patient_dir.iterdir() if p.is_dir()])
        metadata_mapping = self.metadata_handler.patient_mapping.get(patient_id, {})
        modality_map: Dict[str, List[str]] = {
            mod.upper(): paths for mod, paths in metadata_mapping.get("files_by_modality", {}).items()
        }

        for session_dir in session_dirs:
            ct_path = self._find_series_for_modality(patient_dir, session_dir, "CT", modality_map.get("CT", []))
            pet_path = self._find_series_for_modality(patient_dir, session_dir, "PET", modality_map.get("PET", []))

            if self.require_both_modalities and not (ct_path and pet_path):
                continue

            result.update({"CT": ct_path, "PET": pet_path})
            result["metadata"] = {
                "session": session_dir.name,
                "ct_path": ct_path,
                "pet_path": pet_path,
                "modalities": [mod for mod, path in (("CT", ct_path), ("PET", pet_path)) if path],
            }
            break
        else:
            # Fall back to any modality we can locate across sessions.
            result["CT"] = self._find_series_for_modality(patient_dir, None, "CT", modality_map.get("CT", []))
            result["PET"] = self._find_series_for_modality(patient_dir, None, "PET", modality_map.get("PET", []))
            result["metadata"] = {
                "session": None,
                "ct_path": result.get("CT"),
                "pet_path": result.get("PET"),
                "modalities": [mod for mod, path in (("CT", result.get("CT")), ("PET", result.get("PET"))) if path],
            }

        return result

    # ------------------------------------------------------------------
    def _find_series_for_modality(
        self,
        patient_dir: Path,
        session_dir: Optional[Path],
        modality: str,
        metadata_paths: List[str],
    ) -> Optional[str]:
        # 1) Try metadata-provided paths.
        for rel_path in metadata_paths:
            candidate = self._resolve_series_path(patient_dir, rel_path)
            if candidate.exists():
                return str(candidate)

        # 2) Look inside the current session.
        search_dirs = [session_dir] if session_dir else [p for p in patient_dir.iterdir() if p.is_dir()]
        for base_dir in search_dirs:
            if base_dir is None or not base_dir.exists():
                continue
            for series_dir in base_dir.rglob("*"):
                if series_dir.is_dir() and modality in series_dir.name.upper():
                    return str(series_dir)
        return None

    # ------------------------------------------------------------------
    def _resolve_series_path(self, patient_dir: Path, raw_path: str) -> Path:
        raw_path = raw_path.strip()
        path = Path(raw_path)
        if path.is_absolute():
            return path

        candidate = patient_dir / path
        if candidate.exists():
            return candidate

        drive_candidate = self.data_root / path
        if drive_candidate.exists():
            return drive_candidate

        return candidate

    # ------------------------------------------------------------------
    def _extract_label(self, patient_id: str, metadata: Dict[str, object]) -> int:
        if self.label_field and self.metadata_handler.metadata_df is not None:
            df = self.metadata_handler.metadata_df
            mask = df["Subject ID"].astype(str) == str(patient_id)
            if mask.any() and self.label_field in df.columns:
                value = df.loc[mask, self.label_field].iloc[0]
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        label = metadata.get("label")
        if isinstance(label, (int, np.integer)):
            return int(label)
        return int(self.fallback_label)

    # ------------------------------------------------------------------
    def _random_flip(self, sample: Dict[str, object]) -> Dict[str, object]:
        axes = [0, 1, 2]
        selected_axes = [axis for axis in axes if self.random.random() < 0.5]
        for key in ("ct", "pet"):
            tensor = sample.get(key)
            if tensor is None:
                continue
            for axis in selected_axes:
                tensor = torch.flip(tensor, dims=(axis + 1,))
            sample[key] = tensor
        return sample


# ----------------------------------------------------------------------
def create_nsclc_dataloader(
    dataset: NSCLCMultimodalDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: Optional[bool] = None,
) -> DataLoader:
    """Return a memory-efficient DataLoader tuned for volumetric data."""

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


__all__ = ["NSCLCMultimodalDataset", "create_nsclc_dataloader"]
