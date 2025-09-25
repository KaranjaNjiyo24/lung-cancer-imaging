"""DICOM data loading utilities for the NSCLC radiogenomics dataset.

The folder structure is assumed to be::

    root/
        PatientID/
            SessionDate/
                SeriesName/
                    image1.dcm
                    image2.dcm
                    ...

This module provides a :class:`NestedDICOMDataset` that integrates with the
PyTorch ``DataLoader`` and yields either single-modality series or
multi-modality pairs (e.g. CT + PET) on demand.  The implementation is lazy –
only the file paths are indexed up-front – keeping memory usage low for large
collections (the project brief mentions ~98 GB of imaging data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch might not be available in all environments
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - fallback for environments without torch
    class Dataset:  # type: ignore
        """Minimal Dataset shim used when PyTorch is not installed."""

        def __len__(self) -> int:  # pragma: no cover - debug helper
            raise NotImplementedError("PyTorch is required for Dataset usage")

        def __getitem__(self, index: int):  # pragma: no cover - debug helper
            raise NotImplementedError("PyTorch is required for Dataset usage")

try:  # pragma: no cover - optional dependency
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None  # type: ignore

from .metadata_handler import NSCLCMetadataHandler

LOGGER = logging.getLogger(__name__)

# Type alias for a callable that converts a list of DICOM paths to an array-like
VolumeLoader = Callable[[List[Path]], np.ndarray]


@dataclass
class SeriesSample:
    """Lightweight container describing a single DICOM series."""

    patient_id: str
    session_id: str
    modality: str
    series_id: str
    dicom_files: List[Path]
    metadata: Dict[str, object]


class NestedDICOMDataset(Dataset):
    """Dataset that walks a nested DICOM directory tree lazily.

    Parameters
    ----------
    root_dir:
        Root directory containing patient folders.
    metadata_handler:
        Optional :class:`NSCLCMetadataHandler` instance.  If ``None`` and
        ``metadata_csv_path`` is provided, the handler will be created
        automatically.
    metadata_csv_path:
        Path to ``metadata.csv`` when a handler instance is not provided.
    modalities:
        Modalities to index (e.g. ``["CT", "PET"]``).  ``None`` keeps every
        modality discovered on disk.
    pair_modalities:
        When ``True`` the dataset yields stacks of modalities for the same
        patient/session (e.g. CT+PET pairs).  When ``False`` each item
        represents a single DICOM series.
    required_modalities:
        Modalities that must be present in a pair when ``pair_modalities`` is
        enabled.  Defaults to ``("CT", "PET")``.
    transform:
        Optional callable applied to the volume before returning it.  Receives
        ``(volume, metadata)`` and should return the transformed volume.  If the
        callable only accepts a single argument it will be invoked with the
        volume alone.
    volume_loader:
        Optional callable responsible for reading the DICOM files.  Defaults to
        a loader backed by ``pydicom``; provide a custom implementation to use a
        different backend or to enable caching.
    lazy:
        When ``True`` only file paths are stored during indexing (recommended
        for large datasets).  Setting to ``False`` would enable eager loading but
        is not implemented at the moment and will raise ``NotImplementedError``.
    """

    def __init__(
        self,
        root_dir: str,
        metadata_handler: Optional[NSCLCMetadataHandler] = None,
        metadata_csv_path: Optional[str] = None,
        modalities: Optional[Sequence[str]] = ("CT", "PET"),
        pair_modalities: bool = False,
        required_modalities: Optional[Sequence[str]] = None,
        transform: Optional[Callable[..., np.ndarray]] = None,
        volume_loader: Optional[VolumeLoader] = None,
        lazy: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            LOGGER.warning("Root directory %s does not exist; dataset will be empty", self.root_dir)

        if metadata_handler is None and metadata_csv_path:
            metadata_handler = NSCLCMetadataHandler(metadata_csv_path)
        self.metadata_handler = metadata_handler
        if self.metadata_handler and self.metadata_handler.metadata_df is None:
            self.metadata_handler.load_metadata()

        self.modalities = {m.upper() for m in modalities} if modalities else None
        if required_modalities is None:
            required_modalities = ("CT", "PET") if pair_modalities else tuple(self.modalities or [])
        self.required_modalities = {m.upper() for m in required_modalities} if required_modalities else set()
        self.pair_modalities = pair_modalities
        self.transform = transform
        self._volume_loader = volume_loader or self._default_volume_loader
        self.lazy = lazy
        if not lazy:
            raise NotImplementedError("Eager loading is not implemented; use lazy=True")

        self._series_index: List[SeriesSample] = self._build_series_index()
        if self.pair_modalities:
            self._samples = self._build_multimodal_samples(self._series_index)
        else:
            self._samples = self._series_index

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        sample = self._samples[index]
        if self.pair_modalities:
            return self._load_multimodal_sample(sample)
        return self._load_single_sample(sample)

    # ------------------------------------------------------------------
    def _load_single_sample(self, sample: SeriesSample) -> Dict[str, object]:
        volume = self._volume_loader(sample.dicom_files)
        volume = self._apply_transform(volume, sample.metadata)
        modality_key = sample.modality.lower()
        return {
            "patient_id": sample.patient_id,
            "session_id": sample.session_id,
            "modality": modality_key,
            "series_id": sample.series_id,
            modality_key: volume,
            "metadata": sample.metadata,
        }

    # ------------------------------------------------------------------
    def _load_multimodal_sample(self, sample: Dict[str, object]) -> Dict[str, object]:
        modalities = sample["series"]  # type: ignore[index]
        model_input: Dict[str, np.ndarray] = {}
        metadata_output: Dict[str, Dict[str, object]] = {}
        for modality, series_sample in modalities.items():
            series_sample = series_sample  # type: ignore[assignment]
            volume = self._volume_loader(series_sample.dicom_files)
            volume = self._apply_transform(volume, series_sample.metadata)
            modality_key = modality.lower()
            model_input[modality_key] = volume
            metadata_output[modality_key] = series_sample.metadata
        return {
            "patient_id": sample["patient_id"],
            "session_id": sample["session_id"],
            **model_input,
            "metadata": metadata_output,
        }

    # ------------------------------------------------------------------
    def _apply_transform(self, volume: np.ndarray, metadata: Dict[str, object]) -> np.ndarray:
        if self.transform is None:
            return volume
        try:
            return self.transform(volume, metadata)
        except TypeError:
            return self.transform(volume)  # type: ignore[misc]

    # ------------------------------------------------------------------
    def _build_series_index(self) -> List[SeriesSample]:
        if not self.root_dir.exists():
            return []

        patient_ids = self._discover_patients()
        series_index: List[SeriesSample] = []
        for patient_id in patient_ids:
            patient_dir = self.root_dir / patient_id
            if not patient_dir.exists():
                LOGGER.debug("Skipping missing patient directory %s", patient_dir)
                continue

            session_dirs = [p for p in patient_dir.iterdir() if p.is_dir()]
            if not session_dirs:
                LOGGER.debug("Patient %s has no session folders", patient_id)
                continue

            for session_dir in sorted(session_dirs):
                session_id = session_dir.name
                series_dirs = [s for s in session_dir.iterdir() if s.is_dir()]
                if not series_dirs:
                    LOGGER.debug("No series directories under %s", session_dir)
                    continue

                for series_dir in sorted(series_dirs):
                    modality = self._infer_modality(series_dir.name, patient_id)
                    if self.modalities and modality not in self.modalities:
                        continue

                    dicom_files = self._collect_dicom_files(series_dir)
                    if not dicom_files:
                        LOGGER.debug("No DICOM files found in %s", series_dir)
                        continue

                    metadata = self._match_metadata(patient_id, session_id, modality, series_dir.name)
                    series_id = metadata.get("series_uid") or series_dir.name
                    series_index.append(
                        SeriesSample(
                            patient_id=patient_id,
                            session_id=session_id,
                            modality=modality,
                            series_id=str(series_id),
                            dicom_files=dicom_files,
                            metadata=metadata,
                        )
                    )
        return series_index

    # ------------------------------------------------------------------
    def _build_multimodal_samples(self, series_index: Iterable[SeriesSample]) -> List[Dict[str, object]]:
        grouped: Dict[Tuple[str, str], Dict[str, SeriesSample]] = {}
        for sample in series_index:
            key = (sample.patient_id, sample.session_id)
            modality_key = sample.modality.upper()
            grouped.setdefault(key, {})[modality_key] = sample

        multimodal_samples: List[Dict[str, object]] = []
        for (patient_id, session_id), modality_map in grouped.items():
            if not self.required_modalities.issubset(modality_map.keys()):
                continue
            multimodal_samples.append(
                {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "series": {mod: modality_map[mod] for mod in self.required_modalities},
                }
            )
        return multimodal_samples

    # ------------------------------------------------------------------
    def _discover_patients(self) -> List[str]:
        fs_patients = {p.name for p in self.root_dir.iterdir() if p.is_dir()} if self.root_dir.exists() else set()
        if not self.metadata_handler:
            return sorted(fs_patients)

        metadata_patients = set(self.metadata_handler.get_patient_ids())
        # Combine both sources to gracefully handle cases where either side is incomplete.
        return sorted(fs_patients.union(metadata_patients))

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_modality(series_name: str, patient_id: str) -> str:
        name = series_name.upper()
        if "PET" in name:
            return "PET"
        if "CT" in name:
            return "CT"
        if "MR" in name or "MRI" in name:
            return "MR"
        if "PT" == name:
            return "PET"
        LOGGER.debug("Falling back to generic modality for %s/%s", patient_id, series_name)
        return name

    # ------------------------------------------------------------------
    @staticmethod
    def _collect_dicom_files(series_dir: Path) -> List[Path]:
        if not series_dir.exists():
            return []

        direct_files = [p for p in series_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
        dicom_files = [p for p in direct_files if p.suffix.lower() in {".dcm", ".dicom", ""}]

        if not dicom_files:
            # Fallback to a recursive search when the files are nested one level deeper.
            dicom_files = [
                p
                for p in series_dir.rglob("*.dcm")
                if p.is_file() and not p.name.startswith(".")
            ]

        return sorted(dicom_files)

    # ------------------------------------------------------------------
    def _match_metadata(
        self,
        patient_id: str,
        session_id: str,
        modality: str,
        series_name: str,
    ) -> Dict[str, object]:
        if not self.metadata_handler:
            return {
                "patient_id": patient_id,
                "session_id": session_id,
                "modality": modality,
                "series_description": series_name,
            }

        candidates = self.metadata_handler.get_patient_series(patient_id, modality)
        if not candidates:
            return {
                "patient_id": patient_id,
                "session_id": session_id,
                "modality": modality,
                "series_description": series_name,
            }

        normalised_session = session_id.replace("-", "")
        series_name_lower = series_name.lower()
        for candidate in candidates:
            study_date = str(candidate.get("study_date") or "").replace("-", "")
            series_desc = str(candidate.get("series_description") or "").lower()
            if study_date and study_date == normalised_session:
                return candidate
            if series_desc and series_desc == series_name_lower:
                return candidate

        return candidates[0]

    # ------------------------------------------------------------------
    @staticmethod
    def _default_volume_loader(dicom_files: List[Path]) -> np.ndarray:
        if pydicom is None:
            raise ImportError(
                "pydicom is not installed. Provide a custom volume_loader or install pydicom."
            )

        slices = []
        for path in dicom_files:
            try:
                dataset = pydicom.dcmread(str(path), stop_before_pixels=False)
                pixel_array = dataset.pixel_array
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("Failed to read DICOM file %s: %s", path, exc)
                continue

            instance_number = getattr(dataset, "InstanceNumber", None)
            image_position = getattr(dataset, "ImagePositionPatient", None)
            slices.append((instance_number, image_position, pixel_array))

        if not slices:
            raise ValueError("No readable DICOM slices found for series")

        slices.sort(key=_slice_sort_key)
        volume = np.stack([slc[2] for slc in slices]).astype(np.float32)
        return volume


# ----------------------------------------------------------------------
def _slice_sort_key(item: Tuple[Optional[int], Optional[Sequence[float]], np.ndarray]) -> Tuple[float, float]:
    instance_number, image_position, _ = item
    instance_val = float(instance_number) if instance_number is not None else float("inf")
    z_position = float(image_position[2]) if image_position is not None and len(image_position) >= 3 else float("inf")
    return instance_val, z_position


__all__ = ["NestedDICOMDataset", "NSCLCMetadataHandler", "SeriesSample"]
