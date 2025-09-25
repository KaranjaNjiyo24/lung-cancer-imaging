"""Robust DICOM pre-processing utilities for multimodal lung cancer imaging."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
import torch
import torch.nn.functional as F


class DICOMProcessor:
    """Load and preprocess DICOM series from nested directory structures."""

    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.target_spacing = tuple(float(x) for x in target_spacing)
        self._context: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def load_dicom_series(self, series_path: str) -> Optional[np.ndarray]:
        """Load an entire DICOM series from ``series_path``.

        The method walks nested folders, handles corrupted slices, standardises
        slice ordering/orientation, rescales pixel values using DICOM metadata
        and resamples the volume to ``self.target_spacing``.
        """

        series_dir = Path(series_path)
        if not series_dir.exists():
            warnings.warn(f"Series directory not found: {series_dir}")
            return None

        dicom_files = self._collect_dicom_files(series_dir)
        if not dicom_files:
            warnings.warn(f"No DICOM files discovered under {series_dir}")
            return None

        slices: List[Tuple[Optional[int], Optional[float], np.ndarray]] = []
        first_ds: Optional[pydicom.Dataset] = None
        for file_path in dicom_files:
            try:
                dataset = pydicom.dcmread(str(file_path), stop_before_pixels=False)
            except (InvalidDicomError, OSError) as exc:
                warnings.warn(f"Skipping unreadable DICOM file {file_path}: {exc}")
                continue

            if not hasattr(dataset, "PixelData"):
                continue

            if first_ds is None:
                first_ds = dataset

            try:
                pixel_array = dataset.pixel_array.astype(np.float32)
            except Exception as exc:  # pragma: no cover - defensive guard
                warnings.warn(f"Failed to extract pixel data from {file_path}: {exc}")
                continue

            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            if slope != 1.0 or intercept != 0.0:
                pixel_array = pixel_array * slope + intercept

            instance_number = getattr(dataset, "InstanceNumber", None)
            position = getattr(dataset, "ImagePositionPatient", None)
            z_pos = float(position[2]) if position is not None and len(position) >= 3 else None
            slices.append((instance_number, z_pos, pixel_array))

        if not slices or first_ds is None:
            warnings.warn(f"Unable to assemble DICOM volume from {series_dir}")
            return None

        # Order slices and standardise orientation.
        slices.sort(key=lambda x: (
            float(x[0]) if x[0] is not None else float("inf"),
            float(x[1]) if x[1] is not None else float("inf"),
        ))
        volume = np.stack([slc[2] for slc in slices]).astype(np.float32)

        z_positions = [slc[1] for slc in slices if slc[1] is not None]
        if len(z_positions) >= 2 and z_positions[1] < z_positions[0]:
            volume = volume[::-1]
            z_positions = list(reversed(z_positions))

        orientation = getattr(first_ds, "ImageOrientationPatient", None)
        volume = self._standardise_orientation(volume, orientation)

        spacing = self._extract_spacing(first_ds, z_positions)
        context = self._extract_series_context(first_ds)
        context["spacing"] = spacing
        context["orientation"] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        self._context = context

        volume = self._resample_to_spacing(volume, spacing, self.target_spacing)
        self._context["spacing"] = self.target_spacing

        return volume.astype(np.float32)

    # ------------------------------------------------------------------
    def preprocess_ct(self, volume: np.ndarray) -> np.ndarray:
        """Apply CT-specific preprocessing (HU clipping/normalisation)."""

        if volume is None:
            raise ValueError("CT volume is None")

        # Clip to lung window and scale to [0, 1]
        clipped = np.clip(volume, -1024.0, 400.0)
        normalised = (clipped + 1024.0) / 1424.0
        normalised = np.clip(normalised, 0.0, 1.0)
        return normalised.astype(np.float32)

    # ------------------------------------------------------------------
    def preprocess_pet(self, volume: np.ndarray) -> np.ndarray:
        """Apply PET-specific preprocessing (SUV scaling and normalisation)."""

        if volume is None:
            raise ValueError("PET volume is None")

        suv_factor = self._context.get("suv_factor") if self._context else None
        scaled = volume.astype(np.float32)
        if suv_factor and suv_factor > 0:
            scaled = scaled * float(suv_factor)

        # Robust intensity scaling: clip high outliers and scale to [0, 1].
        upper = np.percentile(scaled, 99.5)
        if upper <= 0:
            upper = np.max(scaled)
        if upper > 0:
            scaled = np.clip(scaled, 0.0, upper) / upper
        return scaled.astype(np.float32)

    # ------------------------------------------------------------------
    def resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize ``volume`` (D, H, W) to ``target_size`` using trilinear interpolation."""

        if volume.ndim != 3:
            raise ValueError("Expected a 3D volume for resizing")

        target_size = tuple(int(max(1, s)) for s in target_size)
        if tuple(volume.shape) == target_size:
            return volume.astype(np.float32)

        tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=target_size, mode="trilinear", align_corners=False)
        return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)

    # ------------------------------------------------------------------
    def _collect_dicom_files(self, series_dir: Path) -> List[Path]:
        files = [p for p in series_dir.rglob("*") if p.is_file()]
        dicom_like = [p for p in files if p.suffix.lower() in {".dcm", ""} or "DICOM" in p.suffix.upper()]
        return sorted(dicom_like)

    # ------------------------------------------------------------------
    def _extract_spacing(
        self,
        dataset: pydicom.Dataset,
        z_positions: List[float],
    ) -> Tuple[float, float, float]:
        pixel_spacing = getattr(dataset, "PixelSpacing", [1.0, 1.0])
        try:
            spacing_y = float(pixel_spacing[0])
            spacing_x = float(pixel_spacing[1])
        except (TypeError, ValueError, IndexError):
            spacing_y = spacing_x = 1.0

        if len(z_positions) >= 2:
            diffs = np.diff(sorted(z_positions))
            diffs = diffs[np.abs(diffs) > 1e-6]
            spacing_z = float(np.median(np.abs(diffs))) if diffs.size else 0.0
        else:
            spacing_z = 0.0

        if spacing_z <= 0.0:
            spacing_z = float(getattr(dataset, "SpacingBetweenSlices", 0.0))
        if spacing_z <= 0.0:
            spacing_z = float(getattr(dataset, "SliceThickness", 1.0))
        if spacing_z <= 0.0:
            spacing_z = 1.0

        return (spacing_z, spacing_y, spacing_x)

    # ------------------------------------------------------------------
    def _extract_series_context(self, dataset: pydicom.Dataset) -> Dict[str, Any]:
        modality = str(getattr(dataset, "Modality", "" )).upper()
        manufacturer = str(getattr(dataset, "Manufacturer", "Unknown"))
        context: Dict[str, Any] = {
            "modality": modality,
            "manufacturer": manufacturer,
            "series_uid": getattr(dataset, "SeriesInstanceUID", None),
            "study_uid": getattr(dataset, "StudyInstanceUID", None),
            "suv_factor": None,
        }

        if modality in {"PT", "PET"}:
            context["suv_factor"] = self._estimate_suv_factor(dataset)
        return context

    # ------------------------------------------------------------------
    def _estimate_suv_factor(self, dataset: pydicom.Dataset) -> Optional[float]:
        units = str(getattr(dataset, "Units", "")).upper()
        if units not in {"BQML", "CNTS"}:
            return None

        patient_weight = float(getattr(dataset, "PatientWeight", 0.0))
        sequence = getattr(dataset, "RadiopharmaceuticalInformationSequence", None)
        if sequence:
            info = sequence[0]
            patient_weight = float(getattr(info, "PatientWeight", patient_weight))
            dose = float(getattr(info, "RadionuclideTotalDose", 0.0))
            start_time = getattr(info, "RadiopharmaceuticalStartTime", None)
            half_life = float(getattr(info, "RadionuclideHalfLife", 0.0))
        else:
            dose = float(getattr(dataset, "RadionuclideTotalDose", 0.0))
            start_time = getattr(dataset, "RadiopharmaceuticalStartTime", None)
            half_life = float(getattr(dataset, "RadionuclideHalfLife", 0.0))

        if dose <= 0 or patient_weight <= 0:
            return None

        # Decay correction if acquisition time is available.
        acquisition_time = getattr(dataset, "AcquisitionTime", None)
        if units == "BQML" and start_time and acquisition_time and half_life > 0:
            elapsed = self._time_difference_seconds(start_time, acquisition_time)
            if elapsed > 0:
                dose *= math.exp(-math.log(2) * elapsed / half_life)

        if dose <= 0:
            return None

        return patient_weight / dose

    # ------------------------------------------------------------------
    @staticmethod
    def _time_difference_seconds(start: str, end: str) -> float:
        def _to_seconds(value: str) -> float:
            value = value.replace(":", "")
            if len(value) < 6:
                value = value.ljust(6, "0")
            hours = int(value[0:2])
            minutes = int(value[2:4])
            seconds = float(value[4:])
            return hours * 3600 + minutes * 60 + seconds

        try:
            start_sec = _to_seconds(start)
            end_sec = _to_seconds(end)
            delta = end_sec - start_sec
            if delta < 0:
                delta += 24 * 3600
            return float(delta)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def _standardise_orientation(self, volume: np.ndarray, orientation: Optional[List[float]]) -> np.ndarray:
        if orientation is None or len(orientation) < 6:
            return volume

        row = np.array(orientation[:3], dtype=np.float32)
        col = np.array(orientation[3:], dtype=np.float32)
        slice_dir = np.cross(row, col)

        adjusted = volume
        if row[0] < 0:
            adjusted = adjusted[:, :, ::-1]
        if col[1] < 0:
            adjusted = adjusted[:, ::-1, :]
        if slice_dir[2] < 0:
            adjusted = adjusted[::-1, :, :]
        return adjusted

    # ------------------------------------------------------------------
    def _resample_to_spacing(
        self,
        volume: np.ndarray,
        current_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float],
    ) -> np.ndarray:
        current = np.array(current_spacing, dtype=np.float32)
        target = np.array(target_spacing, dtype=np.float32)
        if np.allclose(current, target, rtol=1e-2, atol=1e-2):
            return volume.astype(np.float32)

        scale = current / target
        new_shape = tuple(int(max(1, round(dim * scale[idx]))) for idx, dim in enumerate(volume.shape))
        return self.resize_volume(volume, new_shape)


__all__ = ["DICOMProcessor"]
