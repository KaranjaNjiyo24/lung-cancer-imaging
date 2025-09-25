"""Utilities for parsing NSCLC radiogenomics metadata.

This module exposes :class:`NSCLCMetadataHandler`, a small helper that keeps
track of which imaging modalities are available for each patient as described
in the ``metadata.csv`` file distributed with the dataset.  The handler is
light-weight and keeps only the processed metadata structures in memory so it
can comfortably scale to the ~100 GB dataset described in the project brief.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


class NSCLCMetadataHandler:
    """Parse and query the NSCLC radiogenomics metadata CSV file.

    Parameters
    ----------
    metadata_csv_path:
        Path to the ``metadata.csv`` file.
    """

    REQUIRED_COLUMNS = {"Subject ID", "Modality"}

    def __init__(self, metadata_csv_path: str):
        self.metadata_csv_path = Path(metadata_csv_path)
        self.metadata_df: Optional[pd.DataFrame] = None
        self.patient_mapping: Dict[str, Dict[str, object]] = {}

    # ---------------------------------------------------------------------
    def load_metadata(self) -> pd.DataFrame:
        """Load and normalise the metadata CSV.

        Returns
        -------
        pandas.DataFrame
            A copy of the processed metadata dataframe.  Returns an empty
            dataframe if the CSV cannot be found or is empty.
        """

        csv_path = self.metadata_csv_path
        if not csv_path.exists():
            LOGGER.warning("Metadata CSV not found at %s", csv_path)
            self.metadata_df = pd.DataFrame()
            self.patient_mapping = {}
            return self.metadata_df

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.error("Failed to read metadata CSV %s: %s", csv_path, exc)
            self.metadata_df = pd.DataFrame()
            self.patient_mapping = {}
            return self.metadata_df

        if df.empty:
            LOGGER.warning("Metadata CSV %s is empty", csv_path)
            self.metadata_df = pd.DataFrame()
            self.patient_mapping = {}
            return self.metadata_df

        # Normalise column names and ensure required ones are present.
        df.columns = [str(col).strip() for col in df.columns]
        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            LOGGER.warning("Metadata CSV missing expected columns: %s", ", ".join(sorted(missing)))
            for column in missing:
                df[column] = pd.NA

        # Clean up fields we rely on downstream.
        df["Subject ID"] = df["Subject ID"].fillna("").astype(str).str.strip()
        if "Modality" in df.columns:
            df["Modality"] = df["Modality"].fillna("").astype(str).str.upper().str.strip()
        if "Study Date" in df.columns:
            df["Study Date"] = df["Study Date"].fillna("").astype(str).str.strip()
        if "Series Description" in df.columns:
            df["Series Description"] = df["Series Description"].fillna("").astype(str).str.strip()
        if "File Location" in df.columns:
            df["File Location"] = df["File Location"].fillna("").astype(str).str.strip()

        # Drop rows without a patient identifier.
        df = df[df["Subject ID"] != ""].copy()

        self.metadata_df = df.reset_index(drop=True)
        self.patient_mapping = self._build_patient_mapping(self.metadata_df)
        return self.metadata_df.copy()

    # ------------------------------------------------------------------
    def _build_patient_mapping(self, df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        """Create per-patient lookup structures used by downstream code."""

        mapping: Dict[str, Dict[str, object]] = {}
        for record in df.to_dict(orient="records"):
            patient_id = record.get("Subject ID") or ""
            modality = (record.get("Modality") or "").upper() or "UNKNOWN"

            patient_entry = mapping.setdefault(
                patient_id,
                {
                    "modalities": set(),
                    "series": [],
                    "files_by_modality": defaultdict(list),
                },
            )
            if modality != "":
                patient_entry["modalities"].add(modality)

            series_info = {
                "series_uid": record.get("Series UID"),
                "collection": record.get("Collection"),
                "study_uid": record.get("Study UID"),
                "study_description": record.get("Study Description"),
                "study_date": record.get("Study Date"),
                "series_description": record.get("Series Description"),
                "manufacturer": record.get("Manufacturer"),
                "modality": modality,
                "sop_class_name": record.get("SOP Class Name"),
                "number_of_images": record.get("Number of Images"),
                "file_size": record.get("File Size"),
                "file_location": record.get("File Location"),
                "download_timestamp": record.get("Download Timestamp"),
            }
            patient_entry["series"].append(series_info)
            if series_info["file_location"]:
                patient_entry["files_by_modality"][modality].append(series_info["file_location"])

        # Convert modality sets into sorted lists for deterministic ordering.
        for entry in mapping.values():
            entry["modalities"] = sorted(entry["modalities"])  # type: ignore[assignment]
            entry["files_by_modality"] = {
                mod: sorted(paths) for mod, paths in entry["files_by_modality"].items()
            }

        return mapping

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self.metadata_df is None:
            self.load_metadata()

    # ------------------------------------------------------------------
    def get_patient_ids(self) -> List[str]:
        """Return all patient identifiers present in the metadata."""

        self._ensure_loaded()
        return sorted(self.patient_mapping.keys())

    # ------------------------------------------------------------------
    def get_patient_modalities(self, patient_id: str) -> List[str]:
        """Return the list of modalities available for the given patient."""

        self._ensure_loaded()
        entry = self.patient_mapping.get(patient_id, {})
        return list(entry.get("modalities", [])) if entry else []

    # ------------------------------------------------------------------
    def get_patient_series(
        self, patient_id: str, modality: Optional[str] = None
    ) -> List[Dict[str, object]]:
        """Return the metadata records for the requested patient/modality."""

        self._ensure_loaded()
        entry = self.patient_mapping.get(patient_id)
        if not entry:
            return []

        series: Iterable[Dict[str, object]] = entry.get("series", [])  # type: ignore[assignment]
        if modality:
            modality = modality.upper()
            series = [s for s in series if (s.get("modality") or "").upper() == modality]
        return list(series)

    # ------------------------------------------------------------------
    def get_patient_files(self, patient_id: str, modality: Optional[str] = None) -> List[str]:
        """Return file locations associated with a patient, optionally filtered by modality."""

        self._ensure_loaded()
        entry = self.patient_mapping.get(patient_id)
        if not entry:
            return []

        files_by_modality: Dict[str, List[str]] = entry.get("files_by_modality", {})  # type: ignore[assignment]
        if modality is None:
            # Flatten all modality-specific lists.
            result: List[str] = []
            for paths in files_by_modality.values():
                result.extend(paths)
            return result

        modality = modality.upper()
        return list(files_by_modality.get(modality, []))

    # ------------------------------------------------------------------
    def get_multimodal_patients(self) -> List[str]:
        """Return patients that have both CT and PET modalities available."""

        self._ensure_loaded()
        multimodal = []
        for patient_id, entry in self.patient_mapping.items():
            modalities = {mod.upper() for mod in entry.get("modalities", [])}
            if {"CT", "PET"}.issubset(modalities):
                multimodal.append(patient_id)
        return sorted(multimodal)

    # ------------------------------------------------------------------
    def filter_by_modality(self, modality: str) -> pd.DataFrame:
        """Return a filtered dataframe containing only the requested modality."""

        self._ensure_loaded()
        if self.metadata_df is None or self.metadata_df.empty:
            return pd.DataFrame()
        modality = modality.upper()
        mask = self.metadata_df.get("Modality", pd.Series([], dtype="object")).str.upper() == modality
        return self.metadata_df.loc[mask].copy()


__all__ = ["NSCLCMetadataHandler"]
