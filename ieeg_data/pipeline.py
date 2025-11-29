from __future__ import annotations  # allow compatibility for Python 3.9

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import h5py
from brainsets.descriptions import BrainsetDescription, DeviceDescription, SessionDescription, SubjectDescription
from brainsets.pipelines import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Species

from ieeg_data.format import IEEGData

parser = ArgumentParser()
parser.add_argument("--allow_corrupted", action="store_true", help="Allow processing of corrupted data.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed data.")


class IEEGPipeline(BrainsetPipeline, ABC):
    """Abstract base class for iEEG data pipelines.
    Unlike BrainsetPipeline, classes inheriting from iEEGPipeline
    assume that data is already locally available and do not implement downloading.
    Therefore, the `download` method is overridden to simply return the manifest item.

    To be implemented by subclasses:
    - get_manifest: Generate the manifest DataFrame (see BrainsetPipeline for details).
    - populate_data: Populate dataset-specific data and metadata.
    - save_additional (optional): Save any additional files or metadata associated with the dataset.

    The following class variables should be defined in subclasses:
    - brainset_id: str
    - name: str
    - url: str | None
    - version: str
    - citation: str | list[str] | None
    """

    parser = parser

    name: ClassVar[str]
    """Human-readable name of the dataset."""

    url: ClassVar[str | None]
    """URL of the dataset. If the dataset is private or unpublished, set to None."""

    version: ClassVar[str]
    """Version of the dataset."""

    citation: ClassVar[str | list[str] | None]
    """
    Citation for the dataset, in BibTeX format. 
    For multiple citations, use a list of strings.
    If dataset is private or unpublished, set to None.
    """

    def init_data(self, session_id: int, subject_id: int, **fields) -> IEEGData:
        """
        Generate data object skeleton for a given session and subject.

        Args:
            session_id (int): Identifier for the session.
            subject_id (int): Identifier for the subject.
            **fields: Additional fields to populate the data object.

        Returns:
            Data: Initialized data object skeleton.
        """
        brainset = BrainsetDescription(
            id=self.brainset_id,
            origin_version=self.version,
            derived_version=self.version,
            source=self.url,
            description=self.name,
        )

        subject = SubjectDescription(
            id=subject_id,
            species=Species.HUMAN,
        )

        session = SessionDescription(
            id=session_id,
            recording_date=datetime.min,  # TODO: add recording date from data
        )

        device = DeviceDescription(
            id="iEEG/EEG",
            recording_tech=RecordingTech.ECOG_ARRAY_ECOGS,
        )

        data = IEEGData(
            brainset=brainset,
            subject=subject,
            session=session,
            device=device,
            **fields,
            allow_corrupted=self.args.allow_corrupted,
            citation=self.citation,
            domain="auto",
        )

        return data

    def download(self, manifest_item):
        # No downloading needed as data is assumed to be locally available
        return manifest_item

    def process(self, manifest_item):
        session_id = manifest_item.session_id
        subject_id = manifest_item.subject_id

        # Check if already processed
        output_path = self.processed_dir / self.brainset_id / subject_id / session_id / "data.h5"
        if output_path.exists() and not self.args.overwrite:
            self.update_status(
                f"Session {session_id} for subject {subject_id} already processed at"
                f"{output_path}. Skipping. Use --overwrite to reprocess."
            )
            return

        # Process the data
        self.update_status(f"Processing session {session_id} for subject {subject_id}...")
        fields = self.populate_data(manifest_item)
        data = self.init_data(session_id, subject_id, **fields)

        # Save the processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            data.save_to_h5(f)

        self.save_additional(output_path.parent, manifest_item)

        # Log details about the processed data
        n_electrodes = data.ieeg.data.shape[1]
        session_length = data.ieeg.data.shape[0] / data.ieeg.sampling_rate

        n_stim_events = 0
        if hasattr(data, "electrical_stimulation"):
            n_stim_events = data.electrical_stimulation.events.shape[0]

        self.update_status(
            f"Processed session {session_id} for subject {subject_id} at {output_path}:\n"
            f"\t- Number of electrodes: {n_electrodes}\n"
            f"\t- Session length: {session_length:.2f} seconds\n"
            f"\t- Number of stimulation events: {n_stim_events}\n"
        )

    @abstractmethod
    def populate_data(self, manifest_item) -> dict:
        """Populate data dictionary with dataset-specific data and metadata.

        Guidelines:
        - Use 'ieeg' key for intracranial EEG data.
        - Don't use the keys 'brainset', 'subject', 'session', 'device' or any other reserved keys.
        - Ensure that RegularTimeSeries and IrregularTimeSeries objects are used appropriately.

        Returns:
            dict: Dictionary containing dataset-specific data and metadata.
        """
        ...

    def save_additional(self, save_dir: Path, manifest_item) -> None:
        """Save any additional files or metadata associated with the dataset.

        Args:
            save_dir (Path): Directory where the additional files should be saved.
        """
        pass
