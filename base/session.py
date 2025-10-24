from __future__ import annotations  # allow compatibility for Python 3.9

import datetime
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import pandas as pd
# from brainsets.descriptions import BrainsetDescription, DeviceDescription, SessionDescription, SubjectDescription
# from brainsets.taxonomy import RecordingTech, Species
from temporaldata import Data

logger = logging.getLogger(__name__)


class SessionBase(ABC):
    """
    This class is an interface used to load the iEEG neural data for a given session. The dataset is assumed to be stored in the root_dir directory. This class must be used as a parent class for all session classes.
    """

    # NOTE: Every subclass must define these variables
    dataset_identifier: ClassVar[str]  # Follow the brainsets convention for naming: firstAuthorLastName_lastAuthorLastName_firstWordOfPublicationTitle_publicationYear (all lowercase letters)
    dataset_version: ClassVar[str]  # Version of the dataset.
    name: ClassVar[str]
    url: ClassVar[str | None]  # If empty, it will be assumed that the dataset is private
    citation: ClassVar[str | None]  # In BibTex format. If empty, it will be assumed that the dataset is private. This can be multiple citations, separated by a newline.

    def __init__(
        self,
        subject_identifier: str,
        session_identifier: str,
        root_dir: str | Path | None = None,
        allow_corrupted: bool = False,
    ):
        # Check if the root_dir is set in the environment variables
        if root_dir is None:
            root_dir = self.find_root_dir()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.subject_identifier = subject_identifier
        self.session_identifier = session_identifier
        self.allow_corrupted = allow_corrupted

        self.data_dict = {
            # Commented out while fixing the dependency issue with brainsets
            # "brainset": BrainsetDescription(
            #     id=self.dataset_identifier,
            #     origin_version=self.dataset_version,
            #     derived_version=self.dataset_version,
            #     source=self.url,
            #     description=self.name,
            # ),
            # "subject": SubjectDescription(
            #     id=self.subject_identifier,
            #     species=Species.HUMAN,
            # ),
            # "session": SessionDescription(
            #     id=self.session_identifier,
            #     recording_date=datetime.datetime.min,  # TODO: add recording date somehow from the data
            # ),
            # "device": DeviceDescription(
            #     id="iEEG/EEG",
            #     recording_tech=RecordingTech.ECOG_ARRAY_ECOGS,
            # ),
            "brainset": self.dataset_identifier,
            "subject": self.subject_identifier,
            "session": self.session_identifier,

            "allow_corrupted": self.allow_corrupted,
            "citation": self.citation,
        }

        # Discover subjects and ensure the subject identifier exists in the dataset
        self.all_subjects = self.__class__.discover_subjects(self.root_dir)
        assert self.subject_identifier in self.all_subjects, f"Subject {self.subject_identifier} not found in dataset. List of subjects: {self.all_subjects}"
        self.subject_dir = self.root_dir / self.subject_identifier

        # Discover sessions and ensure the session identifier exists in the dataset
        self.all_sessions = self.__class__.discover_sessions(self.subject_identifier, root_dir=self.root_dir)
        all_session_identifiers = [session["session_identifier"] for session in self.all_sessions]
        assert self.session_identifier in all_session_identifiers, f"Session {self.session_identifier} not found in {self.all_sessions}"
        self.session = self.all_sessions[all_session_identifiers.index(self.session_identifier)]

    @classmethod
    def find_root_dir(cls) -> str:
        """
        Find the root directory of the dataset in the environment variables.
        """
        try:
            return os.environ["ROOT_DIR_" + cls.dataset_identifier.upper()]
        except KeyError:
            raise ValueError(f"When loading dataset {cls.dataset_identifier}, ROOT_DIR_{cls.dataset_identifier.upper()} not set in environment variables. Please either set the ROOT_DIR_{cls.dataset_identifier.upper()} environment variable or pass the root_dir argument to the constructor.") from None

    @classmethod
    @abstractmethod
    def discover_subjects(cls, root_dir: str | Path | None = None) -> list:
        """
        Discover all subjects in the dataset.

        Args:
            root_dir (str | Path | None): The root directory of the dataset. If not provided, the root directory will be found in the environment variables.

        Returns:
            list: List of subject identifiers.
        """
        raise NotImplementedError("Not implemented")

    @classmethod
    @abstractmethod
    def discover_sessions(cls, subject_identifier: str, root_dir: str | Path | None = None):
        """
        Discover available sessions for the subject subject_identifier in the root_dir. This is a static method that can be used to discover sessions for any subject in the dataset.

        Args:
            subject_identifier (str): The identifier of the subject to discover sessions for.
            root_dir (str | Path | None): The root directory of the dataset. If not provided, the root directory will be found in the environment variables.

        Returns:
            list: List of dictionaries containing:
                - session_identifier: the session identifier (e.g., "031411")
                - files: a dictionary with arbitrary keys for file paths. The specific keys depend on the dataset format.
                    For example, for BIDS datasets, this typically includes:
                        - ieeg_file: path to the file containing the iEEG data (BIDSPath object)
                        - ieeg_electrodes_file: path to the electrodes file with coordinates
                        - ieeg_channels_file: path to the channels file with labels and metadata
                        - events_file: path to the events file (optional)
        """
        raise NotImplementedError("Not implemented")

    def get_data(self):
        """
        Get the data for the session in the temporaldata format.

        Returns:
            data: The data for the session in the format of a temporaldata.Data object.
        """
        return Data(**self.data_dict, domain="auto")

    def save_data(self, save_root_dir: str | Path):
        """
        Save the data for the session in the temporaldata format.

        Args:
            save_root_dir (str | Path): The root directory to save the data to.

        Returns:
            path,data (tuple[str | Path, temporaldata.Data]): Tuple containing:
                path: The path to the saved data.
                data: The data for the session in the format of a temporaldata.Data object.
        """
        path = Path(save_root_dir) / self.dataset_identifier / self.subject_identifier / self.session_identifier
        data = self.get_data()

        if path.exists():
            logger.info(f"Data for subject {self.subject_identifier} and session {self.session_identifier} already exists at {path}. Skipping.")
            return path, data

        path.mkdir(parents=True, exist_ok=True)

        # Save to HDF5
        with h5py.File(path / "data.h5", "w") as f:
            data.to_hdf5(f)

        logger.info(f"Saved data for subject {self.subject_identifier} and session {self.session_identifier} to {path}")
        return path, data

    @classmethod
    def save_all_subjects_sessions(cls, root_dir: str | Path | None, save_root_dir: str | Path):
        """Save all subjects and sessions to the specified directory.

        Args:
            root_dir (str | Path | None): Root directory of the dataset. If None, will be found in environment variables.
            save_root_dir (str | Path): Root directory to save the processed data.
        """
        for subject_identifier in cls.discover_subjects(root_dir=root_dir):
            for session in cls.discover_sessions(subject_identifier=subject_identifier, root_dir=root_dir):
                session_identifier = session["session_identifier"]
                if (Path(save_root_dir) / cls.dataset_identifier / subject_identifier / session_identifier / "data.h5").exists():
                    print(f"Data for {cls.dataset_identifier}/{subject_identifier}/{session_identifier} already exists. Skipping.")
                    continue

                session = cls(
                    subject_identifier=subject_identifier,
                    session_identifier=session_identifier,
                    root_dir=root_dir,
                    allow_corrupted=False,
                )
                path, data = session.save_data(save_root_dir=save_root_dir)

                session_length = data.ieeg.data.shape[0] / data.ieeg.sampling_rate
                n_electrodes = data.ieeg.data.shape[1]
                if "electrical_stimulation" in data.keys():
                    n_stim_events = data.electrical_stimulation.timestamps.shape[0]
                else:
                    n_stim_events = 0
                print(f"Saved data: {path}")
                print(f"\t\tSession length: {session_length:.2f} seconds\t\t{n_electrodes} electrodes\t\t{n_stim_events} stimulation events")

    def _load_ieeg_electrodes(self, electrodes_file: str | Path, channels_file: str | Path):
        """
        This is an optional function to implement (only if the session contains ieeg data). Load the electrodes from the electrodes file.

        Args:
            electrodes_file (str | Path): The path to the electrodes file.
            channels_file (str | Path): The path to the channels file.

        Returns:
            electrodes (temporaldata.ArrayDict): The electrodes in the format of a temporaldata.ArrayDict. The labels are the channel names, the coordinates are the x, y, z coordinates of the electrodes, and the types are the types of the channels.
        """
        raise NotImplementedError("Not implemented")

    def _load_ieeg_data(self, ieeg_file: str | Path, suppress_warnings: bool = True):
        """
        This is an optional function to implement (only if the session contains ieeg data). Load the iEEG data from the ieeg file.

        Args:
            ieeg_file (str | Path): The path to the iEEG file.
            suppress_warnings (bool): Whether to suppress warnings when loading the iEEG data. Default is True.

        Returns:
            ieeg_data (temporaldata.RegularTimeSeries): The iEEG data in the format of a temporaldata.RegularTimeSeries. The data is a numpy array of shape (n_channels, n_samples), the sampling rate is the sampling rate of the data (int, in Hz), the domain start is 0, and the domain is automatically determined based on the data.
        """
        raise NotImplementedError("Not implemented")

    def _load_electrical_stimulation(self):
        """
        This is an optional function to implement (only if the session contains electrical stimulation data). Load the electrical stimulation data from the electrical stimulation file.

        Returns:
            electrical_stimulation (temporaldata.IrregularTimeSeries): The electrical stimulation data in the format of a temporaldata.IrregularTimeSeries. The timestamps are the timestamps of the electrical stimulation, the stimulation_site are the sites of the electrical stimulation, the duration are the duration of the electrical stimulation, the waveform_type are the types of the electrical stimulation, the current are the currents of the electrical stimulation, the frequency are the frequencies of the electrical stimulation, the pulse_width are the pulse widths of the electrical stimulation, and the domain is automatically determined based on the data.
        """
        raise NotImplementedError("Not implemented")
