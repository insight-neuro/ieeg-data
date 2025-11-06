from __future__ import annotations  # allow compatibility for Python 3.9

import json
import logging
import os, glob
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, RegularTimeSeries, IrregularTimeSeries

from neo.io import BlackrockIO

# Handle both relative imports (when used as module) and absolute imports (when run directly)
try:
    from ..base import SessionBase
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from base import SessionBase

logger = logging.getLogger(__name__)


class BlackrockSession(SessionBase):
    """
    This class is used to load the iEEG neural data in the Blackrock format.
    """

    name = "BIDMC Neurodynamics"
    dataset_identifier = "bi_blackrock_neurodynamics_2025"
    dataset_version = "0.1.0"
    url = "" # private dataset
    citation = "" # private dataset

    def __init__(self, subject_identifier, session_identifier, root_dir=None, allow_corrupted=False):
        super().__init__(subject_identifier, session_identifier, root_dir=root_dir, allow_corrupted=allow_corrupted)

        self.data_dict["channels"] = self._load_ieeg_electrodes()
        self.data_dict["ieeg"] = self._load_ieeg_data()

        if "images_file" in self.session["files"]:
            self.data_dict["images"] = self._load_images()
        if "triggers_file" in self.session["files"]:
            self.data_dict["triggers"] = self._load_triggers()

    @classmethod
    def discover_subjects(cls, root_dir: str | Path | None = None) -> list:
        if root_dir is None:
            root_dir = cls.find_root_dir()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        # Find all subject directories in the root directory
        return [d.name for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]

    @classmethod
    def discover_sessions(cls, subject_identifier: str, root_dir: str | Path | None = None) -> list:
        if root_dir is None:
            root_dir = cls.find_root_dir()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        # Find all session directories in the subject directory
        subject_dir = root_dir / subject_identifier
        session_identifiers = [d.name for d in subject_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]

        all_sessions = []
        for session_identifier in session_identifiers:
            session_dir = subject_dir / session_identifier
            assert (session_dir / f"{session_identifier}-001.nev").exists(), f"NEV file not found: {session_dir / f"{session_identifier}-001.nev"}"
            assert (session_dir / "electrodes.h5").exists(), f"Electrode file not found: {session_dir / "electrodes.h5"}"

            session_files = {
                "ieeg_file": session_dir / f"{session_identifier}-001.nev",
                "electrodes_file": session_dir / "electrodes.h5"
            }
            if (session_dir / "images.h5").exists():
                session_files["images_file"] = session_dir / "images.h5"
            if (session_dir / "triggers.h5").exists():
                session_files["triggers_file"] = session_dir / "triggers.h5"
            all_sessions.append(
                {
                    "session_identifier": session_identifier,
                    "files": session_files,
                }
            )
        return all_sessions

    def _load_ieeg_electrodes(self):
        with h5py.File(self.session["files"]["electrodes_file"], "r") as f:
            electrodes = ArrayDict.from_hdf5(f)
        return electrodes

    def _load_ieeg_data(self):
        io = BlackrockIO(filename=self.session["files"]["ieeg_file"])
        io.parse_header()

        sampling_rates = io.header['signal_channels']['sampling_rate']
        # Assert all sampling rates are the same
        assert np.all(sampling_rates == sampling_rates[0]), f"Not all channels have the same sampling rate. Found rates: {np.unique(sampling_rates)}"
        sampling_rate = sampling_rates[0]

        channel_indices = list(range(len(self.data_dict["channels"].id)))
        raw_analog_signal = io.get_analogsignal_chunk(block_index=0, seg_index=0, channel_indexes=channel_indices)    # shape: (n_samples, n_channels)
        raw_analog_signal = io.rescale_signal_raw_to_float(raw_analog_signal, dtype='float32', channel_indexes=channel_indices)     # shape: (n_samples, n_channels). in uV

        return RegularTimeSeries(
            data=raw_analog_signal,
            sampling_rate=sampling_rate,
            domain_start=0.0,
            domain="auto",
        )
    
    def _load_images(self):
        with h5py.File(self.session["files"]["images_file"], "r") as f:
            images = IrregularTimeSeries.from_hdf5(f)
        return images

    def _load_triggers(self):
        with h5py.File(self.session["files"]["triggers_file"], "r") as f:
            triggers = IrregularTimeSeries.from_hdf5(f)
        return triggers

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    
    save_root_dir = os.getenv("PROCESSED_DATA_DIR")
    if save_root_dir is None:
        raise ValueError("PROCESSED_DATA_DIR environment variable not set.")

    BlackrockSession.save_all_subjects_sessions(root_dir=None, save_root_dir=save_root_dir)