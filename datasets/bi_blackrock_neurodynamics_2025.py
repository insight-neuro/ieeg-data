from __future__ import annotations  # allow compatibility for Python 3.9

import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, RegularTimeSeries

from neo.io import BlackrockIO
from .base import SessionBase

logger = logging.getLogger(__name__)


class BlackrockSession(SessionBase):
    """
    This class is used to load the iEEG neural data in the Blackrock format.
    """

    def __init__(self, subject_identifier, session_identifier, root_dir=None, allow_corrupted=False):
        super().__init__(subject_identifier, session_identifier, root_dir=root_dir, allow_corrupted=allow_corrupted)

        self.data_dict["channels"] = self._load_ieeg_electrodes(self.session["ieeg_electrodes_file"], self.session["ieeg_channels_file"])
        self.data_dict["ieeg"] = self._load_ieeg_data(self.session["ieeg_file"])

    @classmethod
    def discover_subjects(cls, root_dir: str | Path | None = None) -> list:
        if root_dir is None:
            root_dir = cls.find_root_dir()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        # Find all subject directories in the root directory
        return [d.name for d in root_dir.iterdir() if d.is_dir()]

    @classmethod
    def discover_sessions(cls, subject_identifier: str, root_dir: str | Path | None = None) -> list:
        if root_dir is None:
            root_dir = cls.find_root_dir()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        # Find all session directories in the subject directory
        subject_dir = root_dir / subject_identifier
        session_identifiers = [d.name for d in subject_dir.iterdir() if d.is_dir()]

        return [
            {
                "session_identifier": session_identifier,
                "events_file": None,  # TODO: add the events later
                "ieeg_file": subject_dir / session_identifier / "ieeg.mat",
                "ieeg_electrodes_file": subject_dir / session_identifier / "electrodes.mat",
                "ieeg_channels_file": subject_dir / session_identifier / "channels.mat",
            }
            for session_identifier in session_identifiers
        ]

    
    # eturns:
    #         list: List of dictionaries containing:
    #             - session_identifier: the session identifier (e.g., "031411")
    #             - events_file: path to the file containing the events
    #             - ieeg_file: path to the file containing the iEEG data. Must be a BIDSPath object.
    #             - ieeg_electrodes_file: path to the file containing the electrodes. This file will contain the coordinates of each electrode.
    #             - ieeg_channels_file: path to the file containing the channels. This file will contain the labels of each channel and session-specific metadata (good vs bad channels, etc.)
        