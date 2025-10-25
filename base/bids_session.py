from __future__ import annotations  # allow compatibility for Python 3.9

import warnings
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from mne_bids import read_raw_bids
from temporaldata import ArrayDict, RegularTimeSeries

from .session import SessionBase


class BIDSSession(SessionBase):
    """
    This class is used to load the iEEG neural data for a given session from the OpenNeuro BIDS dataset file format as used in OpenNeuro. The dataset is assumed to be stored in the root_dir directory.
    """

    def __init__(
        self,
        subject_identifier: str,
        session_identifier: str,
        root_dir: str | Path | None = None,
        allow_corrupted: bool = False,
    ):
        super().__init__(
            subject_identifier,
            session_identifier,
            root_dir=root_dir,
            allow_corrupted=allow_corrupted,
        )

        self.data_dict["channels"] = self._load_ieeg_electrodes()
        self.data_dict["ieeg"] = self._load_ieeg_data()

    @classmethod
    def discover_subjects(cls, root_dir: str | Path | None = None) -> list:
        if root_dir is None:
            root_dir = cls.find_root_dir()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        participants_file = root_dir / "participants.tsv"
        if not participants_file.exists():
            raise FileNotFoundError(f"participants.tsv not found in {root_dir} (looking for path: {participants_file})")

        participants_df = pd.read_csv(participants_file, sep="\t")
        assert "participant_id" in participants_df.columns, "participants.tsv found but no 'participant_id' column present"
        return participants_df["participant_id"].to_list()

    def _load_ieeg_electrodes(self) -> ArrayDict:
        electrodes_file = self.session["files"]["ieeg_electrodes_file"]
        channels_file = self.session["files"]["ieeg_channels_file"]
        
        electrodes_df = pd.read_csv(electrodes_file, sep="\t")
        channels_df = pd.read_csv(channels_file, sep="\t")

        # Remove any rows that contain NaN values (usually meaning non-iEEG channels)
        electrodes_df = electrodes_df.dropna()

        # Filter channels to only include ECOG or SEEG types and good channels if not allowing corrupted data
        if "type" in channels_df.columns:
            channels_df = channels_df[channels_df["type"].str.upper().isin(["ECOG", "SEEG"])]
        if ("status" in channels_df.columns) and (not self.allow_corrupted):
            channels_df = channels_df[channels_df["status"].str.upper().isin(["GOOD"])]

        # Merge electrode coordinates into channels dataframe
        # For each channel, find the corresponding electrode and copy x, y, z coordinates
        channels_df = channels_df[["name", "type"]].merge(electrodes_df[["name", "x", "y", "z"]], on="name", how="left")

        electrodes = ArrayDict(
            id=channels_df["name"].array.astype(str),
            x=channels_df["x"].array.astype(float),
            y=channels_df["y"].array.astype(float),
            z=channels_df["z"].array.astype(float),
            brain_area=np.array(["UNKNOWN"] * len(channels_df)),  # TODO: add brain area
            type=channels_df["type"].array.astype(str),
        )
        return electrodes

    def _load_ieeg_data(self, suppress_warnings: bool = True):
        ieeg_file = self.session["files"]["ieeg_file"]
        
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No BIDS -> MNE mapping found")
                warnings.filterwarnings("ignore", message="Unable to map the following column")
                warnings.filterwarnings("ignore", message="Not setting positions")
                warnings.filterwarnings("ignore", message="DigMontage is only a subset of info.")
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne_bids")
                raw = read_raw_bids(ieeg_file, verbose=False)
        else:
            raw = read_raw_bids(ieeg_file, verbose=True)

        raw = raw.pick(self.data_dict["channels"].id.tolist())  # type: ignore[attr-defined]

        return RegularTimeSeries(
            data=raw.get_data().astype(np.float32).T * 1e6,  # shape should be (n_samples, n_channels), and convert to microvolts
            sampling_rate=int(raw.info["sfreq"]),
            domain_start=0.0,  # Start of the domain (in seconds)
            domain="auto",  # Automatically determine the domain based on the data # type:ignore
        )

