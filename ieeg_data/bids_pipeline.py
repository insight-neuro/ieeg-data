import warnings
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
from mne_bids import read_raw_bids
from temporaldata import ArrayDict, RegularTimeSeries

from ieeg_data.pipeline import IEEGPipeline


class BIDSPipeline(IEEGPipeline, ABC):
    """
    This class is used to load the iEEG neural data for a given session from the OpenNeuro BIDS dataset file format as used in OpenNeuro.
    """

    @classmethod
    def discover_subjects(cls, raw_dir: Path) -> list[str]:
        """
        Discover all subjects in the BIDS dataset located at raw_dir
        """

        participants_file = raw_dir / "participants.tsv"
        if not participants_file.exists():
            raise FileNotFoundError(f"participants.tsv not found in {raw_dir} (looking for path: {participants_file})")

        participants_df = pd.read_csv(participants_file, sep="\t")

        if "participant_id" not in participants_df.columns:
            raise ValueError("participants.tsv found but no 'participant_id' column present")

        return participants_df["participant_id"].to_list()

    def populate_data(self, manifest_item) -> dict:
        return {
            "channels": self._load_ieeg_electrodes(manifest_item.electrodes_file, manifest_item.channels_file),
            "ieeg": self._load_ieeg_data(manifest_item.ieeg_file),
        }

    def _load_ieeg_electrodes(self, electrodes_file: Path, channels_file: Path) -> ArrayDict:
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

    def _load_ieeg_data(self, ieeg_file: Path, suppress_warnings: bool = True):
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
            data=raw.get_data().astype(np.float32).T
            * 1e6,  # shape should be (n_samples, n_channels), and convert to microvolts
            sampling_rate=int(raw.info["sfreq"]),
            domain_start=0.0,  # Start of the domain (in seconds)
            domain="auto",  # Automatically determine the domain based on the data # type:ignore
        )
