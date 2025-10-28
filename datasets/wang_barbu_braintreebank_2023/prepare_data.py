from __future__ import annotations  # allow compatibility for Python 3.9

import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, RegularTimeSeries

from base import SessionBase

logger = logging.getLogger(__name__)


class BrainTreebankSession(SessionBase):
    """
    This class is used to load the iEEG neural data for a given session from the BrainTreebank dataset at https://braintreebank.dev/
    """

    name = "BrainTreebank"
    dataset_identifier = "wang_barbu_braintreebank_2023"
    url = "https://braintreebank.dev/"
    dataset_version = "1.0.0"
    citation = """@misc{wang2024braintreebanklargescaleintracranial,
      title={Brain Treebank: Large-scale intracranial recordings from naturalistic language stimuli}, 
      author={Christopher Wang and Adam Uri Yaari and Aaditya K Singh and Vighnesh Subramaniam and Dana Rosenfarb and Jan DeWitt and Pranav Misra and Joseph R. Madsen and Scellig Stone and Gabriel Kreiman and Boris Katz and Ignacio Cases and Andrei Barbu},
      year={2024},
      eprint={2411.08343},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC},
      url={https://arxiv.org/abs/2411.08343}, 
}"""

    def __init__(
        self,
        subject_identifier,
        session_identifier,
        root_dir=None,
        allow_corrupted=False,
    ):
        # Note: BrainTreebankSession inherits from SessionBase.
        # The data format is not BIDS despite the name.
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
        return list("sub_" + str(i) for i in range(1, 11))  # from 1 to 10. sub_1, sub_2, ..., sub_10

    @classmethod
    def discover_sessions(cls, subject_identifier: str, root_dir: str | Path | None = None):
        if root_dir is None:
            root_dir = cls.find_root_dir()
        root_dir = Path(root_dir)

        all_subject_trials = [
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
            (4, 2),
            (5, 0),
            (6, 0),
            (6, 1),
            (6, 4),
            (7, 0),
            (7, 1),
            (8, 0),
            (9, 0),
            (10, 0),
            (10, 1),
        ]
        this_subject_trial_ids = [trial_id for subject_id, trial_id in all_subject_trials if subject_id == int(subject_identifier[4:])]  # e.g. sub_1 -> [0, 1, 2] - session IDs

        return [
            {
                "session_identifier": f"trial{trial_id:03}",
                "files": {
                    "neural_data_file": root_dir / f"{subject_identifier}_trial{trial_id:03}.h5",
                    "electrode_labels_file": root_dir / f"electrode_labels/{subject_identifier}/electrode_labels.json",
                    "corrupted_electrodes_file": root_dir / "corrupted_elec.json",
                    "localization_file": root_dir / f"localization/{subject_identifier}/depth-wm.csv",
                    "events_file": None,  # TODO: add the events later
                }
            }
            for trial_id in this_subject_trial_ids
        ]

    def __clean_electrode_label(self, electrode_label: str) -> str:
        return electrode_label.replace("*", "").replace("#", "")

    def __filter_electrode_labels(self, electrode_labels: list[str]) -> list[str]:
        """
        Filter the electrode labels to remove corrupted electrodes and electrodes that don't have brain signal
        """
        filtered_electrode_labels = electrode_labels
        # Step 1. Remove corrupted electrodes
        if not self.allow_corrupted:
            corrupted_electrodes_file = self.session["files"]["corrupted_electrodes_file"]
            with open(corrupted_electrodes_file) as f:
                corrupted_electrodes = json.load(f)[self.subject_identifier]
                corrupted_electrodes = [self.__clean_electrode_label(e) for e in corrupted_electrodes]
            filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in corrupted_electrodes]
        # Step 2. Remove trigger electrodes
        trigger_electrodes = [e for e in electrode_labels if (e.upper().startswith("DC") or e.upper().startswith("TRIG"))]
        filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in trigger_electrodes]
        return filtered_electrode_labels

    def _load_ieeg_electrodes(self):
        # Load electrode labels
        electrode_labels_file = self.session["files"]["electrode_labels_file"]
        with open(electrode_labels_file) as f:
            electrode_labels = json.load(f)
        electrode_labels = [self.__clean_electrode_label(e) for e in electrode_labels]
        self._non_filtered_electrode_labels = electrode_labels  # used in load_ieeg_data for accessing the original electrode indices in h5 file

        electrode_labels = self.__filter_electrode_labels(electrode_labels)

        # Load localization data
        localization_file = self.session["files"]["localization_file"]
        df = pd.read_csv(localization_file)
        df["Electrode"] = df["Electrode"].apply(self.__clean_electrode_label)
        coordinates: np.ndarray = np.zeros((len(electrode_labels), 3), dtype=np.float32)
        for label_idx, label in enumerate(electrode_labels):
            row = df[df["Electrode"] == label].iloc[0]
            # Convert coordinates from subject (LPI) to MNI (RAS) space. NOTE: this is not the same as the MNI space used in the BIDS specification. Awaiting proper MNI coordinates from braintreebank.
            # L = Left (+), P = Posterior (+), I = Inferior (+)
            # MNI (RAS): X = Right (+), Y = Anterior (+), Z = Superior (+)
            # So:
            #   X_MNI = -L (flip sign)
            #   Y_MNI = -P (flip sign)
            #   Z_MNI = -I (flip sign)
            x_mni = -row["L"]
            y_mni = -row["P"]
            z_mni = -row["I"]
            coordinates[label_idx] = np.array([x_mni, y_mni, z_mni], dtype=np.float32)

        return ArrayDict(
            id=np.array(electrode_labels),
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            type=np.array(["SEEG"] * len(electrode_labels)),
            brain_area=np.array(["UNKNOWN"] * len(electrode_labels)),
        )

    def _load_ieeg_data(self, suppress_warnings: bool = True):
        neural_data_file = self.session["files"]["neural_data_file"]
        with h5py.File(neural_data_file, "r", locking=False) as f:
            h5_neural_data_keys = {electrode_label: f"electrode_{electrode_i}" for electrode_i, electrode_label in enumerate(self._non_filtered_electrode_labels)}
            # Get data length first
            electrode_data_length = f["data"][h5_neural_data_keys[self._non_filtered_electrode_labels[0]]].shape[0]

            electrode_labels = self.data_dict["channels"].id  # type: ignore[attr-defined]
            # Pre-allocate tensor with specific dtype
            neural_data_cache = np.zeros((len(electrode_labels), electrode_data_length), dtype=np.float32)
            # Load data
            for electrode_id, electrode_label in enumerate(electrode_labels):
                neural_data_key = h5_neural_data_keys[electrode_label]
                neural_data_cache[electrode_id] = f["data"][neural_data_key]

        return RegularTimeSeries(
            data=neural_data_cache.T,
            sampling_rate=2048,
            domain_start=0.0,
            domain="auto",
        )

    def save_data(self, save_root_dir: str | Path) -> tuple:
        path, data = super().save_data(save_root_dir)
        logger.info(f"Saved data for subject {self.subject_identifier} and session {self.session_identifier} to {path}")

        session_length = data.ieeg.data.shape[0] / data.ieeg.sampling_rate
        n_electrodes = data.ieeg.data.shape[1]
        logger.info(f"\t\tSession length: {session_length:.2f} seconds\t\t{n_electrodes} electrodes")
        return path, data
        

if __name__ == "__main__":
    import dotenv, logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    dotenv.load_dotenv()
    save_root_dir = os.getenv("PROCESSED_DATA_DIR")
    if save_root_dir is None:
        raise ValueError("PROCESSED_DATA_DIR environment variable not set.")

    BrainTreebankSession.save_all_subjects_sessions(save_root_dir=save_root_dir, overwrite=True)