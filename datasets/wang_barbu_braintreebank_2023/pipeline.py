import json

import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, RegularTimeSeries

from ieeg_data.pipeline import IEEGPipeline


class Pipeline(IEEGPipeline):
    brainset_id = "wang_barbu_braintreebank_2023"
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

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
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

        manifest = pd.DataFrame(
            [
                {
                    "session_id": f"trial_{trial_id:03}",
                    "neural_data": raw_dir / f"sub_{subject_id}_trial{trial_id:03}.h5",
                    "electrode_labels": raw_dir / f"electrode_labels/sub_{subject_id}/electrode_labels.json",
                    "corrupted_electrodes": raw_dir / "corrupted_elec.json",
                    "localization": raw_dir / f"localization/sub_{subject_id}/depth-wm.csv",
                    "events": None,  # TODO: add the events later
                }
                for subject_id, trial_id in all_subject_trials
            ]
        ).set_index("session_id")

        return manifest

    def populate_data(self, manifest_item) -> dict:
        return {
            "channels": self._load_ieeg_electrodes(manifest_item.electrode_labels, manifest_item.localization),
            "ieeg": self._load_ieeg_data(manifest_item.neural_data),
        }

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
                corrupted_electrodes = [e.replace("*", "").replace("#", "") for e in corrupted_electrodes]
            filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in corrupted_electrodes]
        # Step 2. Remove trigger electrodes
        trigger_electrodes = [
            e for e in electrode_labels if (e.upper().startswith("DC") or e.upper().startswith("TRIG"))
        ]
        filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in trigger_electrodes]
        return filtered_electrode_labels

    def _load_ieeg_electrodes(self, electrode_labels_file: str, localization_file: str):
        # Load electrode labels
        with open(electrode_labels_file) as f:
            electrode_labels = json.load(f)
        electrode_labels = [self.__clean_electrode_label(e) for e in electrode_labels]
        self._non_filtered_electrode_labels = (
            electrode_labels  # used in load_ieeg_data for accessing the original electrode indices in h5 file
        )

        electrode_labels = self.__filter_electrode_labels(electrode_labels)

        # Load localization data
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

    def _load_ieeg_data(self, neural_data_file: str):
        with h5py.File(neural_data_file, "r", locking=False) as f:
            h5_neural_data_keys = {
                electrode_label: f"electrode_{electrode_i}"
                for electrode_i, electrode_label in enumerate(self._non_filtered_electrode_labels)
            }
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
