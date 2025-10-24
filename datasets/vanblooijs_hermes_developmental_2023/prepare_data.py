from __future__ import annotations  # allow compatibility for Python 3.9

import os
from pathlib import Path

import pandas as pd
from mne_bids import BIDSPath
from temporaldata import IrregularTimeSeries

from base.bids_session import BIDSSession


class CCEPSession(BIDSSession):
    name = "CCEP"
    dataset_identifier = "vanblooijs_hermes_developmental_2023"
    dataset_version = "1.2.4"
    url = "https://openneuro.org/datasets/ds004080/versions/1.2.4"
    citation = """@dataset{ds004080:1.2.4,
  author = {D. van Blooijs AND M.A. van den Boom AND J.F. van der Aar AND G.J.M. Huiskamp AND G. Castegnaro AND M. Demuru AND W.J.E.M. Zweiphenning AND P. van Eijsden AND K. J. Miller AND F.S.S. Leijten AND D. Hermes},
  title = {"CCEP ECoG dataset across age 4-51"},
  year = {2023},
  doi = {doi:10.18112/openneuro.ds004080.v1.2.4},
  publisher = {OpenNeuro}
}"""

    def __init__(
        self,
        subject_identifier,
        session_identifier,
        root_dir=None,
        allow_corrupted=False,
    ):
        super().__init__(
            subject_identifier,
            session_identifier,
            root_dir=root_dir,
            allow_corrupted=allow_corrupted,
        )

        self.data_dict["electrical_stimulation"] = self._load_electrical_stimulation()

    @classmethod
    def discover_sessions(cls, subject_identifier: str, root_dir: str | Path | None = None) -> list:
        if root_dir is None:
            root_dir = cls.find_root_dir()
        subject_dir = Path(root_dir) / subject_identifier

        # Find all runs in the subject (equivalent to sessions)
        session_subdir = next(subject_dir.iterdir()) / "ieeg"  # it is always either subject_dir + ses-1 or ses-1b
        assert session_subdir.exists(), f"Session subdirectory not found: {session_subdir}"
        eeg_files = list(session_subdir.glob(f"*{subject_identifier}*.eeg"))
        assert len(eeg_files) >= 1, f"Expected at least 1 eeg file, found {len(eeg_files)}."
        runs = [str(eeg_file).split(subject_identifier)[-1][1:-9].split("run-")[-1] for eeg_file in eeg_files]  #

        # Find the electrodes file
        electrodes_files = list(session_subdir.glob("*electrodes.tsv"))
        assert len(electrodes_files) == 1, f"Expected 1 electrodes file, found {len(electrodes_files)}: {electrodes_files}"
        electrodes_file = str(electrodes_files[0])

        # Add all runs (sessions) to the all_sessions list
        all_sessions = []
        for run in runs:
            eeg_path = BIDSPath(
                subject=subject_identifier[4:],  # remove the "sub-" prefix
                session=next(subject_dir.iterdir()).name[4:],  # remove the "ses-" prefix
                task="SPESclin",
                run=run,
                datatype="ieeg",
                root=root_dir,
            )
            eeg_file_path = str(eeg_path)
            session_identifier = eeg_file_path.split(subject_identifier)[-1][1:-10]  # extract identifier in the form like ses-1_task-SPESclin_run-031556
            all_sessions.append(
                {
                    "session_identifier": session_identifier,
                    "files": {
                        "events_file": eeg_file_path[:-10] + "_events.tsv",
                        "ieeg_file": eeg_path,
                        "ieeg_electrodes_file": electrodes_file,
                        "ieeg_channels_file": eeg_file_path[:-10] + "_channels.tsv",
                    }
                }
            )
        return all_sessions

    def _load_electrical_stimulation(self) -> IrregularTimeSeries:
        events_file = self.session["files"]["events_file"]
        events_df = pd.read_csv(events_file, sep="\t")
        events_df = events_df[events_df["trial_type"].str.upper().isin(["ELECTRICAL_STIMULATION"])]

        return IrregularTimeSeries(
            timestamps=events_df["onset"].to_numpy(),
            stimulation_site=events_df["electrical_stimulation_site"].str.upper().values.astype(str),  # like 'VT1-VT2'
            duration=events_df["duration"].values,
            waveform_type=events_df["electrical_stimulation_type"].str.upper().values.astype(str),  # all monophasic
            current=events_df["electrical_stimulation_current"].values,
            # frequency=events_df['electrical_stimulation_frequency'].values, # all 0.2 Hz; this is single-pulse stim so the frequency is not well defined here.
            pulse_width=events_df["electrical_stimulation_pulsewidth"].values,  # equal to duration since the pulses are monophasic
            timekeys=["timestamps"],
            domain="auto",
        )

    def save_data(self, save_root_dir: str | Path) -> tuple:
        path, data = super().save_data(save_root_dir)

        session_length = data.ieeg.data.shape[0] / data.ieeg.sampling_rate
        n_electrodes = data.ieeg.data.shape[1]
        n_stim_events = data.electrical_stimulation.timestamps.shape[0]
        print(f"\t\tSession length: {session_length:.2f} seconds\t\t{n_electrodes} electrodes\t\t{n_stim_events} stimulation events")
        return path, data


if __name__ == "__main__":
    root_dir = "/home/zaho/orcd/pool/bfm_dataset/ccep/ds004080-1.2.2/"
    import dotenv

    dotenv.load_dotenv()
    save_root_dir = os.getenv("PROCESSED_DATA_DIR")
    if save_root_dir is None:
        raise ValueError("PROCESSED_DATA_DIR environment variable not set.")

    CCEPSession.save_all_subjects_sessions(root_dir=root_dir, save_root_dir=save_root_dir)

