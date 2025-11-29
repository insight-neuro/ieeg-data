import pandas as pd
from mne_bids import BIDSPath
from temporaldata import IrregularTimeSeries

from ieeg_data.bids_pipeline import BIDSPipeline


class Pipeline(BIDSPipeline):
    name = "CCEP"
    brainset_id = "vanblooijs_hermes_developmental_2023"
    version = "1.2.4"
    url = "https://openneuro.org/datasets/ds004080/versions/1.2.4"
    citation = """@dataset{ds004080:1.2.4,
  author = {D. van Blooijs AND M.A. van den Boom AND J.F. van der Aar AND G.J.M. Huiskamp AND G. Castegnaro AND M. Demuru AND W.J.E.M. Zweiphenning AND P. van Eijsden AND K. J. Miller AND F.S.S. Leijten AND D. Hermes},
  title = {"CCEP ECoG dataset across age 4-51"},
  year = {2023},
  doi = {doi:10.18112/openneuro.ds004080.v1.2.4},
  publisher = {OpenNeuro}
}"""

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        subject_ids = cls.discover_subjects(raw_dir)

        sessions = []
        for subject_id in subject_ids:
            # Find all runs in the subject (equivalent to sessions)
            subject_dir = raw_dir / subject_id
            session_subdir = next(subject_dir.iterdir()) / "ieeg"  # it is always either subject_dir + ses-1 or ses-1b
            if not session_subdir.exists():
                raise FileNotFoundError(f"Session subdirectory not found: {session_subdir}")

            eeg_files = list(session_subdir.glob(f"*{subject_id}*.eeg"))
            if not eeg_files:
                raise FileNotFoundError(f"Expected at least 1 eeg file, found {len(eeg_files)}.")
            runs = [str(eeg_file).split(subject_id)[-1][1:-9].split("run-")[-1] for eeg_file in eeg_files]  #

            # Find the electrodes file
            electrodes_files = list(session_subdir.glob("*electrodes.tsv"))
            if len(electrodes_files) != 1:
                raise FileNotFoundError(
                    f"Expected 1 electrodes file, found {len(electrodes_files)}: {electrodes_files}"
                )
            electrodes_file = str(electrodes_files[0])

            # Add all runs (sessions) to the all_sessions list
            for run in runs:
                eeg_path = BIDSPath(
                    subject=subject_id[4:],  # remove the "sub-" prefix
                    session=next(subject_dir.iterdir()).name[4:],  # remove the "ses-" prefix
                    task="SPESclin",
                    run=run,
                    datatype="ieeg",
                    root=raw_dir,
                )
                eeg_file_path = str(eeg_path)
                session_id = eeg_file_path.split(subject_id)[-1][1:-9]  # extract session identifier from file name
                sessions.append(
                    {
                        "sesssion_id": session_id,
                        "events_file": eeg_file_path[:-10] + "_events.tsv",
                        "ieeg_file": eeg_path,
                        "ieeg_electrodes_file": electrodes_file,
                        "ieeg_channels_file": eeg_file_path[:-10] + "_channels.tsv",
                    }
                )

        manifest = pd.DataFrame(sessions).set_index("sesssion_id")
        return manifest

    def populate_data(self, manifest_item) -> dict:
        data = super().populate_data(manifest_item)
        data["electrical_stimulation"] = self._load_electrical_stimulation(manifest_item.events_file)
        return data

    def _load_electrical_stimulation(self, events_file) -> IrregularTimeSeries:
        events_df = pd.read_csv(events_file, sep="\t")
        events_df = events_df[events_df["trial_type"].str.upper().isin(["ELECTRICAL_STIMULATION"])]

        return IrregularTimeSeries(
            timestamps=events_df["onset"].to_numpy(),
            stimulation_site=events_df["electrical_stimulation_site"].str.upper().values.astype(str),  # like 'VT1-VT2'
            duration=events_df["duration"].values,
            waveform_type=events_df["electrical_stimulation_type"].str.upper().values.astype(str),  # all monophasic
            current=events_df["electrical_stimulation_current"].values,
            # frequency=events_df['electrical_stimulation_frequency'].values, # all 0.2 Hz; this is single-pulse stim so the frequency is not well defined here.
            pulse_width=events_df[
                "electrical_stimulation_pulsewidth"
            ].values,  # equal to duration since the pulses are monophasic
            timekeys=["timestamps"],
            domain="auto",
        )
