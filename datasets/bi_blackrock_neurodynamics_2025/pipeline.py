import json
import shutil

import h5py
import numpy as np
import pandas as pd
from neo.io import BlackrockIO
from temporaldata import ArrayDict, IrregularTimeSeries, RegularTimeSeries

from ieeg_data.pipeline import IEEGPipeline


class Pipeline(IEEGPipeline):
    name = "BIDMC Neurodynamics"
    brainset_id = "bi_blackrock_neurodynamics_2025"
    version = "0.1.0"
    url = ""  # private dataset
    citation = ""  # private dataset

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        subject_dirs = [raw_dir / d.name for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]

        session_ids = [
            (subject_dir, d.name)
            for subject_dir in subject_dirs
            for d in subject_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("_", "."))
        ]

        sessions = []

        for subject_dir, session_id in session_ids:
            session_dir = subject_dir / session_id
            if not (session_dir / f"{session_id}-001.nev").exists():
                raise FileNotFoundError(f"NEV file not found: {session_dir / f'{session_id}-001.nev'}")

            # See if there is a general electrodes file for the subject
            # If not, use the session-specific electrodes file
            electrodes_file = subject_dir / "electrodes.h5"
            if not electrodes_file.exists():
                electrodes_file = session_dir / "electrodes.h5"

            if not electrodes_file.exists():
                raise FileNotFoundError(f"Electrode file not found: {electrodes_file}")

            session = {
                "session_id": session_id,
                "ieeg_file": session_dir / f"{session_id}-001.nev",
                "electrodes_file": electrodes_file,
            }

            if (session_dir / "images.h5").exists():
                session["images_file"] = session_dir / "images.h5"

                if not (session_dir / "images.json").exists():
                    raise FileNotFoundError(
                        f"Images JSON file not found: {session_dir / 'images.json'}. This file is required to save the images."
                    )

                session["images_json"] = session_dir / "images.json"

            if (session_dir / "triggers.json").exists():
                session["triggers_file"] = session_dir / "triggers.h5"

            sessions.append(session)

        manifest = pd.DataFrame(sessions).set_index("session_id")
        return manifest

    def populate_data(self, manifest_item) -> dict:
        data = {
            "channels": self._load_ieeg_electrodes(manifest_item.electrodes_file),
            "ieeg": self._load_ieeg_data(manifest_item.ieeg_file),
        }

        if hasattr(manifest_item, "images_file"):
            data["images"] = self._load_images(manifest_item.images_file)

        if hasattr(manifest_item, "triggers_file"):
            data["triggers"] = self._load_triggers(manifest_item.triggers_file)

        return data

    def save_additional(self, save_dir, manifest_item) -> None:
        """Save any additional files or metadata associated with the dataset.

        Args:
            save_dir (Path): Directory where the additional files should be saved.
        """

        if hasattr(manifest_item, "images_json"):
            with open(manifest_item.images_json) as f_src:
                images_json = json.load(f_src)
            with open(save_dir / "images.json", "w") as f_dest:
                json.dump(images_json, f_dest, indent=4)

            # If there is a trial_data directory, copy that as well
            parent_dir = manifest_item.images_json.parent
            for item in parent_dir.iterdir():
                if item.is_dir() and item.name.startswith("trial_data"):
                    dest_dir = save_dir / item.name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)  # remove existing directory to avoid conflicts
                    shutil.copytree(item, dest_dir)
                    break  # Only copy first matching directory

    def _load_ieeg_electrodes(self, electrodes_file) -> ArrayDict:
        with h5py.File(electrodes_file, "r") as f:
            electrodes = ArrayDict.from_hdf5(f)
        return electrodes

    def _load_ieeg_data(self, ieeg_file) -> RegularTimeSeries:
        io = BlackrockIO(filename=ieeg_file)
        io.parse_header()

        sampling_rates = io.header["signal_channels"]["sampling_rate"]
        # Assert all sampling rates are the same
        assert np.all(sampling_rates == sampling_rates[0]), (
            f"Not all channels have the same sampling rate. Found rates: {np.unique(sampling_rates)}"
        )
        sampling_rate = sampling_rates[0]

        channel_indices = list(range(len(self.data_dict["channels"].id)))
        raw_analog_signal = io.get_analogsignal_chunk(
            block_index=0, seg_index=0, channel_indexes=channel_indices
        )  # shape: (n_samples, n_channels)
        raw_analog_signal = io.rescale_signal_raw_to_float(
            raw_analog_signal, dtype="float32", channel_indexes=channel_indices
        )  # shape: (n_samples, n_channels). in uV

        return RegularTimeSeries(
            data=raw_analog_signal,
            sampling_rate=sampling_rate,
            domain_start=0.0,
            domain="auto",
        )

    def _load_images(self, images_file) -> IrregularTimeSeries:
        with h5py.File(images_file, "r") as f:
            images = IrregularTimeSeries.from_hdf5(f)
        return images

    def _load_triggers(self, triggers_file) -> IrregularTimeSeries:
        with h5py.File(triggers_file, "r") as f:
            triggers = IrregularTimeSeries.from_hdf5(f)
        return triggers
