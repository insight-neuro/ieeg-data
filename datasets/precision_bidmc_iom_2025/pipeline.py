# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#     "numpy>=1.24.0",
#     "ieeg-data",
# ]
# ///

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, Data, RegularTimeSeries

from ieeg_data.pipeline import IEEGPipeline

from .file_loader import parse_probe_map
from .intanutil import intan_rhd as intan


class Pipeline(IEEGPipeline):
    name = "Precision Intraoperative Monitoring Data at BIDMC (2025)"
    brainset_id = "precision_bidmc_iom_2025"
    version = "0.1.0"
    url = ""  # private dataset
    citation = ""  # private dataset

    # Standard parameters for Precision arrays
    N_CHANNELS = 1024

    # post init
    def _post_init(self):
        self.parser.add_argument(
            "--impedance_threshold_min",
            type=float,
            default=1e3,
            help="Minimum impedance threshold for filtering bad channels in Ohms for channel quality filtering.",
        )

        self.parser.add_argument(
            "--impedance_threshold_max",
            type=float,
            default=2.5e6,
            help="Maximum impedance threshold for filtering bad channels in Ohms for channel quality filtering.",
        )

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        # Find all subject directories (format: NSR-###-###)
        subject_dirs = [raw_dir / d.name for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]
        subject_dirs.sort()

        sessions = []
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name

            # Locate probe map file
            probe_map_files = list(
                list(subject_dir.glob("*map*.xml"))
                + list(subject_dir.glob("Precision_map*.xml"))
                + list(subject_dir.parent.glob("*map*.xml"))
            )
            if not probe_map_files:
                raise FileNotFoundError(
                    f"No probe map file found for subject {subject_id} in {subject_dir} or parent directory."
                )
            probe_map_file = probe_map_files[0]

            # Find impedance files (optional), sorted alphabetically
            impedance_files = sorted(subject_dir.glob("*Impedance*.csv"))

            # Iterate over session directories
            for session_dir in subject_dir.iterdir():
                if not session_dir.is_dir() or session_dir.name.startswith(("_", ".")):
                    continue

                session_id = session_dir.name

                # Look for info.rhd file
                info_file = session_dir / "info.rhd"
                if not info_file.exists():
                    continue

                session = {
                    "session_id": session_id,
                    "ieeg_file": info_file,
                    "probe_map_file": probe_map_file,
                }

                # Add hand pose file if available
                if (session_dir / "handpose.h5").exists():
                    session["handpose_file"] = session_dir / "handpose.h5"

                # Add impedance file if available
                if impedance_files:
                    session["impedance_file"] = impedance_files[-1]

                # Check for triggers (digital input data or separate trigger file)
                time_dat = session_dir / "time.dat"
                if time_dat.exists():
                    session["time_file"] = time_dat

                sessions.append(session)

        sessions.sort(key=lambda x: x["session_id"])
        manifest = pd.DataFrame(sessions).set_index("session_id")
        return manifest

    def populate_data(self, manifest_item) -> dict:
        data = {
            "channels": self._load_ieeg_electrodes(
                manifest_item.probe_map_file, getattr(manifest_item, "impedance_file", None)
            ),
            "ieeg": self._load_ieeg_data(manifest_item.ieeg_file),
        }

        if "handpose_file" in manifest_item:
            handpose, handpose_continuous_intervals = self._load_handpose(manifest_item.handpose_file)
            data["handpose"] = handpose
            data["handpose_continuous_intervals"] = handpose_continuous_intervals

        return data

    def _parse_probe_map(self, xml_file: Path) -> tuple[dict, np.ndarray]:
        """
        Parse the Precision probe map XML file using rdutils.

        Returns:
            probe_map: Dictionary mapping channel name (e.g., 'A-001') to (x, y) coordinates
            probe_map_ordered: Numpy array of shape (n_channels, 2) with coordinates
        """
        # Use the rdutils parse_probe_map function
        probe_map, probe_map_ordered = parse_probe_map(str(xml_file), n_channels=self.N_CHANNELS)

        return probe_map, probe_map_ordered

    def _load_clean_channels_from_impedance(
        self, probe_map_ordered: np.ndarray, impedance_file: Path | None
    ) -> tuple[list, np.ndarray]:
        """
        Load and filter channels based on impedance measurements.

        Args:
            probe_map_ordered: Numpy array of shape (n_channels, 2) with (x, y) coordinates

        Returns:
            Tuple of (clean_channels, z_list):
                - clean_channels: List of channel indices that pass impedance thresholds
                - z_list: Array of all impedance values
        """
        if impedance_file is None:
            self.update_status("No impedance file found, using all channels")
            return list(range(self.N_CHANNELS)), np.zeros(self.N_CHANNELS)

        # Read impedance CSV file using np.loadtxt
        try:
            z_array = np.loadtxt(impedance_file, dtype=str, delimiter=",")
            # Extract impedance column (column index 4, skip header row)
            z_list = z_array[1:, 4].astype(float)
        except Exception as e:
            self.update_status(f"Could not read impedance file: {e}. Using all channels")
            return list(range(self.N_CHANNELS)), np.zeros(self.N_CHANNELS)

        # Filter channels by impedance thresholds and probe map validity
        lo_cut, hi_cut = self.impedance_threshold
        clean_channels = []

        for i in range(self.N_CHANNELS):
            if i >= len(z_list):
                break
            # Get coordinates from probe map
            xc, yc = probe_map_ordered[i, 0], probe_map_ordered[i, 1]
            # Check impedance thresholds and coordinate validity
            if xc > -1 and yc > -1 and xc < 33 and z_list[i] < hi_cut and z_list[i] > lo_cut:
                clean_channels.append(i)

        self.update_status(
            f"Found {len(clean_channels)}/{self.N_CHANNELS} clean channels "
            f"({len(clean_channels) / self.N_CHANNELS * 100:.1f}%)"
        )

        return clean_channels, z_list

    def _load_ieeg_electrodes(self, probe_map_file: Path, impedance_file: Path | None) -> ArrayDict:
        """Load electrode information from probe map and impedance data."""
        probe_map, probe_map_ordered = self._parse_probe_map(probe_map_file)

        # Get clean channels based on impedance (now returns tuple)
        clean_channels, z_list = self._load_clean_channels_from_impedance(probe_map_ordered, impedance_file)

        # Create electrode array
        n_channels = self.N_CHANNELS
        channel_ids = [f"ch{i:04d}" for i in range(n_channels)]

        # Extract coordinates from probe_map_ordered (numpy array of shape (n_channels, 2))
        # probe_map_ordered[i] = (x, y) for channel i
        coords = np.zeros((n_channels, 3))
        for i in range(n_channels):
            if i < len(probe_map_ordered):
                x, y = probe_map_ordered[i]
                # Check if coordinates are valid (not -999 which indicates missing)
                if x != -999 and y != -999:
                    coords[i] = [x, y, 0]
                else:
                    coords[i] = [np.nan, np.nan, np.nan]  # Mark missing coordinates
            else:
                coords[i] = [np.nan, np.nan, np.nan]

        # Mark channel status
        channel_status = np.array(["good" if i in clean_channels else "bad" for i in range(n_channels)])

        electrodes = ArrayDict(
            id=np.array(channel_ids),
            coordinates=coords,
            status=channel_status,
        )

        return electrodes

    def _read_intan_data(self, info_file: Path) -> tuple[np.ndarray, float, dict]:
        """
        Read Intan RHD data files using rdutils.

        Returns:
            neural_data: Array of shape (n_channels, n_samples)
            sampling_rate: Sampling rate in Hz
            metadata: Dictionary with recording metadata
        """
        # Read the Intan data with DAT files and voltage rescaling
        # dat=True means read from separate .dat files (amplifier.dat, time.dat, etc.)
        # rescale_to_voltage=True means convert to microvolts (multiply by 0.195)
        data = intan.read_data(str(info_file), dat=True, rescale_to_voltage=True)

        # Extract neural data (amplifier channels)
        neural_data = data["amplifier_data"]  # Shape: (n_channels, n_samples), units: microvolts

        # Get sampling rate from frequency parameters
        sampling_rate = data["frequency_parameters"]["amplifier_sample_rate"]

        metadata = {
            "date": data.get("date", ""),
            "time": data.get("time", ""),
            "notes": data.get("notes", {}),
            "sample_rate": sampling_rate,
        }

        # Store digital input for trigger detection
        # board_dig_in_data has shape (n_digital_channels, n_samples)
        if "board_dig_in_data" in data and data["board_dig_in_data"] is not None:
            metadata["digital_input"] = data["board_dig_in_data"]

        return neural_data, sampling_rate, metadata

    def _load_ieeg_data(self, ieeg_file: Path):
        """Load and preprocess iEEG data from Intan files."""

        # Read raw data
        self.update_status(f"Loading data from {ieeg_file}")
        neural_data, sampling_rate, metadata = self._read_intan_data(ieeg_file)

        self.metadata = metadata  # Store for trigger detection

        # Transpose to (n_samples, n_channels) for RegularTimeSeries
        neural_data = neural_data.T

        self.update_status(f"Loaded data shape: {neural_data.shape}, sampling rate: {sampling_rate} Hz")

        return RegularTimeSeries(
            data=neural_data.astype(np.float32),
            sampling_rate=float(sampling_rate),
            domain_start=0.0,
            domain="auto",
        )

    def _load_handpose(self, handpose_file: Path):
        with h5py.File(handpose_file, "r") as f:
            handpose = Data.from_hdf5(f).materialize()

        handpose_continuous_intervals = handpose.continuous_intervals
        return handpose.handpose, handpose_continuous_intervals
