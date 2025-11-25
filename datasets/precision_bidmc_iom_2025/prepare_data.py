from __future__ import annotations  # allow compatibility for Python 3.9

import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from temporaldata import ArrayDict, RegularTimeSeries, IrregularTimeSeries, Data

# Handle both relative imports (when used as module) and absolute imports (when run directly)
try:
    from ..base import SessionBase
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from base import SessionBase

# Add rdutils to path
rdutils_path = Path(__file__).parent
if str(rdutils_path) not in sys.path:
    sys.path.insert(0, str(rdutils_path))

from .file_loader import parse_probe_map
from .intanutil import intan_rhd as intan

logger = logging.getLogger(__name__)


class PrecisionSession(SessionBase):
    """
    This class is used to load iEEG neural data from Precision Neuroscience arrays in Intan RHD format.
    Data is recorded using the Precision Neuroscience 1024-channel thin-film cortical arrays.
    """

    name = "Precision Intraoperative Monitoring Data at BIDMC (2025)"
    dataset_identifier = "precision_bidmc_iom_2025"
    dataset_version = "0.1.0"
    url = ""  # private dataset
    citation = ""  # private dataset

    # Standard parameters for Precision arrays
    N_CHANNELS = 1024

    def __init__(self, subject_identifier, session_identifier, root_dir=None, allow_corrupted=False, 
                 impedance_threshold=(1e3, 2.5e6)):
        """
        Initialize PrecisionSession.
        
        Args:
            subject_identifier: Subject ID (e.g., "NSR-005-003")
            session_identifier: Session/recording ID
            root_dir: Root directory containing the data
            allow_corrupted: Whether to allow loading corrupted data
            impedance_threshold: Tuple of (min, max) impedance in Ohms for channel quality filtering
        """
        self.impedance_threshold = impedance_threshold
        
        super().__init__(subject_identifier, session_identifier, root_dir=root_dir, allow_corrupted=allow_corrupted)

        self.data_dict["channels"] = self._load_ieeg_electrodes()
        self.data_dict["ieeg"] = self._load_ieeg_data()

        if "handpose_file" in self.session["files"]:
            handpose, handpose_continuous_intervals = self._load_handpose()
            self.data_dict["handpose"] = handpose
            self.data_dict["handpose_continuous_intervals"] = handpose_continuous_intervals

    @classmethod
    def discover_subjects(cls, root_dir: str | Path | None = None) -> list:
        """Discover all subject directories."""
        if root_dir is None:
            root_dir = cls.find_root_dir()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        # Find all subject directories (format: NSR-###-###)
        return sorted([d.name for d in root_dir.iterdir() 
                      if d.is_dir() and not d.name.startswith('_') and not d.name.startswith('.')])

    @classmethod
    def discover_sessions(cls, subject_identifier: str, root_dir: str | Path | None = None) -> list:
        """Discover all recording sessions for a subject."""
        if root_dir is None:
            root_dir = cls.find_root_dir()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        subject_dir = root_dir / subject_identifier
        
        # Find probe map file
        probe_map_files = list(subject_dir.glob("*map*.xml")) + list(subject_dir.glob("Precision_map*.xml")) + list(subject_dir.parent.glob("*map*.xml"))
        if not probe_map_files:
            raise FileNotFoundError(f"No probe map XML file found in {subject_dir}")
        probe_map_file = probe_map_files[0]
        
        # Find impedance files (optional), sorted alphabetically
        impedance_files = sorted(subject_dir.glob("*Impedance*.csv"))
        
        # Find all session directories containing info.rhd files
        all_sessions = []
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir() or session_dir.name.startswith('_') or session_dir.name.startswith('.'):
                continue
            
            # Look for info.rhd file
            info_file = session_dir / "info.rhd"
            if not info_file.exists():
                continue
            
            session_identifier = session_dir.name
            
            session_files = {
                "ieeg_file": info_file,
                "probe_map_file": probe_map_file,
            }
            
            # Add hand pose file if available
            if (session_dir / "handpose.h5").exists():
                session_files["handpose_file"] = session_dir / "handpose.h5"
            
            # Add impedance file if available
            if impedance_files:
                session_files["impedance_file"] = impedance_files[-1]
            
            # Check for triggers (digital input data or separate trigger file)
            time_dat = session_dir / "time.dat"
            if time_dat.exists():
                session_files["time_file"] = time_dat
            
            all_sessions.append({
                "session_identifier": session_identifier,
                "files": session_files,
            })
        
        return sorted(all_sessions, key=lambda x: x["session_identifier"])

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

    def _load_clean_channels_from_impedance(self, probe_map_ordered: np.ndarray) -> tuple[list, np.ndarray]:
        """
        Load and filter channels based on impedance measurements.
        
        Args:
            probe_map_ordered: Numpy array of shape (n_channels, 2) with (x, y) coordinates
        
        Returns:
            Tuple of (clean_channels, z_list):
                - clean_channels: List of channel indices that pass impedance thresholds
                - z_list: Array of all impedance values
        """
        if "impedance_file" not in self.session["files"]:
            logger.warning("No impedance file found, using all channels")
            return list(range(self.N_CHANNELS)), np.zeros(self.N_CHANNELS)
        
        impedance_file = self.session["files"]["impedance_file"]
        
        # Read impedance CSV file using np.loadtxt
        try:
            z_array = np.loadtxt(impedance_file, dtype=str, delimiter=',')
            # Extract impedance column (column index 4, skip header row)
            z_list = z_array[1:, 4].astype(float)
        except Exception as e:
            logger.warning(f"Could not read impedance file: {e}. Using all channels")
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
        
        logger.info(f"Found {len(clean_channels)}/{self.N_CHANNELS} clean channels "
                   f"({len(clean_channels)/self.N_CHANNELS*100:.1f}%)")
        
        return clean_channels, z_list

    def _load_ieeg_electrodes(self):
        """Load electrode information from probe map and impedance data."""
        probe_map_file = self.session["files"]["probe_map_file"]
        probe_map, probe_map_ordered = self._parse_probe_map(probe_map_file)
        
        # Get clean channels based on impedance (now returns tuple)
        clean_channels, z_list = self._load_clean_channels_from_impedance(probe_map_ordered)
        
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
        channel_status = np.array(['good' if i in clean_channels else 'bad' 
                                  for i in range(n_channels)])
        
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
        neural_data = data['amplifier_data']  # Shape: (n_channels, n_samples), units: microvolts
        
        # Get sampling rate from frequency parameters
        sampling_rate = data['frequency_parameters']['amplifier_sample_rate']
        
        metadata = {
            'date': data.get('date', ''),
            'time': data.get('time', ''),
            'notes': data.get('notes', {}),
            'sample_rate': sampling_rate,
        }
        
        # Store digital input for trigger detection
        # board_dig_in_data has shape (n_digital_channels, n_samples)
        if 'board_dig_in_data' in data and data['board_dig_in_data'] is not None:
            metadata['digital_input'] = data['board_dig_in_data']
        
        return neural_data, sampling_rate, metadata

    def _load_ieeg_data(self):
        """Load and preprocess iEEG data from Intan files."""
        info_file = self.session["files"]["ieeg_file"]
        
        # Read raw data
        logger.info(f"Loading data from {info_file}")
        neural_data, sampling_rate, metadata = self._read_intan_data(info_file)
        
        self.metadata = metadata  # Store for trigger detection
        
        # Transpose to (n_samples, n_channels) for RegularTimeSeries
        neural_data = neural_data.T
        
        logger.info(f"Loaded data shape: {neural_data.shape}, sampling rate: {sampling_rate} Hz")
        
        return RegularTimeSeries(
            data=neural_data.astype(np.float32),
            sampling_rate=float(sampling_rate),
            domain_start=0.0,
            domain="auto",
        )
        
    def _load_images(self):
        with h5py.File(self.session["files"]["images_file"], "r") as f:
            images = IrregularTimeSeries.from_hdf5(f).materialize()
        return images

    def _load_triggers(self):
        with h5py.File(self.session["files"]["triggers_file"], "r") as f:
            triggers = IrregularTimeSeries.from_hdf5(f).materialize()
        return triggers

    def _load_handpose(self):
        with h5py.File(self.session["files"]["handpose_file"], "r") as f:
            handpose = Data.from_hdf5(f).materialize()
        
        handpose_continuous_intervals = handpose.continuous_intervals
        return handpose.handpose, handpose_continuous_intervals


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    
    save_root_dir = os.getenv("PROCESSED_DATA_DIR")
    if save_root_dir is None:
        raise ValueError("PROCESSED_DATA_DIR environment variable not set.")

    # Enable logging
    logging.basicConfig(level=logging.INFO)
    
    PrecisionSession.save_all_subjects_sessions(root_dir=None, save_root_dir=save_root_dir)