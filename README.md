# Neural Conversion Library for Standardized Data

This repository provides tools for converting various iEEG neural data formats into a standardized `temporaldata` format. The goal is to facilitate data sharing, analysis, and interoperability across different research groups and studies.

## Installation

To install the necessary requirements and packages, please run the following commands. First, optionally, install a virtual environment with:
```python
python -m venv .venv
source .venv/bin/activate # On Windows: .venv/Scripts/activate
pip install --upgrade pip
```
Then, use `pip` to install the necessary packages to run the code in this repository:
```python
pip install -r requirements.txt
```
Then, create a `.env` file, using the `.env.example` template, where you will have to specify the `PROCESSED_DATA_DIR`, `RAW_DATA_DIR`, and optionally any dataset-specific directories.

## Data Format

Each pre-processed session will be saved to `PROCESSED_DATA_DIR/dataset_identifier/subject_identifier/session_identifier/data.h5`, where `PROCESSED_DATA_DIR` can be specified as an environmental variable in `.env`. The data will be stored in the following `temporaldata` format:
```python
session = Data(
    # metadata
    brainset = "STRING",
    subject = "STRING",
    session = "STRING",
    citation = "STRING", # in bib format
    
    # In case the data includes iEEG. NTOE: EEG is also included here, as it is a generalization (the same data format).
    ieeg = RegularTimeSeries(
        data = seeg_data,  # Shape: (n_timepoints, n_electrodes). Voltage in uV
        sampling_rate = 2048,  # Hz

        domain_start = 0.0,  # Start at 0 seconds
        domain = "auto"  # Let it infer from data length and sampling rate
    ),
    channels = ArrayDict(
        id = np.array(["LAMY1"]), # Shape: (n_electrodes, )
        
        # Coordinates of the corresponding electrodes. Usually, these will be the MNI coordinates. 
        # Note: in some datasets, there will be an exception (if MNI are unavailable or type of probe is different)
        x = np.array([0.0]), # Shape: (n_electrodes, ). if unknown, can be np.nan
        y = np.array([0.0]),
        z = np.array([0.0]),
        brain_area = np.array(["UNKNOWN"]),

        type = np.array(["SEEG"]) # options: SEEG, ECOG, EEG, etc
    ),
    ieeg_artifacts = Interval(
        start = np.array([0.0]), # shape: (n_artifacts, )
        end = np.array([0.0]),
        affected_channels = np.array([[1]], type=bool), # shape: (n_artifacts, n_electrodes)
        description = np.array(["UNKNOWN"]),
        timekeys = ['start', 'end'],  # Only time should be adjusted during operations
    ),

    # In case the data includes any type of triggers. Note: These could be redundant with the other tags below.
    triggers = IrregularTimeSeries(
        timestamps = trigger_times,
        type = np.array(["MOUSE_CLICK"]),
        note = np.array([""]), # Optional note together with the trigger. Can be empty.
        
        timekeys = ['timestamps'],  # Only timestamps should be adjusted during operations
    ),
    
    # In case the data includes stimulation. Note: frequency is not a parameter here! Use many electrical_stimulation events (as separate pulses) to denote the stimulation at a particular frequency.
    electrical_stimulation = IrregularTimeSeries(
        timestamps = stimulus_times,  # Shape (n_stim,). If multiple electrodes/electrode pairs at the same time, there will be multiple entries in the timestamp
        waveform_type = np.array(["BIPHASIC"]),  # (n_stim,).
        stimulation_site = np.array(["ELEC1-ELEC2"]),  # (n_stim,). Can be two electrode labels separated by a dash
        amplitude = np.array([1.0]), # mA
        pulsewidth = np.array([0.001]), # seconds
        duration = np.array([0.014]), # seconds
        timekeys = ['timestamps'],  # Only timestamps should be adjusted during operations
    ),
    
    # In case the data includes images shown
    images = IrregularTimeSeries(
        timestamps = stimulus_times,  # When stimuli were shown. (n_stimuli,)
        stimulus_ids = np.array(stimulus_images),  # Image identifiers
        timekeys = ['timestamps'],  # Only timestamps should be adjusted during operations
    ),
    
    # In case the data includes sound played
    sounds = None, # same structure as images
    continuous_sound = None, # unused for now, but may be RegularTimeSeries - the actual raw waveform of the sound
    
    domain=ieeg.domain
)
```