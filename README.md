# Neural Conversion Library for Standardized Data

This repository provides tools for converting various iEEG neural data formats into a standardized `temporaldata` format. The goal is to facilitate data sharing, analysis, and interoperability across different research groups and studies. This library leverages the `brainsets` library to handle the conversion process efficiently.

## Quick Start

| Due to `brainsets` currently only supporting Linux and MacOS, this library is only supported on those operating systems. Windows users can use WSL (Windows Subsystem for Linux) to run this library. 


First, install the required dependencies (we recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/)):
```bash
uv sync # If you want to use a BIDS dataset, make sure to include the --extras bids flag
```

Or alternatively, using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install .[bids]
```

Once the dependencies are installed, you can convert a dataset using the brainsets CLI. Check the [brainsets documentation](https://brainsets.readthedocs.io/en/latest/concepts/using_existing_data.html) for more details on how to use the CLI. Make sure to initialize the brainsets configuration first:
```bash
brainsets config
```
| Note: All ieeg-data processing requires predownloaded datasets. Please ensure that you have the datasets available locally before running the processing commands. 

To process a dataset, run:
```bash
brainsets prepare BRAINSET_NAME /path/to/dataset/ --local --cores N
```
Replace `BRAINSET_NAME` with the appropriate brainset for your dataset (e.g., `bids_eeg`, `bids_ieeg`, etc.), `/path/to/dataset/` with the path to your dataset, and `N` with the number of CPU cores you want to use for processing.

In addition to the standard `brainsets prepare` CLI arguments, `ieeg-data` also supports the following optional arguments:
- `--allow_corrupted`: Allows processing of corrupted data files.
- `--overwrite`: Overwrites existing processed data files.

## Defining New Pipelines

To define a new data pipeline for a specific dataset, create a new class that inherits from `IEEGPipeline` in the `ieeg_data/pipeline.py` file. You will need to implement the following methods:
- `get_manifest(cls, raw_dir: Path, args: Namespace)`: Generate a manifest DataFrame that lists all subjects and sessions in the dataset (see `BrainsetPipeline` for details).
- `populate_data(self, manifest_item: NamedTuple)`: Populate dataset-specific data and metadata for each session. Returns a dictionary of fields to be used in the `temporaldata.Data` object.
- `save_additional(self, save_dir: Path, manifest_item: NamedTuple)` (optional): Save any additional files or metadata associated with the dataset.

If you want to provide additional CLI arguments, you can define them in `__post_init__` method of your pipeline class as shown below:
```python
def __post_init__(self):
    self.args.add_argument("--my_custom_arg", type=str, default="default_value", help="Description of my custom argument.")
```

## Data Format

Each pre-processed session will be saved to `PROCESSED_DATA_DIR/dataset_identifier/subject_identifier/session_identifier/data.h5`, where `PROCESSED_DATA_DIR` can be specified in the `brainsets` configuration. The data will be stored in the following `temporaldata` format:

```python
session = IEEGData(
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
        duration = np.array([0.100]), # seconds. Shape: (n_stimuli, ). How long the image was presented on the screen.
        size = np.array([8]), # Shape: (n_stimuli, ). Size of the stimulus in degrees of the visual field.
        stimulus_ids = np.array([0]),  # Shape: (n_stimuli,). Should point to stimulus_ids in the images.json accompanying the data.h5 file.
        timekeys = ['timestamps'],  # Only timestamps should be adjusted during operations
    ),
    
    # In case the data includes sound played
    sounds = None, # same structure as images
    continuous_sound = None, # unused for now, but may be RegularTimeSeries - the actual raw waveform of the sound
    
    domain=ieeg.domain
)
```