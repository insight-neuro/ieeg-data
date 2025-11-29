from brainsets.descriptions import BrainsetDescription, DeviceDescription, SessionDescription, SubjectDescription
from temporaldata import ArrayDict, Data, Interval, RegularTimeSeries


class IEEGData(Data):
    """Standardized class representing intracranial EEG (iEEG) data.

    This class can be extended with iEEG-specific methods and attributes as needed.
    """

    brainset: BrainsetDescription
    subject: SubjectDescription
    session: SessionDescription
    device: DeviceDescription

    ieeg = RegularTimeSeries
    channels = ArrayDict
    ieeg_artifacts = Interval
