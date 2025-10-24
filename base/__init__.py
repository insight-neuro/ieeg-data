"""Base classes for iEEG dataset loading."""

from .session import SessionBase

# Note: BIDSSession is not imported here by default to avoid requiring mne-bids
# for datasets that don't use BIDS format. Import it explicitly if needed:
# from base.bids_session import BIDSSession

__all__ = ["SessionBase"]

