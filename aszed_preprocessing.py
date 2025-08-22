import mne
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
import os
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASZEDPreprocessor:
    def __init__(
        self,
        sfreq_target: float = 256.0,
        l_freq: float = 1.0,
        h_freq: float = 40.0,
        notch_freq: float = 50.0,
        epoch_duration: float = 2.0,
        epoch_overlap: float = 1.0,
        reject_peak_to_peak_uv: Optional[float] = 1000.0,  # µV; set None to disable
        average_ref: bool = True,
    ):
        self.sfreq_target = sfreq_target
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.epoch_duration = epoch_duration
        self.epoch_overlap = epoch_overlap
        self.reject_peak_to_peak_uv = reject_peak_to_peak_uv
        self.average_ref = average_ref

        # fixed channel mapping for ASZED -LE format
        self._channel_mapping = {
            'EEG Fp1-LE': 'Fp1', 'EEG Fp2-LE': 'Fp2',
            'EEG F3-LE': 'F3',   'EEG F4-LE': 'F4',
            'EEG C3-LE': 'C3',   'EEG C4-LE': 'C4',
            'EEG P3-LE': 'P3',   'EEG P4-LE': 'P4',
            'EEG O1-LE': 'O1',   'EEG O2-LE': 'O2',
            'EEG F7-LE': 'F7',   'EEG F8-LE': 'F8',
            'EEG T3-LE': 'T7',   'EEG T4-LE': 'T8',
            'EEG T5-LE': 'P7',   'EEG T6-LE': 'P8',
            'EEG Fz-LE': 'Fz',   'EEG Cz-LE': 'Cz',
            'EEG Pz-LE': 'Pz',   'EEG A2-LE': 'A2',
        }

    def process_single_file(self, file_path: Path, base_path: Path):
        """Process a single EDF file and return a dict with epochs data + metadata."""
        full_path = base_path / file_path
        if not full_path.exists():
            logger.warning(f"File not found: {full_path}")
            return None

        try:
            # Load
            raw = mne.io.read_raw_edf(full_path, preload=True, verbose=False)

            # Keep EEG only
            raw.pick('eeg')

            # Safe channel rename: only map keys that are present to avoid errors
            if self._channel_mapping:
                present_map = {k: v for k, v in self._channel_mapping.items() if k in raw.ch_names}
                if present_map:
                    raw.rename_channels(present_map)

            # Filtering: IIR avoids long-kernel FIR issues on short signals
            try:
                raw.notch_filter(self.notch_freq, method='iir', verbose=False)
            except Exception as e:
                logger.debug(f"Notch filter issue on {full_path.name}: {e}")

            try:
                raw.filter(self.l_freq, self.h_freq, method='iir', verbose=False)
            except Exception as e:
                logger.debug(f"Band-pass filter issue on {full_path.name}: {e}")

            # Re-reference if requested
            if self.average_ref:
                raw.set_eeg_reference('average', projection=False, verbose=False)

            # Resample
            if raw.info['sfreq'] != self.sfreq_target:
                raw.resample(self.sfreq_target)

            # Epoching
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=self.epoch_duration,
                overlap=self.epoch_overlap,
                preload=True,
                verbose=False
            )

            # Drop bad (peak-to-peak). Convert µV -> V for MNE.
            if self.reject_peak_to_peak_uv is not None:
                reject = {'eeg': float(self.reject_peak_to_peak_uv) * 1e-6}
                epochs.drop_bad(reject=reject)
            else:
                reject = None  # for logging only

            if len(epochs) == 0:
                logger.warning(f"All epochs were dropped for file: {file_path} (reject={reject})")
                return None

            # Return dict (so downstream code can pickle/store shapes easily)
            return {
                'epochs': epochs.get_data(),                 # (n_epochs, n_channels, n_times)
                'channel_names': epochs.ch_names,
                'sfreq': float(epochs.info['sfreq']),
                'n_epochs': int(len(epochs)),
                'epoch_duration_s': float(self.epoch_duration),
                'epoch_overlap_s': float(self.epoch_overlap),
                'file': str(file_path),
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

def process_aszed_dataset(
    metadata_path: str,
    aszed_root: str,
    output_path: str,
    max_subjects: int = 10
) -> Dict[str, dict]:
    """Process ASZED dataset given a metadata CSV and save a pickle of per-subject info."""
    metadata = pd.read_csv(metadata_path)
    preprocessor = ASZEDPreprocessor()
    aszed_path = Path(aszed_root)
    processed_subjects: Dict[str, dict] = {}
    subject_count = 0

    logger.info(f"Processing up to {max_subjects} subjects from {len(metadata)} total files")

    # group by subject
    for subject_id, subject_data in metadata.groupby('subject_id'):
        if subject_count >= max_subjects:
            break

        logger.info(f"Processing {subject_id} ({len(subject_data)} files)")

        subject_epochs = []
        first_row = subject_data.iloc[0]
        subject_info = {
            'subject_id': subject_id,
            'label': first_row['label'],
            'label_name': first_row['label_name'],
            'sessions': {}
        }

        # process up to 8 files per subject (as in your code)
        for idx, (_, row) in enumerate(subject_data.iterrows()):
            if idx >= 8:
                break

            file_result = preprocessor.process_single_file(Path(row['filename']), aszed_path)
            if file_result is not None:
                session_key = f"session_{row['session']}_phase_{row['phase']}"
                subject_info['sessions'][session_key] = file_result
                subject_epochs.append(file_result['epochs'])

        if subject_epochs:
            # concatenate along epochs axis
            all_epochs = np.concatenate(subject_epochs, axis=0)
            subject_info['combined_epochs'] = all_epochs
            subject_info['total_epochs'] = int(all_epochs.shape[0])
            subject_info['n_channels'] = int(all_epochs.shape[1])
            subject_info['n_samples'] = int(all_epochs.shape[2])

            processed_subjects[subject_id] = subject_info
            subject_count += 1
            logger.info(f"✅ {subject_id}: {all_epochs.shape} epochs, shape: {all_epochs.shape}")
        else:
            logger.info(f"⚠️ {subject_id}: no usable epochs collected")

    os.makedirs(output_path, exist_ok=True)
    output_file = Path(output_path) / 'aszed_processed_subjects.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(processed_subjects, f)
    logger.info(f"✅ Saved {len(processed_subjects)} processed subjects to {output_file}")

    return processed_subjects

if __name__ == "__main__":
    ASZED_ROOT = r"data\raw\ASZED"
    processed = process_aszed_dataset(
        metadata_path='data/raw/ASZED_metadata_with_labels.csv',
        aszed_root=ASZED_ROOT,
        output_path='data/processed/',
        max_subjects=10
    )

    print(f"\n=== Processing Results ===")
    for subject_id, info in processed.items():
        print(f"{subject_id}: {info['total_epochs']} epochs, "
              f"shape: ({info['n_channels']} channels, {info['n_samples']} samples), "
              f"label: {info['label_name']}")
