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
    def __init__(self, sfreq_target=256, l_freq=1.0, h_freq=40.0, notch_freq=50.0, epoch_duration=2.0, epoch_overlap=1.0):
        self.sfreq_target = sfreq_target
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.epoch_duration = epoch_duration
        self.epoch_overlap = epoch_overlap
    
    def process_single_file(self, file_path, aszed_root):
        try:
            full_path = aszed_root / file_path
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                return None
            
            raw = mne.io.read_raw_edf(full_path, preload=True, verbose=False)
            raw_clean = self.clean_raw(raw)
            epochs = self.create_epochs(raw_clean)
            
            if len(epochs) == 0:
                return None
            
            return {
                'epochs': epochs.get_data(),
                'channel_names': epochs.ch_names,
                'sfreq': epochs.info['sfreq'],
                'n_epochs': len(epochs)
            }
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def clean_raw(self, raw):
        raw = raw.copy()
        raw.pick_types(eeg=True, exclude='bads')
        self._standardize_channels(raw)
        
        if raw.info['sfreq'] != self.sfreq_target:
            raw.resample(self.sfreq_target)
        
        # More lenient filtering for short signals
        if len(raw.times) > 1000:
            raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)
            raw.notch_filter(self.notch_freq, verbose=False)
        
        raw.set_eeg_reference('average', projection=True, verbose=False)
        raw.apply_proj(verbose=False)
        return raw
    
    def _standardize_channels(self, raw):
        """Fixed channel mapping for ASZED -LE format"""
        channel_mapping = {
            'EEG Fp1-LE': 'Fp1', 'EEG Fp2-LE': 'Fp2',
            'EEG F3-LE': 'F3', 'EEG F4-LE': 'F4',
            'EEG C3-LE': 'C3', 'EEG C4-LE': 'C4',
            'EEG P3-LE': 'P3', 'EEG P4-LE': 'P4',
            'EEG O1-LE': 'O1', 'EEG O2-LE': 'O2',
            'EEG F7-LE': 'F7', 'EEG F8-LE': 'F8',
            'EEG T3-LE': 'T7', 'EEG T4-LE': 'T8',
            'EEG T5-LE': 'P7', 'EEG T6-LE': 'P8',
            'EEG Fz-LE': 'Fz', 'EEG Cz-LE': 'Cz', 
            'EEG Pz-LE': 'Pz', 'EEG A2-LE': 'A2'
        }
        mne.rename_channels(raw.info, channel_mapping)
    
    def create_epochs(self, raw):
        epochs = mne.make_fixed_length_epochs(raw, duration=self.epoch_duration, overlap=self.epoch_overlap, preload=True, verbose=False)
        epochs.drop_bad(reject={'eeg': 500e-6})  # Relaxed threshold
        return epochs

def process_aszed_dataset(metadata_path, aszed_root, output_path, max_subjects=10):
    metadata = pd.read_csv(metadata_path)
    preprocessor = ASZEDPreprocessor()
    aszed_path = Path(aszed_root)
    processed_subjects = {}
    subject_count = 0
    
    logger.info(f"Processing up to {max_subjects} subjects from {len(metadata)} total files")
    
    # ✅ CORRECT: Main processing loop
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
        
        for idx, (_, row) in enumerate(subject_data.iterrows()):
            if idx >= 8:
                break
            file_result = preprocessor.process_single_file(Path(row['filename']), aszed_path)
            if file_result:
                session_key = f"session_{row['session']}_phase_{row['phase']}"
                subject_info['sessions'][session_key] = file_result
                subject_epochs.append(file_result['epochs'])
        
        if subject_epochs:
            all_epochs = np.concatenate(subject_epochs, axis=0)
            subject_info['combined_epochs'] = all_epochs
            subject_info['total_epochs'] = all_epochs.shape[0]  # ✅ FIXED:  added
            subject_info['n_channels'] = all_epochs.shape[1]
            subject_info['n_samples'] = all_epochs.shape[2]
            processed_subjects[subject_id] = subject_info
            subject_count += 1
            logger.info(f"✅ {subject_id}: {all_epochs.shape} epochs, shape: {all_epochs.shape}")  # ✅ FIXED
    
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
        print(f"{subject_id}: {info['total_epochs']} epochs, shape: ({info['n_channels']} channels, {info['n_samples']} samples), label: {info['label_name']}")

