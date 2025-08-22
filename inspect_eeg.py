import mne

file_path = "data/raw/ASZED/subset_1/subject_10/1/Phase 1.edf"

# Load raw EEG
raw = mne.io.read_raw_edf(file_path, preload=True)

# Print metadata
print(raw.info)
