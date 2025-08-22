import pandas as pd

# Load metadata of all EDF files
metadata = pd.read_csv("data/ASZED_metadata.csv")

# Load subject info (diagnosis) - note the filename has .csv.csv
subject_info = pd.read_csv("data/ASZED_subject_info.csv.csv")

# Rename columns to match what we expect
subject_info = subject_info.rename(columns={
    'sn': 'subject',
    'category': 'diagnosis'
})

# Keep only relevant columns: subject and diagnosis
subject_info = subject_info[['subject', 'diagnosis']]

# Merge on 'subject'
metadata = metadata.merge(subject_info, on='subject', how='left')

# Check for missing labels
missing = metadata['diagnosis'].isnull().sum()
print(f"Missing labels: {missing}")

# Print label distribution
print("\nLabel distribution:")
print(metadata['diagnosis'].value_counts())

# Save updated metadata
metadata.to_csv("data/ASZED_metadata_with_labels.csv", index=False)
print("\nMetadata updated with diagnosis labels!")

