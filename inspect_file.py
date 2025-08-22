import os
import mne
import pandas as pd

base_path = "data/raw/ASZED"

records = []

# Loop over subsets
for subset in ["subset_1", "subset_2"]:
    subset_path = os.path.join(base_path, subset)
    
    # Loop over subjects
    for subject in os.listdir(subset_path):
        subject_path = os.path.join(subset_path, subject)
        if not os.path.isdir(subject_path):
            continue
        
        # Loop over subfolders (1, 2, 3)
        for block in os.listdir(subject_path):
            block_path = os.path.join(subject_path, block)
            if not os.path.isdir(block_path):
                continue
            
            # Loop over EDF files
            for fname in os.listdir(block_path):
                if fname.endswith(".edf"):
                    file_path = os.path.join(block_path, fname)
                    records.append({
                        "subset": subset,
                        "subject": subject,
                        "block": block,
                        "file": fname,
                        "path": file_path
                    })

# Save metadata to CSV
df = pd.DataFrame(records)
df.to_csv("data/ASZED_metadata.csv", index=False)
print("Metadata created! First 5 rows:")
print(df.head())

