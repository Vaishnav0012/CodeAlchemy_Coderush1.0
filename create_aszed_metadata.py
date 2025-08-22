import pandas as pd
import os
from pathlib import Path

def scan_aszed_dataset(aszed_root_path):
    """Scan ASZED dataset and create metadata"""
    records = []
    aszed_path = Path(aszed_root_path)
    
    print(f"Scanning: {aszed_path}")
    
    # Process both subsets
    for subset in ['subset_1', 'subset_2']:
        subset_path = aszed_path / subset
        
        if not subset_path.exists():
            print(f"Warning: {subset} not found")
            continue
            
        print(f"Processing {subset}...")
        
        # Get all subject folders
        subject_folders = [d for d in subset_path.iterdir() 
                          if d.is_dir() and 'subject' in d.name.lower()]
        print(f"Found {len(subject_folders)} subjects in {subset}")
        
        for subject_folder in sorted(subject_folders):
            subject_id = subject_folder.name
            subject_num = int(''.join(filter(str.isdigit, subject_id)))
            
            # Get session folders (1, 2, 3, etc.)
            session_folders = [d for d in subject_folder.iterdir() 
                             if d.is_dir() and d.name.isdigit()]
            
            for session_folder in session_folders:
                session_id = session_folder.name
                edf_files = list(session_folder.glob('*.edf'))
                
                for edf_file in sorted(edf_files):
                    relative_path = edf_file.relative_to(aszed_path)
                    
                    # Placeholder labels - adjust based on actual ground truth
                    label = 0 if subject_num <= 80 else 1
                    
                    record = {
                        'subject_id': subject_id,
                        'subject_number': subject_num,
                        'subset': subset,
                        'session': session_id,
                        'phase': edf_file.stem,
                        'filename': str(relative_path).replace('\\', '/'),
                        'full_path': str(edf_file),
                        'label': label,
                        'label_name': 'control' if label == 0 else 'patient',
                        'file_exists': edf_file.exists(),
                        'file_size_mb': round(edf_file.stat().st_size / (1024*1024), 2)
                    }
                    records.append(record)
    
    return pd.DataFrame(records)

# Your correct ASZED path
ASZED_ROOT = r"data\raw\ASZED"

print("=== Creating ASZED Metadata ===")
print(f"Dataset location: {os.path.abspath(ASZED_ROOT)}")

# Verify path exists
if not os.path.exists(ASZED_ROOT):
    print(f"❌ ASZED path not found: {ASZED_ROOT}")
    exit()

# Scan dataset
metadata_df = scan_aszed_dataset(ASZED_ROOT)

if len(metadata_df) == 0:
    print("❌ No EDF files found!")
    exit()

# Results
print(f"\n=== Dataset Summary ===")
print(f"Total EDF files: {len(metadata_df)}")
print(f"Unique subjects: {metadata_df['subject_id'].nunique()}")
print(f"Subsets: {metadata_df['subset'].value_counts().to_dict()}")
print(f"Label distribution: {metadata_df['label_name'].value_counts().to_dict()}")
print(f"Average file size: {metadata_df['file_size_mb'].mean():.1f} MB")

print(f"\n=== Sample Data ===")
print(metadata_df[['subject_id', 'session', 'phase', 'label_name', 'file_size_mb']].head(10))

# Save metadata
output_file = 'data/raw/ASZED_metadata_with_labels.csv'
metadata_df.to_csv(output_file, index=False)

print(f"\n✅ Saved {len(metadata_df)} records to: {output_file}")
print(f"🎯 Next step: python src/aszed_preprocessing.py")

