import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy import signal, stats
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGFeatureExtractor:
    def __init__(self, sfreq=256):
        self.sfreq = sfreq
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        self.channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 
                             'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 
                             'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'A2']
    
    def extract_statistical_features(self, epoch):
        """Extract basic statistical features"""
        features = {}
        for ch_idx, ch_name in enumerate(self.channel_names):
            if ch_idx >= epoch.shape[0]:  # Safety check
                continue
                
            ch_data = epoch[ch_idx, :]
            features[f'{ch_name}_mean'] = np.mean(ch_data)
            features[f'{ch_name}_std'] = np.std(ch_data)
            features[f'{ch_name}_var'] = np.var(ch_data)
            features[f'{ch_name}_skew'] = stats.skew(ch_data)
            features[f'{ch_name}_kurtosis'] = stats.kurtosis(ch_data)
            features[f'{ch_name}_rms'] = np.sqrt(np.mean(ch_data**2))
            features[f'{ch_name}_ptp'] = np.ptp(ch_data)  # Peak-to-peak
        
        return features
    
    def extract_frequency_features(self, epoch):
        """Extract frequency domain features"""
        features = {}
        
        for ch_idx, ch_name in enumerate(self.channel_names):
            if ch_idx >= epoch.shape[0]:  # Safety check
                continue
                
            ch_data = epoch[ch_idx, :]
            
            # Power Spectral Density
            freqs, psd = signal.welch(ch_data, self.sfreq, nperseg=min(128, len(ch_data)//4))
            
            # Band power calculation
            for band_name, (low_freq, high_freq) in self.freq_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                    features[f'{ch_name}_{band_name}_power'] = band_power
                else:
                    features[f'{ch_name}_{band_name}_power'] = 0
            
            # Band power ratios
            total_power = np.trapz(psd, freqs)
            if total_power > 0:
                for band_name in self.freq_bands.keys():
                    features[f'{ch_name}_{band_name}_ratio'] = features[f'{ch_name}_{band_name}_power'] / total_power
            else:
                for band_name in self.freq_bands.keys():
                    features[f'{ch_name}_{band_name}_ratio'] = 0
        
        return features
    
    def extract_connectivity_features(self, epoch):
        """Extract inter-channel connectivity features"""
        features = {}
        n_channels = min(epoch.shape[0], len(self.channel_names))
        
        if n_channels < 2:
            return features
        
        # Correlation matrix
        try:
            corr_matrix = np.corrcoef(epoch[:n_channels])
            
            # Handle case where correlation matrix is scalar
            if corr_matrix.ndim == 0:
                features['connectivity_mean'] = float(corr_matrix)
                features['connectivity_std'] = 0.0
                features['connectivity_max'] = float(corr_matrix)
                features['connectivity_min'] = float(corr_matrix)
            else:
                # Extract upper triangular part (avoid duplicates)
                triu_indices = np.triu_indices(n_channels, k=1)
                correlations = corr_matrix[triu_indices]
                
                # Remove NaN values
                correlations = correlations[~np.isnan(correlations)]
                
                if len(correlations) > 0:
                    features['connectivity_mean'] = np.mean(correlations)
                    features['connectivity_std'] = np.std(correlations)
                    features['connectivity_max'] = np.max(correlations)
                    features['connectivity_min'] = np.min(correlations)
                else:
                    features['connectivity_mean'] = 0.0
                    features['connectivity_std'] = 0.0
                    features['connectivity_max'] = 0.0
                    features['connectivity_min'] = 0.0
        except:
            # Fallback values if correlation fails
            features['connectivity_mean'] = 0.0
            features['connectivity_std'] = 0.0
            features['connectivity_max'] = 0.0
            features['connectivity_min'] = 0.0
        
        return features
    
    def extract_all_features(self, epoch):
        """Extract all features for a single epoch"""
        all_features = {}
        
        # Extract different types of features
        stat_features = self.extract_statistical_features(epoch)
        freq_features = self.extract_frequency_features(epoch)
        conn_features = self.extract_connectivity_features(epoch)
        
        # Combine all features
        all_features.update(stat_features)
        all_features.update(freq_features)
        all_features.update(conn_features)
        
        return all_features

def process_subjects_for_ml(processed_subjects_path, output_path):
    """Convert processed subjects to ML-ready features"""
    
    # Load preprocessed data
    try:
        with open(processed_subjects_path, 'rb') as f:
            processed_subjects = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {processed_subjects_path}")
        logger.error("Make sure you've run the preprocessing script first!")
        return None
    
    logger.info(f"✅ Loaded {len(processed_subjects)} subjects for feature extraction")
    
    # Initialize feature extractor
    extractor = EEGFeatureExtractor()
    
    # Storage for features and labels
    all_features = []
    all_labels = []
    subject_ids = []
    
    # Extract features for each subject
    total_epochs = 0
    for subject_id, subject_data in processed_subjects.items():
        logger.info(f"🔄 Extracting features for {subject_id}")
        
        epochs = subject_data['combined_epochs']
        label = subject_data['label']
        n_epochs = epochs.shape[0]
        total_epochs += n_epochs
        
        logger.info(f"   Processing {n_epochs} epochs...")
        
        # Process each epoch
        for epoch_idx, epoch in enumerate(epochs):
            # Extract features for this epoch
            features = extractor.extract_all_features(epoch)
            
            # Store results
            all_features.append(features)
            all_labels.append(label)
            subject_ids.append(f"{subject_id}_epoch_{epoch_idx}")
    
    if not all_features:
        logger.error("❌ No features extracted! Check your data.")
        return None
        
    # Convert to DataFrame
    logger.info("📊 Converting to DataFrame...")
    features_df = pd.DataFrame(all_features)
    
    # Add metadata
    features_df['subject_id'] = [sid.split('_epoch_')[0] for sid in subject_ids]
    features_df['label'] = all_labels
    features_df['epoch_id'] = subject_ids
    
    # Feature statistics
    n_features = len(features_df.columns) - 3  # Exclude metadata columns
    n_samples = len(features_df)
    
    logger.info(f"✅ Extracted {n_features} features from {n_samples} epochs")
    
    # Save features
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_file = output_dir / 'aszed_features.csv'
    features_df.to_csv(features_file, index=False)
    
    # Save feature names for reference
    feature_names = [col for col in features_df.columns if col not in ['subject_id', 'label', 'epoch_id']]
    with open(output_dir / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    logger.info(f"💾 Features saved to: {features_file}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("🎯 FEATURE EXTRACTION SUMMARY")
    print("="*70)
    print(f"📊 Total features extracted: {n_features}")
    print(f"🧠 Total epochs processed: {n_samples}")
    print(f"👥 Subjects processed: {len(processed_subjects)}")
    
    # Label distribution
    label_counts = features_df['label'].value_counts()
    print(f"📈 Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / n_samples) * 100
        print(f"   {label}: {count} epochs ({percentage:.1f}%)")
    
    # Feature type breakdown
    stat_features = len([f for f in feature_names if any(stat in f for stat in ['mean', 'std', 'var', 'skew', 'kurtosis', 'rms', 'ptp'])])
    freq_features = len([f for f in feature_names if any(freq in f for freq in ['delta', 'theta', 'alpha', 'beta', 'gamma'])])
    conn_features = len([f for f in feature_names if 'connectivity' in f])
    
    print(f"📋 Feature breakdown:")
    print(f"   Statistical features: {stat_features}")
    print(f"   Frequency features: {freq_features}")
    print(f"   Connectivity features: {conn_features}")
    
    print(f"\n🎯 Ready for machine learning!")
    print("="*70)
    
    return features_df

if __name__ == "__main__":
    # Process your preprocessed data
    processed_subjects_path = 'data/processed/aszed_processed_subjects.pkl'
    output_path = 'data/features/'
    
    print("🧠 Starting EEG Feature Extraction...")
    print("="*70)
    
    features_df = process_subjects_for_ml(processed_subjects_path, output_path)
    
    if features_df is not None:
        print(f"\n🚀 Next step: Build baseline models with {len(features_df)} feature vectors!")
        print("\nRun this command next:")
        print("python src/baseline_model.py")
    else:
        print("\n❌ Feature extraction failed. Check your preprocessed data.")
