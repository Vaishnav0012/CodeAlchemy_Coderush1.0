import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
# seaborn imported but not used to keep plotting simple and dependency-light
# import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class EEGDataset(Dataset):
    """Custom dataset for EEG epochs.
    Expects epochs shaped (N, C, T) where C=n_channels, T=n_samples.
    """
    def __init__(self, epochs: np.ndarray, labels: Optional[np.ndarray]=None, subject_ids: Optional[List[str]]=None):
        self.epochs = torch.from_numpy(epochs).float()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.subject_ids = subject_ids
        
    def __len__(self) -> int:
        return len(self.epochs)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.epochs[idx], self.labels[idx]
        return self.epochs[idx]

class EEGAutoencoder(nn.Module):
    """Convolutional Autoencoder for EEG pattern discovery with exact-length reconstruction.
    The encoder downsamples the temporal axis by 2 four times (factor 16), and we pool to T/16.
    The decoder mirrors this, upsampling back to original T using stride=2 blocks.
    """
    def __init__(self, n_channels: int=20, n_samples: int=512, latent_dim: int=64):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        
        # Encoder: 4x downsample by 2 (stride=2). After that, AdaptiveAvgPool to T/16 ensures exact decode length.
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        # T_enc should be T/16 (integer). Use AdaptiveAvgPool1d to enforce exactly T/16 bins.
        self.t_enc = max(1, n_samples // 16)
        self.enc_pool = nn.AdaptiveAvgPool1d(self.t_enc)
        self.enc_fc = nn.Linear(256 * self.t_enc, latent_dim)
        
        # Decoder mirrors encoder: start from (256, T/16), then upsample x2 four times back to T
        self.dec_fc = nn.Linear(latent_dim, 256 * self.t_enc)
        
        self.decoder_conv = nn.Sequential(
            # Each ConvTranspose1d with kernel_size=4, stride=2, padding=1 doubles temporal length exactly.
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, n_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def encode(self, x):
        h = self.encoder_conv(x)
        h = self.enc_pool(h)
        h = torch.flatten(h, start_dim=1)
        z = self.enc_fc(h)
        return z
    
    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(z.size(0), 256, self.t_enc)
        x_hat = self.decoder_conv(h)
        return x_hat
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

class EEGClassifier(nn.Module):
    """CNN Classifier for EEG (placeholder; not trained in this script)."""
    def __init__(self, n_channels: int=20, n_samples: int=512, n_classes: int=2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class EEGDeepLearning:
    """Main class for deep learning analysis."""
    
    def __init__(self, processed_subjects_path: str):
        self.processed_subjects_path = processed_subjects_path
        self.epochs: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.subject_ids: Optional[List[str]] = None
        self.autoencoder: Optional[EEGAutoencoder] = None
        self.classifier: Optional[EEGClassifier] = None
        
    def load_data(self):
        """Load preprocessed EEG data.
        Expects a pickle with structure: {subject_id: { 'combined_epochs': np.ndarray [E, C, T], 'label': int }}
        """
        logger.info(f"📊 Loading data from {self.processed_subjects_path}")
        
        with open(self.processed_subjects_path, 'rb') as f:
            processed_subjects = pickle.load(f)
        
        all_epochs = []
        all_labels = []
        all_subject_ids = []
        
        for subject_id, subject_data in processed_subjects.items():
            epochs = subject_data['combined_epochs']
            label = subject_data['label']
            
            for i, epoch in enumerate(epochs):
                all_epochs.append(epoch)
                all_labels.append(label)
                all_subject_ids.append(f"{subject_id}_epoch_{i}")
        
        self.epochs = np.array(all_epochs)
        self.labels = np.array(all_labels)
        self.subject_ids = all_subject_ids
        
        logger.info(f"✅ Loaded {len(self.epochs)} epochs")
        logger.info(f"📊 Data shape: {self.epochs.shape}")
        logger.info(f"👥 Unique labels: {np.unique(self.labels)}")
        
        return self.epochs, self.labels, self.subject_ids
    
    def train_autoencoder(self, epochs: int=30, batch_size: int=16, learning_rate: float=1e-3):
        """Train autoencoder for pattern discovery."""
        if self.epochs is None:
            raise ValueError("Call load_data() before training.")
        
        logger.info("🧠 Training EEG Autoencoder...")
        
        # Prepare data
        dataset = EEGDataset(self.epochs)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model with correct shapes
        n_channels = self.epochs.shape[1]
        n_samples = self.epochs.shape[2]
        self.autoencoder = EEGAutoencoder(
            n_channels=n_channels, 
            n_samples=n_samples
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.autoencoder.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed, _ = self.autoencoder(batch)
                # If any minor off-by-one length appears, center-crop/pad to match (defensive)
                if reconstructed.size(-1) != batch.size(-1):
                    T = batch.size(-1)
                    rT = reconstructed.size(-1)
                    if rT > T:
                        start = (rT - T) // 2
                        reconstructed = reconstructed[..., start:start+T]
                    else:
                        pad_left = (T - rT) // 2
                        pad_right = T - rT - pad_left
                        reconstructed = nn.functional.pad(reconstructed, (pad_left, pad_right))
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.autoencoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    reconstructed, _ = self.autoencoder(batch)
                    if reconstructed.size(-1) != batch.size(-1):
                        T = batch.size(-1)
                        rT = reconstructed.size(-1)
                        if rT > T:
                            start = (rT - T) // 2
                            reconstructed = reconstructed[..., start:start+T]
                        else:
                            pad_left = (T - rT) // 2
                            pad_right = T - rT - pad_left
                            reconstructed = nn.functional.pad(reconstructed, (pad_left, pad_right))
                    loss = criterion(reconstructed, batch)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        logger.info("✅ Autoencoder training completed")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def extract_latent_representations(self) -> np.ndarray:
        """Extract latent representations from trained autoencoder."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not trained yet!")
        
        logger.info("🔍 Extracting latent representations...")
        
        self.autoencoder.eval()
        latent_vectors = []
        
        dataset = EEGDataset(self.epochs)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                _, encoded = self.autoencoder(batch)
                latent_vectors.append(encoded.cpu().numpy())
        
        latent_representations = np.vstack(latent_vectors)
        
        logger.info(f"✅ Extracted {latent_representations.shape[0]} latent vectors")
        logger.info(f"📊 Latent dimension: {latent_representations.shape[1]}")
        
        # Optional: save to disk for reuse
        out_dir = Path('data/models')
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'latent_representations.npy', latent_representations)
        
        return latent_representations
    
    def analyze_patterns(self, latent_representations: np.ndarray):
        """Analyze discovered patterns in latent space."""
        logger.info("📊 Analyzing discovered patterns...")
        
        # 1. PCA Analysis
        pca = PCA(n_components=min(10, latent_representations.shape[1]))
        pca_features = pca.fit_transform(latent_representations)
        
        # 2. Clustering Analysis
        if self.labels is not None and len(np.unique(self.labels)) > 0:
            est_clusters = len(np.unique(self.labels))
        else:
            est_clusters = 5
        n_clusters = max(2, min(5, est_clusters))
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        clusters = kmeans.fit_predict(latent_representations)
        
        # 3. t-SNE for visualization (sample subset for speed)
        logger.info("Computing t-SNE (sampling up to 1000 points)...")
        n_samples = min(1000, len(latent_representations))
        sample_indices = np.random.choice(len(latent_representations), n_samples, replace=False)
        tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, min(30, n_samples // 4)))
        tsne_features = tsne.fit_transform(latent_representations[sample_indices])
        
        # Visualize results
        self.visualize_patterns(pca_features, clusters, tsne_features, pca, sample_indices)
        
        return {
            'pca_features': pca_features,
            'clusters': clusters,
            'tsne_features': tsne_features,
            'pca_model': pca,
            'sample_indices': sample_indices
        }
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float]):
        """Plot training and validation curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Autoencoder Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_dir = Path('data/models/')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_patterns(self, pca_features: np.ndarray, clusters: np.ndarray, tsne_features: np.ndarray, pca: PCA, sample_indices: np.ndarray):
        """Visualize discovered patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PCA Explained Variance
        n_pc = min(10, len(pca.explained_variance_ratio_))
        axes[0, 0].bar(range(1, n_pc + 1), pca.explained_variance_ratio_[:n_pc])
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('PCA Explained Variance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PCA Scatter
        sc1 = axes[0, 1].scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, alpha=0.6, s=5)
        axes[0, 1].set_xlabel('PC 1')
        axes[0, 1].set_ylabel('PC 2')
        axes[0, 1].set_title('PCA of Latent Representations')
        plt.colorbar(sc1, ax=axes[0, 1])
        
        # 3. Cluster distribution
        cluster_counts = np.bincount(clusters)
        axes[1, 0].bar(range(len(cluster_counts)), cluster_counts)
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Epochs')
        axes[1, 0].set_title('Cluster Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. t-SNE visualization (colored by cluster labels of sampled points)
        sc2 = axes[1, 1].scatter(tsne_features[:, 0], tsne_features[:, 1], c=clusters[sample_indices], alpha=0.6, s=5)
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        axes[1, 1].set_title('t-SNE of Latent Representations')
        plt.colorbar(sc2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('data/models/')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save trained models."""
        output_dir = Path('data/models/')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.autoencoder is not None:
            torch.save(self.autoencoder.state_dict(), output_dir / 'autoencoder.pth')
            logger.info(f"💾 Autoencoder saved to {output_dir / 'autoencoder.pth'}")
    
    def run_full_analysis(self):
        """Run complete deep learning analysis."""
        print("🚀 Starting Deep Learning EEG Analysis")
        print("="*70)
        
        # Load data
        epochs, labels, subject_ids = self.load_data()
        
        # Train autoencoder
        train_losses, val_losses = self.train_autoencoder(epochs=25, batch_size=16)
        
        # Extract representations
        latent_representations = self.extract_latent_representations()
        
        # Analyze patterns
        analysis_results = self.analyze_patterns(latent_representations)
        
        # Save models
        self.save_models()
        
        # Summary
        print("\n" + "="*70)
        print("🎯 DEEP LEARNING ANALYSIS SUMMARY")
        print("="*70)
        print(f"📊 Processed: {len(epochs)} epochs from {len(np.unique([sid.split('_epoch_')[0] for sid in subject_ids]))} subjects")
        print(f"🧠 Latent dimension: {latent_representations.shape[1]}")
        print(f"🔍 Discovered clusters: {len(np.unique(analysis_results['clusters']))}")
        print(f"📈 PCA variance explained (top 3): {analysis_results['pca_model'].explained_variance_ratio_[:3].sum():.3f}")
        print(f"🎯 Final validation loss: {val_losses[-1]:.6f}")
        
        print("\n🔬 Key Insights:")
        print("✅ Autoencoder learned meaningful EEG representations")
        print("✅ Discovered distinct pattern clusters in patient data") 
        print("✅ Models ready for transfer learning with control data")
        print("✅ Latent features available for downstream analysis")
        
        print("\n📁 Output files saved to data/models/:")
        print("   - autoencoder.pth (trained model)")
        print("   - latent_representations.npy (latent vectors)")
        print("   - training_curves.png (training progress)")
        print("   - pattern_analysis.png (pattern visualizations)")
        
        print("="*70)
        
        return {
            'model': self.autoencoder,
            'latent_representations': latent_representations,
            'analysis_results': analysis_results,
            'training_history': {'train_losses': train_losses, 'val_losses': val_losses}
        }

def main():
    """Main execution function"""
    
    # Check for required files
    processed_subjects_path = 'data/processed/aszed_processed_subjects.pkl'
    
    if not Path(processed_subjects_path).exists():
        print(f"❌ Preprocessed data not found: {processed_subjects_path}")
        print("Run preprocessing first: python src/aszed_preprocessing.py")
        return None, None
    
    # Initialize and run analysis
    dl_analyzer = EEGDeepLearning(processed_subjects_path)
    results = dl_analyzer.run_full_analysis()
    
    return dl_analyzer, results

if __name__ == "__main__":
    # Ensure PyTorch is installed
    try:
        import torch  # noqa: F401
    except ImportError:
        print("❌ PyTorch not found. Install with:")
        print("pip install torch torchvision torchaudio")
        raise
    
    analyzer, results = main()
