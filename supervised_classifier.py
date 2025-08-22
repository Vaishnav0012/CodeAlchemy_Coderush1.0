import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupervisedEEGClassifier:
    """Final classifier for patient vs control classification"""
    
    def __init__(self):
        self.rf_model = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_data(self, features_path, processed_subjects_path):
        """Load feature data and raw epochs with automatic shape adjustment"""
        logger.info("📊 Loading balanced dataset...")

        features_df = pd.read_csv(features_path)
        with open(processed_subjects_path, 'rb') as f:
            processed_subjects = pickle.load(f)

        all_epochs, all_labels, all_ids = [], [], []
        expected_channels = 20
        expected_samples = 512

        for subject_id, data in processed_subjects.items():
            epochs = data['combined_epochs']
            label = data['label']

            # Convert 24-channel → 20-channel if needed
            if epochs.shape[1] == 24:
                epochs = epochs[:, :20, :]
            elif epochs.shape[1] != expected_channels:
                logger.warning(f"Skipping {subject_id}: unsupported {epochs.shape[1]} channels")
                continue

            # Pad or trim samples
            for i in range(epochs.shape[0]):
                epoch = epochs[i]
                if epoch.shape[1] < expected_samples:
                    epoch = np.pad(epoch, ((0,0),(0,expected_samples - epoch.shape[1])), mode='edge')
                elif epoch.shape[1] > expected_samples:
                    epoch = epoch[:, :expected_samples]
                all_epochs.append(epoch)
                all_labels.append(label)
                all_ids.append(f"{subject_id}_epoch_{i}")

        if not all_epochs:
            raise ValueError("No valid epochs found!")

        raw_epochs = np.array(all_epochs)
        raw_labels = np.array(all_labels)

        # Filter feature dataframe
        metadata_cols = ['subject_id', 'label', 'epoch_id']
        feature_cols = [col for col in features_df.columns if col not in metadata_cols]
        valid_mask = features_df['epoch_id'].isin(all_ids)
        features_df_filtered = features_df[valid_mask]

        X_features = features_df_filtered[feature_cols].values
        y_features = features_df_filtered['label'].values

        logger.info(f"✅ Dataset loaded successfully: {len(raw_epochs)} epochs")
        return X_features, y_features, raw_epochs, raw_labels, feature_cols
    
    def train_random_forest(self, X, y, feature_names):
        logger.info("🌳 Training Random Forest Classifier...")
        if len(np.unique(y)) < 2:
            logger.warning("Only one class found, cannot train.")
            return None, None, None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )
        self.rf_model.fit(X_train_scaled, y_train)

        y_pred = self.rf_model.predict(X_test_scaled)
        y_pred_proba = self.rf_model.predict_proba(X_test_scaled)[:,1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cv_scores = cross_val_score(self.rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

        self.results['rf'] = {
            'model': self.rf_model, 'accuracy': accuracy, 'auc': auc, 'cv_scores': cv_scores,
            'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba,
            'feature_names': feature_names, 'feature_importance': self.rf_model.feature_importances_
        }

        logger.info(f"✅ RF Results: Accuracy={accuracy:.3f}, AUC={auc:.3f}, CV={cv_scores.mean():.3f}")
        return accuracy, auc, cv_scores
    
    def generate_simple_report(self):
        logger.info("📊 Generating report...")
        if 'rf' not in self.results:
            logger.error("No RF results.")
            return

        rf = self.results['rf']
        top_indices = np.argsort(rf['feature_importance'])[-10:]
        top_features = [rf['feature_names'][i] for i in top_indices]
        top_importance = rf['feature_importance'][top_indices]

        fig, axes = plt.subplots(1, 3, figsize=(18,5))
        # Feature importance
        axes[0].barh(range(len(top_features)), top_importance)
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features)
        axes[0].set_xlabel('Importance'); axes[0].set_title('Top 10 Features'); axes[0].grid(True, alpha=0.3)
        # Confusion matrix
        cm = confusion_matrix(rf['y_test'], rf['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual'); axes[1].set_title('Confusion Matrix')
        axes[1].set_xticklabels(['Control','Patient']); axes[1].set_yticklabels(['Control','Patient'])
        # ROC curve
        fpr, tpr, _ = roc_curve(rf['y_test'], rf['y_pred_proba'])
        axes[2].plot(fpr, tpr, label=f'ROC (AUC={rf["auc"]:.3f})'); axes[2].plot([0,1],[0,1],'k--', alpha=0.5)
        axes[2].set_xlabel('FPR'); axes[2].set_ylabel('TPR'); axes[2].set_title('ROC Curve'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        output_dir = Path('data/results/')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir/'classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n🎯 CLASSIFICATION RESULTS")
        print(classification_report(rf['y_test'], rf['y_pred'], target_names=['Control','Patient']))
        tn, fp, fn, tp = cm.ravel()
        print(f"Sensitivity: {tp/(tp+fn):.3f}, Specificity: {tn/(tn+fp):.3f}")
        print("Top 5 Features:")
        top5_idx = np.argsort(rf['feature_importance'])[-5:]
        for i, idx in enumerate(reversed(top5_idx), 1):
            print(f"{i}. {rf['feature_names'][idx]}: {rf['feature_importance'][idx]:.4f}")

def main():
    features_path = 'data/features/aszed_features.csv'
    processed_subjects_path = 'data/processed/aszed_processed_subjects.pkl'
    if not Path(features_path).exists() or not Path(processed_subjects_path).exists():
        print("❌ Required data files missing!")
        return

    classifier = SupervisedEEGClassifier()
    try:
        X_features, y_features, X_raw, y_raw, feature_names = classifier.load_data(features_path, processed_subjects_path)
        rf_acc, rf_auc, rf_cv = classifier.train_random_forest(X_features, y_features, feature_names)
        if rf_acc is not None:
            classifier.generate_simple_report()
        else:
            print("⚠️ Could not train classifier.")
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
