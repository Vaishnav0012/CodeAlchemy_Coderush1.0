import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASZEDBaselineModel:
    def __init__(self, features_path):
        self.features_path = features_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Load and prepare feature data"""
        logger.info(f"📊 Loading features from {self.features_path}")
        
        # Load features
        df = pd.read_csv(self.features_path)
        
        # Separate features from metadata
        metadata_cols = ['subject_id', 'label', 'epoch_id']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        X = df[feature_cols].values
        y = df['label'].values
        subjects = df['subject_id'].values
        
        self.feature_names = feature_cols
        
        logger.info(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features")
        
        # Check for class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        logger.info(f"📈 Class distribution:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"   Label {label}: {count} samples ({count/len(y)*100:.1f}%)")
        
        return X, y, subjects, df
    
    def train_model(self, X, y):
        """Train Random Forest model"""
        logger.info("🌳 Training Random Forest model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(X_scaled, y)
        
        logger.info("✅ Model training completed")
        
        return X_scaled
    
    def evaluate_model(self, X_scaled, y, subjects):
        """Evaluate model performance"""
        logger.info("📊 Evaluating model performance...")
        
        # Check if we have multiple classes
        unique_labels = np.unique(y)
        
        if len(unique_labels) == 1:
            logger.warning("⚠️  Only one class present - cannot perform full evaluation")
            logger.info(f"All samples belong to class: {unique_labels[0]}")
            
            # Basic statistics
            train_accuracy = self.model.score(X_scaled, y)
            logger.info(f"Training accuracy: {train_accuracy:.3f}")
            
            return {"training_accuracy": train_accuracy, "note": "Single class dataset"}
        
        # Full evaluation for multi-class data
        try:
            # Subject-wise split for more realistic evaluation
            unique_subjects = np.unique(subjects)
            
            if len(unique_subjects) >= 3:
                # Use subject-wise cross-validation
                cv_scores = []
                for test_subject in unique_subjects:
                    train_mask = subjects != test_subject
                    test_mask = subjects == test_subject
                    
                    X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
                    y_train, y_test = y[train_mask], y[test_mask]
                    
                    # Skip if test set has no samples or only one class
                    if len(X_test) == 0 or len(np.unique(y_test)) < 2:
                        continue
                    
                    temp_model = RandomForestClassifier(
                        n_estimators=100, max_depth=10, random_state=42
                    )
                    temp_model.fit(X_train, y_train)
                    score = temp_model.score(X_test, y_test)
                    cv_scores.append(score)
                
                if cv_scores:
                    logger.info(f"Subject-wise CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            # Regular train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train on training set
            temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
            temp_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = temp_model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Test accuracy: {accuracy:.3f}")
            
            print("\n" + "="*50)
            print("📊 CLASSIFICATION REPORT")
            print("="*50)
            print(classification_report(y_test, y_pred))
            
            return {
                "test_accuracy": accuracy,
                "cv_scores": cv_scores if 'cv_scores' in locals() else None
            }
            
        except Exception as e:
            logger.warning(f"⚠️  Limited evaluation due to data constraints: {e}")
            train_accuracy = self.model.score(X_scaled, y)
            return {"training_accuracy": train_accuracy, "error": str(e)}
    
    def analyze_feature_importance(self, top_n=20):
        """Analyze and visualize feature importance"""
        if self.model is None:
            logger.error("❌ Model not trained yet!")
            return
        
        logger.info(f"🔍 Analyzing top {top_n} most important features...")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Display top features
        print("\n" + "="*60)
        print(f"🏆 TOP {top_n} MOST IMPORTANT FEATURES")
        print("="*60)
        
        for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")
        
        # Save feature importance
        output_dir = Path('data/features/')
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        try:
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(top_n)
            
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance Score')
            plt.title(f'Top {top_n} Feature Importance - Random Forest')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_dir / 'feature_importance_plot.png', dpi=300, bbox_inches='tight')
            logger.info("💾 Feature importance plot saved")
            plt.show()
            
        except Exception as e:
            logger.warning(f"⚠️  Could not create plot: {e}")
        
        return importance_df
    
    def run_full_analysis(self):
        """Run complete baseline analysis"""
        print("🚀 Starting Baseline Model Analysis")
        print("="*70)
        
        # Load data
        X, y, subjects, df = self.load_data()
        
        # Train model
        X_scaled = self.train_model(X, y)
        
        # Evaluate model
        results = self.evaluate_model(X_scaled, y, subjects)
        
        # Analyze features
        importance_df = self.analyze_feature_importance()
        
        # Summary
        print("\n" + "="*70)
        print("🎯 BASELINE MODEL SUMMARY")
        print("="*70)
        print(f"📊 Dataset: {X.shape[0]} epochs, {X.shape[1]} features")
        print(f"👥 Subjects: {len(np.unique(subjects))}")
        print(f"🏷️  Classes: {len(np.unique(y))}")
        
        if 'training_accuracy' in results:
            print(f"🎯 Training Accuracy: {results['training_accuracy']:.3f}")
        if 'test_accuracy' in results:
            print(f"🎯 Test Accuracy: {results['test_accuracy']:.3f}")
        
        print("\n🔮 Next Steps:")
        if len(np.unique(y)) == 1:
            print("1. ⚠️  Process more subjects to get control samples")
            print("2. 🔄 Re-run with balanced dataset")
            print("3. 🧠 Build deep learning models")
        else:
            print("1. ✅ Build deep learning models")
            print("2. 📊 Compare model performances")
            print("3. 🚀 Deploy best model")
        
        print("="*70)
        
        return results, importance_df

def main():
    # Initialize and run analysis
    features_path = 'data/features/aszed_features.csv'
    
    if not Path(features_path).exists():
        print(f"❌ Features file not found: {features_path}")
        print("Run feature extraction first: python src/feature_extraction.py")
        return
    
    model = ASZEDBaselineModel(features_path)
    results, importance_df = model.run_full_analysis()
    
    return model, results, importance_df

if __name__ == "__main__":
    model, results, importance_df = main()
