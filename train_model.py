"""
Multi-Modal Heart Disease Model Training Script
Train on Framingham dataset or similar cardiovascular datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import pickle
import os
from datetime import datetime

# ==================== MODEL ARCHITECTURE ====================
class MultiModalHeartDiseaseModel(nn.Module):
    def __init__(self):
        super(MultiModalHeartDiseaseModel, self).__init__()
        
        # Cardiovascular Branch (5 features)
        self.cardio_branch = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Metabolic Branch (6 features)
        self.metabolic_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Electrolyte/Lab Branch (5 features)
        self.lab_branch = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Demographics/Risk Branch (6 features)
        self.demo_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Modal-specific output heads
        self.cardio_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.metabolic_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.lab_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.demo_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
    
    def forward(self, cardio, metabolic, lab, demo):
        # Process each modality
        cardio_out = self.cardio_branch(cardio)
        metabolic_out = self.metabolic_branch(metabolic)
        lab_out = self.lab_branch(lab)
        demo_out = self.demo_branch(demo)
        
        # Modal predictions
        cardio_risk = self.cardio_head(cardio_out)
        metabolic_risk = self.metabolic_head(metabolic_out)
        lab_risk = self.lab_head(lab_out)
        demo_risk = self.demo_head(demo_out)
        
        # Combine all branches
        combined = torch.cat([cardio_out, metabolic_out, lab_out, demo_out], dim=1)
        
        # Final prediction
        final_pred = self.fusion(combined)
        
        return final_pred, {
            'cardiovascular': cardio_risk,
            'metabolic': metabolic_risk,
            'labs': lab_risk,
            'demographics': demo_risk
        }

# ==================== DATASET CLASS ====================
class HeartDiseaseDataset(Dataset):
    def __init__(self, cardio, metabolic, labs, demo, targets):
        self.cardio = torch.FloatTensor(cardio)
        self.metabolic = torch.FloatTensor(metabolic)
        self.labs = torch.FloatTensor(labs)
        self.demo = torch.FloatTensor(demo)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.cardio[idx],
            self.metabolic[idx],
            self.labs[idx],
            self.demo[idx],
            self.targets[idx]
        )

# ==================== DATA PREPARATION ====================
class DataPreparator:
    def __init__(self, data_path: str = None, use_synthetic: bool = True):
        self.data_path = data_path
        self.use_synthetic = use_synthetic
        self.scalers = {}
        
    def generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic heart disease dataset with clinical correlations"""
        np.random.seed(42)
        
        # Generate base features
        age = np.random.normal(55, 12, n_samples).clip(30, 85)
        sex = np.random.binomial(1, 0.5, n_samples)  # 0=female, 1=male
        bmi = np.random.normal(27, 5, n_samples).clip(18, 45)
        
        # Cardiovascular (with age correlation)
        systolic_bp = np.random.normal(130 + age * 0.3, 18, n_samples).clip(90, 200)
        diastolic_bp = np.random.normal(80 + age * 0.1, 12, n_samples).clip(60, 120)
        heart_rate = np.random.normal(75, 12, n_samples).clip(50, 120)
        hypertension = (systolic_bp > 140).astype(int)
        
        # Metabolic (with age and BMI correlation)
        total_chol = np.random.normal(200 + age * 0.5 + bmi * 1.5, 40, n_samples).clip(120, 350)
        hdl = np.random.normal(50 - bmi * 0.5, 12, n_samples).clip(25, 90)
        ldl = total_chol - hdl - np.random.normal(30, 10, n_samples)
        triglycerides = np.random.normal(150 + bmi * 2, 60, n_samples).clip(50, 400)
        fasting_glucose = np.random.normal(95 + bmi * 1.2, 20, n_samples).clip(70, 250)
        diabetes = (fasting_glucose > 126).astype(int)
        
        # Electrolytes/Labs (with kidney function correlation)
        creatinine = np.random.normal(1.0 + age * 0.008, 0.3, n_samples).clip(0.5, 3.0)
        egfr = np.maximum(10, 120 - age * 0.8 - creatinine * 20)
        
        sodium = np.random.normal(140, 3, n_samples).clip(130, 150)
        # Potassium correlation with kidney function
        potassium = np.random.normal(4.2 + (creatinine - 1) * 0.5, 0.4, n_samples).clip(3.0, 6.5)
        calcium = np.random.normal(9.5, 0.5, n_samples).clip(8.0, 11.0)
        
        # Risk factors
        smoking = np.random.binomial(1, 0.25, n_samples)
        family_history = np.random.binomial(1, 0.3, n_samples)
        physical_activity = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2])
        
        # Generate target with realistic clinical correlations
        risk_score = (
            (age - 40) * 0.02 +
            sex * 0.15 +
            (systolic_bp - 120) * 0.005 +
            (ldl - 100) * 0.002 +
            (hdl < 40) * 0.2 +
            diabetes * 0.25 +
            smoking * 0.3 +
            hypertension * 0.2 +
            family_history * 0.15 +
            (egfr < 60) * 0.2 +
            (potassium > 5.5) * 0.25 +
            (potassium < 3.5) * 0.2 +
            (sodium < 135) * 0.15 +
            (bmi > 30) * 0.1 +
            np.random.normal(0, 0.2, n_samples)
        )
        
        # Convert to probability with a mild scaling to avoid saturation,
        # then sample labels probabilistically so prevalence is realistic
        prob = 1 / (1 + np.exp(-risk_score / 3.0))
        heart_disease = (np.random.rand(n_samples) < prob).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'prevalent_hypertension': hypertension,
            'total_cholesterol': total_chol,
            'hdl': hdl,
            'ldl': ldl,
            'triglycerides': triglycerides,
            'fasting_glucose': fasting_glucose,
            'diabetes': diabetes,
            'sodium': sodium,
            'potassium': potassium,
            'calcium': calcium,
            'creatinine': creatinine,
            'egfr': egfr,
            'smoking': smoking,
            'physical_activity': physical_activity,
            'family_history': family_history,
            'heart_disease': heart_disease
        })
        
        return df
    
    def load_framingham_data(self) -> pd.DataFrame:
        """Load and preprocess Framingham dataset"""
        try:
            df = pd.read_csv(self.data_path)
            
            # Map Framingham columns to our schema
            column_mapping = {
                'age': 'age',
                'male': 'sex',
                'currentSmoker': 'smoking',
                'cigsPerDay': 'smoking_intensity',
                'BPMeds': 'bp_medication',
                'prevalentStroke': 'prev_stroke',
                'prevalentHyp': 'prevalent_hypertension',
                'diabetes': 'diabetes',
                'totChol': 'total_cholesterol',
                'sysBP': 'systolic_bp',
                'diaBP': 'diastolic_bp',
                'BMI': 'bmi',
                'heartRate': 'heart_rate',
                'glucose': 'fasting_glucose',
                'TenYearCHD': 'heart_disease'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Handle missing values BEFORE any calculations
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].fillna(df[col].median())
            
            # Calculate HDL and LDL if not present (approximate)
            if 'hdl' not in df.columns:
                df['hdl'] = df['total_cholesterol'] * 0.25
            if 'ldl' not in df.columns:
                df['ldl'] = df['total_cholesterol'] * 0.65
            if 'triglycerides' not in df.columns:
                df['triglycerides'] = df['total_cholesterol'] * 0.3
            
            # Synthesize lab values based on clinical correlations
            df = self.synthesize_lab_values(df)
            
            # Final check for any remaining missing values
            df = df.fillna(df.median())
            
            return df
            
        except FileNotFoundError:
            print(f"Framingham data not found at {self.data_path}")
            print("Using synthetic data instead...")
            return self.generate_synthetic_data()
    
    def synthesize_lab_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic lab values with clinical correlations"""
        n = len(df)
        
        # Creatinine (higher with age, diabetes, hypertension)
        base_creatinine = 0.9
        df['creatinine'] = (
            base_creatinine +
            (df['age'] - 50) * 0.005 +
            df.get('diabetes', 0) * 0.3 +
            df.get('prevalent_hypertension', 0) * 0.2 +
            np.random.normal(0, 0.2, n)
        ).clip(0.5, 4.0)
        
        # eGFR (inversely related to creatinine and age)
        df['egfr'] = (140 - df['age']) * (1.0 / df['creatinine']) * 0.85
        df.loc[df['sex'] == 0, 'egfr'] *= 0.85  # Lower for females
        df['egfr'] = df['egfr'].clip(15, 120)
        
        # Sodium (lower in heart failure)
        df['sodium'] = np.random.normal(140, 2.5, n).clip(130, 150)
        
        # Potassium (affected by kidney function)
        df['potassium'] = (
            4.0 +
            (df['creatinine'] - 1.0) * 0.8 +
            np.random.normal(0, 0.3, n)
        ).clip(3.0, 6.5)
        
        # Calcium
        df['calcium'] = np.random.normal(9.4, 0.4, n).clip(8.0, 11.0)
        
        # Physical activity (inverse of BMI)
        if 'physical_activity' not in df.columns:
            activity_prob = 1 / (1 + np.exp(-(30 - df['bmi']) / 3))
            # Ensure no NaN values before converting to int
            df['physical_activity'] = (activity_prob * 3).fillna(2).astype(int)
        
        # Family history
        if 'family_history' not in df.columns:
            df['family_history'] = np.random.binomial(1, 0.3, n)
        
        return df
    
    def prepare_data(self) -> Tuple[Dict, Dict, Dict, Dict, np.ndarray]:
        """Prepare and split data into modalities"""
        # Load data
        if self.use_synthetic or self.data_path is None:
            df = self.generate_synthetic_data()
            print(f"Generated {len(df)} synthetic samples")
        else:
            df = self.load_framingham_data()
            print(f"Loaded {len(df)} real samples")
        
        # Define feature groups
        cardio_features = ['systolic_bp', 'diastolic_bp', 'heart_rate', 
                          'prevalent_hypertension', 'age']
        metabolic_features = ['total_cholesterol', 'hdl', 'ldl', 'triglycerides',
                             'fasting_glucose', 'diabetes']
        lab_features = ['sodium', 'potassium', 'calcium', 'creatinine', 'egfr']
        demo_features = ['age', 'bmi', 'smoking', 'family_history', 
                        'sex', 'physical_activity']
        
        # Extract features
        X_cardio = df[cardio_features].values
        X_metabolic = df[metabolic_features].values
        X_labs = df[lab_features].values
        X_demo = df[demo_features].values
        y = df['heart_disease'].values
        
        # Initialize and fit scalers
        self.scalers = {
            'cardiovascular': StandardScaler().fit(X_cardio),
            'metabolic': StandardScaler().fit(X_metabolic),
            'labs': StandardScaler().fit(X_labs),
            'demographics': StandardScaler().fit(X_demo)
        }
        
        # Scale features
        X_cardio_scaled = self.scalers['cardiovascular'].transform(X_cardio)
        X_metabolic_scaled = self.scalers['metabolic'].transform(X_metabolic)
        X_labs_scaled = self.scalers['labs'].transform(X_labs)
        X_demo_scaled = self.scalers['demographics'].transform(X_demo)
        
        return (
            {'train': X_cardio_scaled, 'test': None},
            {'train': X_metabolic_scaled, 'test': None},
            {'train': X_labs_scaled, 'test': None},
            {'train': X_demo_scaled, 'test': None},
            y
        )

# ==================== TRAINER CLASS ====================
class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': []
        }
    
    def train_epoch(self, dataloader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for cardio, metabolic, labs, demo, targets in dataloader:
            cardio = cardio.to(self.device)
            metabolic = metabolic.to(self.device)
            labs = labs.to(self.device)
            demo = demo.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            outputs, modal_outputs = self.model(cardio, metabolic, labs, demo)
            
            # Main loss
            loss = criterion(outputs, targets)
            
            # Add auxiliary losses for each modality
            aux_loss = (
                criterion(modal_outputs['cardiovascular'], targets) +
                criterion(modal_outputs['metabolic'], targets) +
                criterion(modal_outputs['labs'], targets) +
                criterion(modal_outputs['demographics'], targets)
            ) * 0.1
            
            total_loss_batch = loss + aux_loss
            
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_targets, all_preds)
        
        return avg_loss, auc
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for cardio, metabolic, labs, demo, targets in dataloader:
                cardio = cardio.to(self.device)
                metabolic = metabolic.to(self.device)
                labs = labs.to(self.device)
                demo = demo.to(self.device)
                targets = targets.to(self.device)
                
                outputs, _ = self.model(cardio, metabolic, labs, demo)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_targets, all_preds)
        
        return avg_loss, auc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             epochs: int = 100, lr: float = 0.001) -> None:
        """Full training loop"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        best_val_auc = 0
        patience_counter = 0
        patience = 20
        
        print(f"Training on {self.device}")
        print("=" * 60)
        
        for epoch in range(epochs):
            train_loss, train_auc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_auc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print("=" * 60)
        print(f"Best Validation AUC: {best_val_auc:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Comprehensive evaluation"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for cardio, metabolic, labs, demo, targets in test_loader:
                cardio = cardio.to(self.device)
                metabolic = metabolic.to(self.device)
                labs = labs.to(self.device)
                demo = demo.to(self.device)
                
                outputs, _ = self.model(cardio, metabolic, labs, demo)
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Binary predictions
        binary_preds = (all_preds > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'auc_roc': roc_auc_score(all_targets, all_preds),
            'accuracy': accuracy_score(all_targets, binary_preds),
            'precision': precision_score(all_targets, binary_preds),
            'recall': recall_score(all_targets, binary_preds),
            'f1_score': f1_score(all_targets, binary_preds),
            'confusion_matrix': confusion_matrix(all_targets, binary_preds)
        }
        
        return metrics
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # AUC plot
        axes[1].plot(self.history['train_auc'], label='Train AUC')
        axes[1].plot(self.history['val_auc'], label='Val AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_title('Training and Validation AUC')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Training history saved to training_history.png")

# ==================== MAIN TRAINING SCRIPT ====================
def main():
    print("Multi-Modal Heart Disease Risk Assessment - Model Training")
    print("=" * 70)
    
    # Configuration
    USE_SYNTHETIC = False  # Set to False if you have real Framingham data
    DATA_PATH = r'e:\Python\heart-disease-platform\framingham.csv'  # Path to Framingham dataset
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Prepare data
    print("Preparing data...")
    preparator = DataPreparator(
        data_path=DATA_PATH if not USE_SYNTHETIC else None,
        use_synthetic=USE_SYNTHETIC
    )
    
    cardio_data, metabolic_data, labs_data, demo_data, targets = preparator.prepare_data()
    
    # Split data
    indices = np.arange(len(targets))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=targets, random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=VAL_SIZE/(1-TEST_SIZE), stratify=targets[train_idx], random_state=42
    )
    
    print(f"Train samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Positive class ratio: {targets.mean():.2%}\n")
    
    # Create datasets
    train_dataset = HeartDiseaseDataset(
        cardio_data['train'][train_idx],
        metabolic_data['train'][train_idx],
        labs_data['train'][train_idx],
        demo_data['train'][train_idx],
        targets[train_idx]
    )
    
    val_dataset = HeartDiseaseDataset(
        cardio_data['train'][val_idx],
        metabolic_data['train'][val_idx],
        labs_data['train'][val_idx],
        demo_data['train'][val_idx],
        targets[val_idx]
    )
    
    test_dataset = HeartDiseaseDataset(
        cardio_data['train'][test_idx],
        metabolic_data['train'][test_idx],
        labs_data['train'][test_idx],
        demo_data['train'][test_idx],
        targets[test_idx]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("Initializing model...")
    model = MultiModalHeartDiseaseModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Train model
    print("Starting training...")
    trainer = ModelTrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(test_loader)
    
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("=" * 70)
    
    # Save scalers in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scalers_path = os.path.join(script_dir, 'scalers.pkl')
    model_path = os.path.join(script_dir, 'heart_disease_model_final.pth')
    
    with open(scalers_path, 'wb') as f:
        pickle.dump(preparator.scalers, f)
    print(f"\nScalers saved to {scalers_path}")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()