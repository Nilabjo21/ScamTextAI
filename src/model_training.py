"""
ScamText AI - Model Training and Evaluation Module
Trains baseline and advanced models for scam detection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib


class BaselineModel:
    """Baseline models: Logistic Regression, Naive Bayes, SVM"""

    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = None
        self.threshold = 0.5
        self._init_model()

    def _init_model(self):
        """Initialize model based on type"""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif self.model_type == 'svm':
            self.model = LinearSVC(max_iter=2000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.model_type.upper()} model...")
        self.model.fit(X_train, y_train)
        print(f"{self.model_type.upper()} model trained")

    def predict(self, X):
        """Make predictions"""
        if hasattr(self.model, 'predict_proba'):
            return (self.predict_proba(X) >= self.threshold).astype(int)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            class_index = list(self.model.classes_).index(1)
            return self.model.predict_proba(X)[:, class_index]
        elif hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            return self.predict(X)

    def tune_threshold(self, X_validation, y_validation):
        """Choose a classification threshold by validation-set F1 only."""
        if not hasattr(self.model, "predict_proba"):
            return self.threshold
        scores = self.predict_proba(X_validation)
        precision, recall, thresholds = precision_recall_curve(y_validation, scores)
        if len(thresholds) == 0:
            raise ValueError("Validation data does not contain enough probability values to tune a threshold.")
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        self.threshold = float(thresholds[int(np.argmax(f1_values))])
        print(f"Validation-selected threshold ({self.model_type}): {self.threshold:.4f}")
        return self.threshold

    def save(self, filepath):
        """Save model to file"""
        joblib.dump({
            "model": self.model,
            "threshold": self.threshold,
            "model_type": self.model_type,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model from file"""
        artifact = joblib.load(filepath)
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.threshold = float(artifact.get("threshold", 0.5))
            self.model_type = artifact.get("model_type", self.model_type)
        else:
            self.model = artifact
        print(f"Model loaded from {filepath}")


class AdvancedModel:
    """Advanced model using DistilBERT transformer"""

    def __init__(self, model_name='distilbert-base-uncased', device='cpu'):
        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        except ImportError as exc:
            raise RuntimeError("Install the transformers package to train DistilBERT.") from exc

        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )
        self.model.to(device)
        self.threshold = 0.5
        print("DistilBERT model loaded")

    def train(self, X_train, y_train, epochs=3, batch_size=32):
        """Train DistilBERT model"""
        print(f"Training DistilBERT model ({epochs} epochs)...")
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import DataCollatorWithPadding

        tokenizer = self.tokenizer
        device = self.device

        class MessageDataset(Dataset):
            def __init__(self, texts, labels=None):
                self.encodings = tokenizer(
                    list(texts),
                    truncation=True,
                    max_length=256,
                )
                self.labels = None if labels is None else list(labels)

            def __len__(self):
                return len(self.encodings["input_ids"])

            def __getitem__(self, index):
                item = {
                    "input_ids": torch.tensor(
                        self.encodings["input_ids"][index], dtype=torch.long
                    ),
                    "attention_mask": torch.tensor(
                        self.encodings["attention_mask"][index], dtype=torch.long
                    ),
                }
                if self.labels is not None:
                    item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
                return item

        loader = DataLoader(
            MessageDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = self.model(**batch)
                optimizer.zero_grad()
                outputs.loss.backward()
                optimizer.step()
                total_loss += float(outputs.loss.item())
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(loader):.4f}")
        print("DistilBERT model trained")

    def predict(self, X):
        """Make predictions in bounded batches."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_proba(self, X):
        """Return class-1 probabilities in bounded batches."""
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import DataCollatorWithPadding

        tokenizer = self.tokenizer
        device = self.device

        class MessageDataset(Dataset):
            def __init__(self, texts):
                self.encodings = tokenizer(
                    list(texts),
                    truncation=True,
                    max_length=256,
                )

            def __len__(self):
                return len(self.encodings["input_ids"])

            def __getitem__(self, index):
                return {
                    "input_ids": torch.tensor(
                        self.encodings["input_ids"][index], dtype=torch.long
                    ),
                    "attention_mask": torch.tensor(
                        self.encodings["attention_mask"][index], dtype=torch.long
                    ),
                }

        loader = DataLoader(
            MessageDataset(X),
            batch_size=32,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                logits = self.model(**batch).logits
                predictions.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        return np.asarray(predictions)

    def tune_threshold(self, X_validation, y_validation):
        """Select a classification threshold by validation-set F1."""
        scores = self.predict_proba(X_validation)
        precision, recall, thresholds = precision_recall_curve(y_validation, scores)
        if len(thresholds) == 0:
            raise ValueError(
                "Validation data does not contain enough probability values to tune a threshold."
            )
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        self.threshold = float(thresholds[int(np.argmax(f1_values))])
        print(f"Validation-selected threshold (DistilBERT): {self.threshold:.4f}")
        return self.threshold

    def save(self, output_dir):
        """Save the fine-tuned model, tokenizer, and decision threshold."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        with (output_path / "threshold.json").open("w", encoding="utf-8") as threshold_file:
            json.dump({"threshold": self.threshold}, threshold_file, indent=2)
        print(f"DistilBERT model saved to {output_path}")


class ModelEvaluator:
    """Evaluate and compare models"""

    def __init__(self):
        self.results = {}

    def evaluate(self, model, X_test, y_test, model_name):
        """Evaluate model on test set"""
        print(f"\nEvaluating {model_name}...")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else None,
        }
        if hasattr(model, "threshold"):
            metrics["threshold"] = float(model.threshold)

        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        print(f"{model_name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

        return metrics

    def compare_models(self):
        """Create comparison table"""
        print("\nModel Comparison:")

        comparison_df = pd.DataFrame({
            model: data['metrics'] for model, data in self.results.items()
        }).T

        print(comparison_df.to_string())
        return comparison_df

    def plot_confusion_matrices(self, output_dir="outputs"):
        """Plot confusion matrices for all models"""
        Path(output_dir).mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, len(self.results), figsize=(15, 4))
        if len(self.results) == 1:
            axes = [axes]

        for idx, (model_name, data) in enumerate(self.results.items()):
            cm = data['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()
        filepath = Path(output_dir) / 'confusion_matrices.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Confusion matrices saved to {filepath}")
        plt.close()

    def plot_roc_curves(self, output_dir="outputs"):
        """Plot ROC curves for all models"""
        Path(output_dir).mkdir(exist_ok=True)

        plt.figure(figsize=(10, 8))

        for model_name, data in self.results.items():
            y_test = data['y_test']
            y_proba = data['y_proba']

            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        filepath = Path(output_dir) / 'roc_curves.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"ROC curves saved to {filepath}")
        plt.close()

    def save_results(self, output_file="results.json"):
        """Save results to JSON"""
        results_dict = {}
        for model_name, data in self.results.items():
            results_dict[model_name] = {
                'metrics': data['metrics'],
                'confusion_matrix': data['confusion_matrix'].tolist(),
                'test_samples': int(len(data['y_test'])),
            }

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    print("Model training module loaded successfully")
