"""
ScamText AI - Main Training Pipeline
Orchestrates the entire workflow from data loading to model evaluation
"""

import sys
import argparse
import os
from pathlib import Path
import json
import tempfile

import joblib
import pandas as pd
import numpy as np

# Support both `python src/pipeline.py` and `python -m src.pipeline`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model_training import BaselineModel, AdvancedModel, ModelEvaluator
from src.prediction import ScamDetector, ExplainabilityEngine


class ScamTextAIPipeline:
    def __init__(self, data_dir="data", output_dir="outputs", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.data = None
        self.models = {}
        self.evaluator = None

    def load_data(self):
        """Load and combine datasets"""
        print("=" * 60)
        print("STEP 1: LOADING DATASETS")
        print("=" * 60)

        loader = DataLoader(str(self.data_dir))
        self.data = loader.combine_datasets()
        loader.save_combined_data("combined_dataset.csv")
        stats = loader.get_data_stats()
        with (self.output_dir / "dataset_statistics.json").open("w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, indent=2)

        return self.data

    def preprocess_data(self):
        """Preprocess and prepare data"""
        print("\n" + "=" * 60)
        print("STEP 2: PREPROCESSING AND FEATURE ENGINEERING")
        print("=" * 60)

        preprocessor = DataPreprocessor()
        data_dict = preprocessor.prepare_data(self.data)

        self.preprocessor = preprocessor
        self.data_dict = data_dict
        split_stats = {
            "total_samples": int(
                len(data_dict["y_train"])
                + len(data_dict["y_val"])
                + len(data_dict["y_test"])
            ),
        }
        for split_name in ("train", "validation", "test"):
            labels = data_dict[f"y_{'val' if split_name == 'validation' else split_name}"]
            split_stats[f"{split_name}_samples"] = int(len(labels))
            split_stats[f"{split_name}_class_counts"] = {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(labels, return_counts=True))
            }
        with (self.output_dir / "split_statistics.json").open("w", encoding="utf-8") as stats_file:
            json.dump(split_stats, stats_file, indent=2)

        print("\nData preparation complete.")
        return data_dict

    def train_baseline_models(self):
        """Train baseline models"""
        print("\n" + "=" * 60)
        print("STEP 3: TRAINING BASELINE MODELS")
        print("=" * 60)

        X_train = self.data_dict['X_train']
        y_train = self.data_dict['y_train']
        X_val = self.data_dict['X_val']
        y_val = self.data_dict['y_val']
        X_test = self.data_dict['X_test']
        y_test = self.data_dict['y_test']

        self.evaluator = ModelEvaluator()

        # Logistic Regression
        lr_model = BaselineModel('logistic')
        lr_model.train(X_train, y_train)
        lr_model.tune_threshold(X_val, y_val)
        lr_model.save(str(self.models_dir / "logistic_regression.pkl"))
        self.evaluator.evaluate(lr_model, X_test, y_test, 'Logistic Regression + TF-IDF')
        self.models['logistic'] = lr_model

        # Naive Bayes
        nb_model = BaselineModel('naive_bayes')
        nb_model.train(X_train, y_train)
        nb_model.tune_threshold(X_val, y_val)
        nb_model.save(str(self.models_dir / "naive_bayes.pkl"))
        self.evaluator.evaluate(nb_model, X_test, y_test, 'Naive Bayes + TF-IDF')
        self.models['naive_bayes'] = nb_model

        return self.models

    def train_advanced_model(self):
        """Train advanced DistilBERT model"""
        print("\n" + "=" * 60)
        print("STEP 4: TRAINING ADVANCED MODEL (DistilBERT)")
        print("=" * 60)

        X_train_raw = self.data_dict['raw_texts_train']
        y_train = self.data_dict['y_train']
        X_val_raw = self.data_dict['raw_texts_val']
        y_val = self.data_dict['y_val']
        X_test_raw = self.data_dict['raw_texts_test']
        y_test = self.data_dict['y_test']

        distilbert_model = AdvancedModel()
        distilbert_model.train(X_train_raw, y_train, epochs=2, batch_size=16)
        distilbert_model.tune_threshold(X_val_raw, y_val)
        distilbert_model.save(str(self.models_dir / "distilbert"))
        self.evaluator.evaluate(distilbert_model, X_test_raw, y_test, 'DistilBERT')
        self.models['distilbert'] = distilbert_model

        return self.models

    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "=" * 60)
        print("STEP 5: MODEL EVALUATION AND COMPARISON")
        print("=" * 60)

        comparison_df = self.evaluator.compare_models()

        # Save comparison
        comparison_df.to_csv(self.output_dir / "model_comparison.csv", index_label="model")
        print(f"\nComparison saved to {self.output_dir / 'model_comparison.csv'}")

        # Generate visualizations
        self.evaluator.plot_confusion_matrices(str(self.output_dir))
        self.evaluator.plot_roc_curves(str(self.output_dir))
        self.evaluator.save_results(str(self.output_dir / "results.json"))
        logistic_model = self.models["logistic"]
        artifact = {
            "model": logistic_model.model,
            "threshold": logistic_model.threshold,
            "vectorizer": self.preprocessor.feature_engineer.tfidf,
        }
        with tempfile.NamedTemporaryFile(
            prefix="scamtext_detector_",
            suffix=".tmp",
            dir=self.models_dir,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            joblib.dump(artifact, temporary_path)
            os.replace(temporary_path, self.models_dir / "scamtext_detector.joblib")
        finally:
            temporary_path.unlink(missing_ok=True)

        return comparison_df

    def run_pipeline(self, include_distilbert=False):
        """Execute complete pipeline"""
        print("\n" + "=" * 60)
        print("SCAMTEXT AI - COMPLETE PIPELINE")
        print("=" * 60)

        self.load_data()
        self.preprocess_data()
        self.train_baseline_models()
        if include_distilbert:
            self.train_advanced_model()
        else:
            print(
                "\nDistilBERT was not selected. The reported experiment compares "
                "the two TF-IDF baselines only."
            )
        comparison = self.evaluate_models()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Models saved to: {self.models_dir}")

        return self.models, comparison

    def run_distilbert_only(self):
        """Retrain only DistilBERT and reuse the saved baseline estimators."""
        self.load_data()
        self.preprocess_data()
        self.evaluator = ModelEvaluator()

        baseline_inputs = (
            ("logistic", "logistic_regression.pkl", "Logistic Regression + TF-IDF"),
            ("naive_bayes", "naive_bayes.pkl", "Naive Bayes + TF-IDF"),
        )
        for model_key, filename, display_name in baseline_inputs:
            model = BaselineModel(model_key)
            model.load(str(self.models_dir / filename))
            self.models[model_key] = model
            self.evaluator.evaluate(
                model,
                self.data_dict["X_test"],
                self.data_dict["y_test"],
                display_name,
            )

        self.train_advanced_model()
        return self.evaluate_models()

    def get_best_model(self):
        """Get the best performing model"""
        if not self.evaluator or not self.evaluator.results:
            return None

        best_model_name = max(
            self.evaluator.results,
            key=lambda name: self.evaluator.results[name]["metrics"]["f1"],
        )
        normalized_name = best_model_name.lower()
        if normalized_name.startswith("logistic regression"):
            model_key = "logistic"
        elif normalized_name.startswith("naive bayes"):
            model_key = "naive_bayes"
        elif normalized_name.startswith("distilbert"):
            model_key = "distilbert"
        else:
            raise ValueError(f"Cannot resolve best model name: {best_model_name}")
        return self.models[model_key], best_model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ScamText AI.")
    parser.add_argument(
        "--with-distilbert",
        action="store_true",
        help="Also fine-tune DistilBERT (requires model download and additional compute).",
    )
    parser.add_argument(
        "--distilbert-only",
        action="store_true",
        help="Train DistilBERT only and reuse the saved baseline models.",
    )
    args = parser.parse_args()
    pipeline = ScamTextAIPipeline()
    if args.distilbert_only:
        comparison = pipeline.run_distilbert_only()
    else:
        models, comparison = pipeline.run_pipeline(include_distilbert=args.with_distilbert)

    print("\n" + "=" * 60)
    print("Ready for deployment.")
    print("=" * 60)
