"""
ScamText AI - Prediction and Explainability Module
Provides predictions with explainable indicators
"""

import numpy as np
import pandas as pd
from src.preprocessing import TextPreprocessor


class ScamDetector:
    """Main prediction interface with explainability"""

    def __init__(self, model, feature_engineer, vectorizer=None):
        self.model = model
        self.feature_engineer = feature_engineer
        self.vectorizer = vectorizer
        self.preprocessor = TextPreprocessor()

    def predict(self, message):
        """
        Predict if message is scam with explainability

        Returns:
        {
            'is_scam': bool,
            'confidence': float,
            'category': str,
            'indicators': dict,
            'top_features': list
        }
        """
        # Extract indicators
        indicators = self.preprocessor.extract_scam_indicators(message)

        # Clean and preprocess
        cleaned = self.preprocessor.preprocess(message)

        # Get prediction
        if not cleaned:
            raise ValueError("Message must contain at least one non-whitespace character.")
        if not hasattr(self.model, "model") or self.model.model is None:
            raise ValueError("A trained classification model is required for prediction.")

        X = self.feature_engineer.tfidf.transform([cleaned])
        prediction = int(self.model.predict(X)[0])
        probabilities = self.model.model.predict_proba(X)[0]
        classes = list(self.model.model.classes_)
        if 1 not in classes:
            raise ValueError("The loaded model was not trained with the scam class (label 1).")
        scam_probability = float(probabilities[classes.index(1)])
        confidence = scam_probability if prediction == 1 else 1.0 - scam_probability

        is_scam = prediction == 1

        top_features = self._get_top_features(X, is_scam)

        return {
            'is_scam': is_scam,
            'confidence': float(confidence),
            'scam_probability': scam_probability,
            'category': None,
            'indicators': indicators,
            'top_features': top_features,
            'raw_message': message,
            'cleaned_message': cleaned
        }

    def predict_batch(self, messages):
        """Predict for batch of messages"""
        results = []
        for msg in messages:
            results.append(self.predict(msg))
        return results

    def _get_top_features(self, X, is_scam, top_n=5):
        """Return the strongest TF-IDF/logistic-coefficient contributions."""
        estimator = self.model.model
        if not hasattr(estimator, "coef_"):
            return []
        feature_values = X.toarray()[0]
        coefficients = estimator.coef_[0]
        contributions = feature_values * coefficients
        indices = (
            contributions.argsort()[-top_n:][::-1]
            if is_scam
            else contributions.argsort()[:top_n]
        )
        names = self.feature_engineer.get_tfidf_feature_names()
        return [
            {"feature": names[index], "contribution": float(contributions[index])}
            for index in indices
            if (contributions[index] > 0 if is_scam else contributions[index] < 0)
        ]

    def generate_explanation(self, prediction_result):
        """Generate human-readable explanation"""
        is_scam = prediction_result['is_scam']
        confidence = prediction_result['confidence']
        indicators = prediction_result['indicators']

        explanation = []

        if is_scam:
            explanation.append(f"🚨 SCAM DETECTED (Confidence: {confidence*100:.1f}%)")
            explanation.append("📌 Category: unavailable (the training labels are binary)")

            if indicators:
                explanation.append("\n⚠️ Red Flags Found:")
                for ind_type, values in indicators.items():
                    explanation.append(f"  • {ind_type.upper()}: {', '.join(values)}")
        else:
            explanation.append(f"✅ NOT FLAGGED AS SPAM (Confidence: {confidence*100:.1f}%)")

        return "\n".join(explanation)


class ExplainabilityEngine:
    """Advanced explainability using SHAP"""

    def __init__(self, model, feature_engineer):
        self.model = model
        self.feature_engineer = feature_engineer

    def get_feature_importance(self, X_test, num_samples=100):
        """Get feature importance using SHAP"""
        try:
            import shap

            print("Computing SHAP values...")
            explainer = shap.LinearExplainer(
                self.model.model,
                X_test[:min(100, len(X_test))],
                feature_names=self.feature_engineer.get_tfidf_feature_names()
            )
            shap_values = explainer.shap_values(X_test[:num_samples])

            return shap_values, explainer
        except ImportError:
            print("⚠️ SHAP not available. Skipping explainability.")
            return None, None
        except Exception as e:
            print(f"⚠️ Error computing SHAP values: {e}")
            return None, None

    def plot_feature_importance(self, shap_values, explainer, output_file="outputs/feature_importance.png"):
        """Plot feature importance"""
        try:
            import shap
            import matplotlib.pyplot as plt

            shap.summary_plot(shap_values, show=False)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {output_file}")
            plt.close()
        except Exception as e:
            print(f"⚠️ Error plotting feature importance: {e}")


if __name__ == "__main__":
    print("Prediction and explainability module loaded successfully")
