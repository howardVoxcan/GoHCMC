import time
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight


# Constants
DATA_FILE = 'preprocessed_reviews.csv'
LABEL_ENCODER_FILE = 'label_encoder.pkl'
MODEL_FILE = 'svm_tfidf_pipeline.pkl'
RANDOM_STATE = 42
N_FOLDS = 5
MAX_ITERATIONS = 10000


class SentimentModelTrainer:
    """Train and evaluate sentiment analysis model using SVM with TF-IDF."""

    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
        self.pipeline = None

    def load_and_prepare_data(self) -> Tuple[pd.Series, pd.Series]:
        """Load CSV data, remove missing values, and encode labels."""
        print("Loading and preprocessing data...")
        df = pd.read_csv(DATA_FILE)
        df.dropna()

        df['label'] = self.label_encoder.fit_transform(df['sentiment'])
        self._save_label_encoder()

        return df['clean_text'], df['label']

    def _save_label_encoder(self) -> None:
        """Persist the label encoder to disk."""
        joblib.dump(self.label_encoder, LABEL_ENCODER_FILE)
        print(f"Label encoder saved to {LABEL_ENCODER_FILE}")

    @staticmethod
    def compute_class_weights(y: pd.Series) -> Dict[int, float]:
        """Calculate balanced class weights for handling class imbalance."""
        print("Computing class weights...")
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        weight_dict = dict(enumerate(weights))
        print(f"Class weights: {weight_dict}")
        return weight_dict

    def cross_validate(self, x: pd.Series, y: pd.Series, param_grid: Dict) -> None:
        """Run stratified k-fold cross-validation with hyperparameter tuning."""
        print("Starting cross-validation with grid search...")
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        accuracies = []
        f1_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(x, y), start=1):
            print(f"\n{'='*50}")
            print(f"Fold {fold}/{N_FOLDS}")
            print(f"{'='*50}")

            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            best_model = self._train_fold(x_train, y_train, param_grid)
            metrics = self._evaluate_fold(best_model, x_test, y_test)

            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_macro'])

        self._print_cv_summary(accuracies, f1_scores)

    def _train_fold(self, x_train: pd.Series, y_train: pd.Series, param_grid: Dict) -> Pipeline:
        """Train model for a single fold using grid search."""
        pipeline = self._create_pipeline()
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=2,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(x_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def _evaluate_fold(self, model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        y_pred = model.predict(x_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        }

    @staticmethod
    def _print_cv_summary(accuracies: list, f1_scores: list) -> None:
        """Display cross-validation results summary."""
        print(f"\n{'='*50}")
        print("Cross-Validation Summary")
        print(f"{'='*50}")
        print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
        print(f"Average F1 Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")

    def train_final_model(self, x: pd.Series, y: pd.Series) -> None:
        """Train final model on complete dataset with best hyperparameters."""
        print("\nTraining final model on full dataset...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
            ('svm', LinearSVC(class_weight='balanced', C=1, max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE))
        ], memory=None)
        self.pipeline.fit(x, y)
        self._save_model()

    def _save_model(self) -> None:
        """Persist trained pipeline to disk."""
        joblib.dump(self.pipeline, MODEL_FILE)
        print(f"Final model saved to {MODEL_FILE}")

    @staticmethod
    def _create_pipeline() -> Pipeline:
        """Create a new sklearn pipeline for grid search."""
        return Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svm', LinearSVC(class_weight='balanced', max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE))
        ], memory=None)


def get_param_grid() -> Dict:
    """Define hyperparameter search space."""
    return {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_features': [5000, 10000],
        'svm__C': [0.5, 1.0]
    }


def main() -> None:
    """Execute complete model training pipeline."""
    start_time = time.time()

    trainer = SentimentModelTrainer()
    x, y = trainer.load_and_prepare_data()
    trainer.compute_class_weights(y)

    param_grid = get_param_grid()
    trainer.cross_validate(x, y, param_grid)
    trainer.train_final_model(x, y)

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\n{'='*50}")
    print(f"Total training time: {elapsed_minutes:.2f} minutes")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
