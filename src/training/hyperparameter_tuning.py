"""Hyperparameter tuning module using GridSearchCV / RandomizedSearchCV."""

from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


class HyperparameterTuner:
    """Finds optimal hyperparameters using stratified cross-validation."""

    def __init__(self, cv: int = 5, scoring: str = "f1_weighted", n_jobs: int = -1):
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def get_search_spaces(self) -> Dict[str, Dict[str, list]]:
        """Return hyperparameter search grids for baseline models."""
        return {
            "logistic_regression": {
                "C": [0.01, 0.1, 1.0, 5.0, 10.0],
                "solver": ["lbfgs", "liblinear"]
            },
            "svm": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"]
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            },
            "knn": {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"],
                "metric": ["cosine", "euclidean"]
            },
            "naive_bayes": {
                "alpha": [0.1, 0.5, 1.0, 2.0]
            }
        }

    def tune_model(self, model_name: str, X_train, y_train) -> Dict[str, Any]:
        """Run grid search for a designated model."""
        grids = self.get_search_spaces()
        if model_name not in grids:
            raise ValueError(f"No parameter grid defined for {model_name}")

        estimators = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "svm": SVC(probability=True, random_state=42),
            "random_forest": RandomForestClassifier(random_state=42, n_jobs=self.n_jobs),
            "knn": KNeighborsClassifier(n_jobs=self.n_jobs),
            "naive_bayes": MultinomialNB()
        }

        base_estimator = estimators[model_name]
        param_grid = grids[model_name]

        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True
        )

        grid_search.fit(X_train, y_train)

        return {
            "best_estimator": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_)
        }
