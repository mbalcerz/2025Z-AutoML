import importlib
import inspect
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


class MiniAutoML:
    def __init__(self, models_config):
        self.models_config = models_config
        self.best_model = None
        self.preprocessor = None
        self.NUMBER_OF_BEST_MODELS = 5
        self.feature_names_ = None

    def _build_preprocessor(self, y_train=None):
        num_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median')),
            ('scale', RobustScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('target_enc', TargetEncoder(smooth='auto'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric_preprocessing",
                    num_pipeline,
                    make_column_selector(dtype_include=np.number),
                ),
                (
                    "categorical_preprocessing",
                    cat_pipeline,
                    make_column_selector(dtype_include=['object', 'category']),
                ),
            ],
            remainder="passthrough",
        )

        return preprocessor

    def _evaluate_model(self, config, X, y):
        module_path, class_name = config['class'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        params = config['params'].copy()
        if 'random_state' not in params and hasattr(model_class, '__init__'):
            sig = inspect.signature(model_class.__init__)
            if 'random_state' in sig.parameters:
                params['random_state'] = 42

        model = model_class(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=1
        )

        return {
            'name': config['name'],
            'model_class': model_class,
            'params': params,
            'score': np.mean(scores)
        }

    def fit(self, X_train, y_train):
        self.preprocessor = self._build_preprocessor(y_train)
        X_train_processed = self.preprocessor.fit_transform(X_train, y_train)
        
        if isinstance(X_train_processed, np.ndarray):
            n_features = X_train_processed.shape[1]
            self.feature_names_ = [f'feature_{i}' for i in range(n_features)]
            X_train_processed = pd.DataFrame(
                X_train_processed,
                columns=self.feature_names_,
                index=X_train.index if hasattr(X_train, 'index') else None
            )

        if hasattr(y_train, 'values'):
            y_train = y_train.values.ravel()

        with tqdm_joblib(tqdm(desc="Evaluating models",
                              total=len(self.models_config))):
            results = Parallel(n_jobs=-1)(
                delayed(self._evaluate_model)(
                    config, X_train_processed, y_train)
                for config in self.models_config
            )

        results.sort(key=lambda x: x['score'], reverse=True)
        best_models = results[:self.NUMBER_OF_BEST_MODELS]

        estimators = []
        for m in best_models:
            params = m['params'].copy()
            if 'random_state' not in params and hasattr(m['model_class'], '__init__'):
                sig = inspect.signature(m['model_class'].__init__)
                if 'random_state' in sig.parameters:
                    params['random_state'] = 42
            estimators.append((m['name'], m['model_class'](**params)))

        print("\nTraining final Stacking Ensemble...")
        self.best_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            cv=5,
            n_jobs=-1,
            passthrough=True
        )
        self.best_model.fit(X_train_processed, y_train)
        print("Model is trained and ready for predictions.")
        return self.best_model

    def predict(self, X_test):
        X_test_processed = self.preprocessor.transform(X_test)
        if isinstance(X_test_processed, np.ndarray):
            X_test_processed = pd.DataFrame(
                X_test_processed,
                columns=self.feature_names_,
                index=X_test.index if hasattr(X_test, 'index') else None
            )
        return self.best_model.predict(X_test_processed)

    def predict_proba(self, X_test):
        X_test_processed = self.preprocessor.transform(X_test)
        if isinstance(X_test_processed, np.ndarray):
            X_test_processed = pd.DataFrame(
                X_test_processed,
                columns=self.feature_names_,
                index=X_test.index if hasattr(X_test, 'index') else None
            )
        return self.best_model.predict_proba(X_test_processed)

    def print_best_models(self):
        if self.best_model is None:
            print("No model has been trained yet.")
            return
        print("Best models in the ensemble:")
        for name, estimator in self.best_model.estimators:
            print(f"- {name}: {estimator}")
