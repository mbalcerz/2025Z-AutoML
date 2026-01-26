import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier


class MiniAutoML:
    def __init__(self, models_config='selected_models.json'):
        self.model_configs = pd.read_json(models_config)

    def fit(self, X_train, y_train):
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer()),
            ('scaler', MinMaxScaler())
        ])
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessing_pipeline = ColumnTransformer(transformers=[
            ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
            ('cat', cat_pipeline, make_column_selector(dtype_include=np.object_))
        ])

        model_ranking = []
        for model_config in self.model_configs.to_dict(orient='records'):
            print(f"Evaluating model: {model_config['name']}")
            classifier = eval(model_config['class'])
            model_params = model_config['params']
            model = Pipeline(steps=[
                ('preprocessing', preprocessing_pipeline),
                ('classifier', classifier(**model_params))
            ])
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            model_ranking.append({'name': model_config['name'], 'class': model_config['class'], 'roc_auc': np.mean(scores)})

        model_ranking = pd.DataFrame(model_ranking)
        best_models = model_ranking.sort_values('roc_auc', ascending=False).iloc[0:5]
        self.best_model = Pipeline(steps=[
            ('preprocessing', preprocessing_pipeline),
            ('classifier', VotingClassifier(estimators=[
                (best_models['name'].values[i], eval(best_models['class'].values[i])(**self.model_configs[self.model_configs['name'] == best_models['name'].values[i]]['params'].values[0]))
                for i in range(len(best_models))
            ], voting='soft'))
        ])
        self.best_model.fit(X_train, y_train)


    def predict(self, X_test):
        return self.best_model.predict(X_test)
    

    def predict_proba(self, X_test):
        return self.best_model.predict_proba(X_test)
