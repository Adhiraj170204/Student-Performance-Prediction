import os
import sys

import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Saves the object to a file using pandas serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)




def evaluate_models(X_train, y_train, X_test, y_test, models: dict,param):
    """
    Tunes each model with 3-fold GridSearchCV on the training set only.
    Returns (report, best_estimators): report is a DataFrame sorted by cv_r2,
    best_estimators maps model name -> gs.best_estimator_ (refit on full train set).
    Test metrics are computed for reporting only and must not be used for selection.
    """
    try:
        missing = [name for name in models if name not in param]
        assert not missing, f"No params entry for models: {missing}"

        rows = []
        best_estimators = {}

        for model_name, model in models.items():
            para = param[model_name]

            gs=GridSearchCV(model,para,cv=3,scoring='r2',n_jobs=-1)
            gs.fit(X_train,y_train)

            best_model = gs.best_estimator_
            best_estimators[model_name] = best_model

            y_train_pred=best_model.predict(X_train)
            y_test_pred=best_model.predict(X_test)

            rows.append({
                'model': model_name,
                'best_params': gs.best_params_,
                'cv_r2': gs.best_score_,
                'train_r2': r2_score(y_train,y_train_pred),
                'test_r2': r2_score(y_test,y_test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test,y_test_pred)),
                'test_mae': mean_absolute_error(y_test,y_test_pred),
            })

        report = pd.DataFrame(rows).sort_values('cv_r2', ascending=False).reset_index(drop=True)
        return report, best_estimators
    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)
