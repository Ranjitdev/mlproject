import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f'File Saved on {dir_path}')
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, X_test, y_train, y_test, models, param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.keys())[i]
            para = list(param.values())[i]
            gs = GridSearchCV(models[model], para, cv=3)
            gs.fit(X_train, y_train)
            models[model].set_params(**gs.best_params_)
            models[model].fit(X_train, y_train)

            # make prediction
            y_train_pred = models[model].predict(X_train)
            y_test_pred = models[model].predict(X_test)

            # evaluate train and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[str(model)] = test_model_score
            print(f'Model {models[model]}',
                  f'Train score {np.round(train_model_score*100, 2)}% ',
                  f'Test score {np.round(test_model_score*100, 2)}%')
            logging.info(
                f'Model {models[model]}Train score {train_model_score}% Test score {test_model_score}%'
            )
        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)