import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import(AdaBoostClassifier,GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor)
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
#Model Training

class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting data into traintest split")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            #X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models={"LinearRegression":LinearRegression(),
                    "Lasso":Lasso(),
                    "Ridge":Ridge(),
                    "K-Neighbors_Regressor":KNeighborsRegressor(),
                    "DecisionTree":DecisionTreeRegressor(),
                    "RandomForestRegressor":RandomForestRegressor(),
                    "XGBRegressor":XGBRegressor(),
                    "CatBoostingRegressor":CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor":AdaBoostRegressor()
                    }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # to get best model from the dictionary
            best_model_score=max(sorted(model_report.values()))

            #to get best model name from the dictionary
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No bestmodel found")
            logging.info(f"Best model found on training and test dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2square=r2_score(y_test,predicted)
            return r2square

        except Exception as e:
            raise CustomException(e,sys)
