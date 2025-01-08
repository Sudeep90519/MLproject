import os,sys,dill,pickle
import numpy as np,pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try :
        
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_object:
            pickle.dump(obj,file_object)
    
    except Exception as e:
        raise CustomException(e,sys)
        
        
def evaluate_model(X_train:np.array ,y_train:np.array ,X_test:np.array ,y_test: np.array,models:dict):
    try :
        train_report,test_report ={},{}
        
        for i in range (len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            #y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            #train_model_score =r2_score(y_train,y_train_pred) 
            test_model_score=r2_score(y_test,y_test_pred)
            
            #train_report[list(models.keys())[i]] = train_model_score
            test_report[list(models.keys())[i]] = test_model_score
            
        ## sorting the dictionary in reverse order to get the best model on top
        #train_report = sorted(train_report.values(),reverse=True)
        test_report = sorted(test_report.items(),key=lambda x: x[1],reverse=True)
        
        return test_report
    

    except Exception as e:
        raise CustomException(e,sys)
               
               
def load_object(file_path):
    
    try:
        with open (file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys) 