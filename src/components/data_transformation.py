import os,sys
import pandas as pd, numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DatatTransformation:
    def __init__(self):
        self.data_transfomation_config = DataTransformationConfig()
        
    def get_transformer_obj(self):
        
        '''
        this function is responsible for data transformation
        '''
        try:
            num_col = [
                "writing_score",
                "reading_score"
            ]
            cat_col = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"                
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean")),
                    ("scaler",StandardScaler(with_mean=False)),
                ]
            )
            
            logging.info("Numerical columns scaling completed")
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("Standard_scalar",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("categorical Columns :{cat_col}")
            logging.info("Numeric Columns :{num_col}")
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_col),
                ("cat_pipeline",cat_pipeline,cat_col)
            ]
                
            )    
                    
            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding completed")
            
            
            return preprocessor
        except Exception as e :
            raise CustomException(e,sys)
        
    
    def Initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            
            logging.info("obtaining preprocessor object")
            
            preprocessor_obj = self.get_transformer_obj()
            
            target_column_name = "math_score"
            
            num_col = ["writing_score","reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframe ")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr =preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr =np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transfomation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transfomation_config.preprocessor_obj_file_path
                )
            
            
        except Exception as e:
            raise CustomException(e,sys)                
    
