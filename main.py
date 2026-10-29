'''
In this file we are going to load the data and pass the data to required
functions for training the ML Algorithm
'''
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging("main")  # calling the function from log_code file
from sklearn.model_selection import train_test_split
from random_sample_technique import clearn_null_values
from yeo_timing import variable_transformation_outliers
from fs import select_best_columns
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler  # Z_score : xi - mean / std
from all_models import common

class CREDIT_CARD_PROJECT:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f"The Number of rows and columns was : {self.df.shape}")
            logger.info(f"NUll values in the values  : {self.df.isnull().sum()}")
            self.df = self.df.drop([150000 , 150001],axis=0)
            self.df = self.df.drop(['MonthlyIncome.1'],axis=1)
            logger.info(f"NUll values in the values  : {self.df.isnull().sum()}")
            self.X = self.df.iloc[: , :-1] # all indepent columns
            self.y = self.df.iloc[: , -1]  # all dependent columns
            self.y = self.y.map({'Good':1 , 'Bad':0}).astype(int)
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X , self.y,test_size=0.2 , random_state=42)
            logger.info(f"Training dataset size : \n {self.X_train.shape}  => {self.y_train.shape}")
            logger.info(f"Testing dataset size : \n {self.X_test.shape} => {self.y_test.shape}")
        except Exception as e:
            er_type,er_msg,er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def handle_missing_values(self):
        try:
            logger.info(f"Before giving to function  : {self.X_train.isnull().sum()}")
            logger.info(f"Before giving to function  : {self.X_test.isnull().sum()}")
            self.X_train,self.X_test = clearn_null_values(self.X_train , self.X_test)
            logger.info(f"After giving to function  : {self.X_train.isnull().sum()}")
            logger.info(f"After giving to function  : {self.X_test.isnull().sum()}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def spliting_data_clearlly(self):
        try:
            self.X_train_numerical = self.X_train.select_dtypes(exclude='object')
            self.X_train_categorical = self.X_train.select_dtypes(include="object")
            self.X_test_numerical = self.X_test.select_dtypes(exclude='object')
            self.X_test_categorical = self.X_test.select_dtypes(include="object")

            logger.info(f"Train Numerical columns : {self.X_train_numerical.shape} : \n : {self.X_test_numerical.columns}")
            logger.info(f"Test Numerical columns : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
            logger.info(f"Train Categorical columns : {self.X_train_categorical.shape} : \n : {self.X_train_categorical.columns}")
            logger.info(f"Test Categorical columns : {self.X_test_categorical.shape} : \n : {self.X_test_categorical.columns}")


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def vt_outliers(self):
        try:
            self.X_train_numerical,self.X_test_numerical = variable_transformation_outliers(self.X_train_numerical , self.X_test_numerical)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def feature_selection(self):
        try:
           self.X_train_numerical,self.X_test_numerical = select_best_columns(self.X_train_numerical,self.X_test_numerical,self.y_train,self.y_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def cat_to_numerical(self):
        try:
            logger.info(f"X_train categorical columns and data was : {self.X_train_categorical.columns} : \n : {self.X_train_categorical.head(10)}")
            logger.info(
                f"X_test categorical columns and data was : {self.X_test_categorical.columns} : \n : {self.X_test_categorical.head(10)}")

            # OneHotEncoding into Nominal columns [Gender , Region]
            one_hot_obj = OneHotEncoder(drop='first')
            one_hot_obj.fit(self.X_train_categorical[['Gender','Region']])
            result = one_hot_obj.transform(self.X_train_categorical[['Gender','Region']])
            result_2 = one_hot_obj.transform(self.X_test_categorical[['Gender','Region']])
            result = result.toarray()
            result_2 = result_2.toarray()
            t = pd.DataFrame(data = result , columns=one_hot_obj.get_feature_names_out())
            t2 = pd.DataFrame(data=result_2, columns=one_hot_obj.get_feature_names_out())
            self.X_train_categorical.reset_index(drop=True,inplace=True)
            self.X_test_categorical.reset_index(drop=True, inplace=True)
            t.reset_index(drop=True , inplace=True)
            t2.reset_index(drop=True, inplace=True)
            self.X_train_categorical = pd.concat([self.X_train_categorical,t],axis=1)
            self.X_test_categorical = pd.concat([self.X_test_categorical, t2], axis=1)
            self.X_train_categorical = self.X_train_categorical.drop(['Gender','Region'],axis=1)
            self.X_test_categorical = self.X_test_categorical.drop(['Gender', 'Region'], axis=1)

            logger.info(
                f"X_train categorical columns and data was After One Hot: {self.X_train_categorical.columns} : \n : {self.X_train_categorical.head(10)}")
            logger.info(
                f"X_test categorical columns and data was After One Hot : {self.X_test_categorical.columns} : \n : {self.X_test_categorical.head(10)}")


            # Odinal Encoding for Rented House | Occupation and Education columns
            ordinal_obj = OrdinalEncoder()
            ordinal_obj.fit(self.X_train_categorical[['Rented_OwnHouse', 'Occupation', 'Education']])
            result_3 = ordinal_obj.transform(self.X_train_categorical[['Rented_OwnHouse', 'Occupation', 'Education']])
            result_4 = ordinal_obj.transform(self.X_test_categorical[['Rented_OwnHouse', 'Occupation', 'Education']])

            p1 = pd.DataFrame(data = result_3 , columns=ordinal_obj.get_feature_names_out()+"_res")
            p2 = pd.DataFrame(data = result_4 , columns=ordinal_obj.get_feature_names_out()+"_res")

            self.X_train_categorical.reset_index(drop=True, inplace=True)
            self.X_test_categorical.reset_index(drop=True, inplace=True)
            p1.reset_index(drop=True, inplace=True)
            p2.reset_index(drop=True, inplace=True)

            self.X_train_categorical = pd.concat([self.X_train_categorical,p1],axis=1)
            self.X_test_categorical = pd.concat([self.X_test_categorical , p2],axis=1)

            self.X_train_categorical = self.X_train_categorical.drop(['Rented_OwnHouse', 'Occupation', 'Education'],axis=1)
            self.X_test_categorical = self.X_test_categorical.drop(['Rented_OwnHouse', 'Occupation', 'Education'],axis=1)

            logger.info(
                f"X_train categorical columns and data was After Odinal : {self.X_train_categorical.columns} : \n : {self.X_train_categorical.head(10)}")
            logger.info(
                f"X_test categorical columns and data was After Odinal : {self.X_test_categorical.columns} : \n : {self.X_test_categorical.head(10)}")
            logger.info(f"Train data : {self.X_train_categorical.isnull().sum()}")
            logger.info(f"Test data : {self.X_test_categorical.isnull().sum()}")

            self.X_train_numerical.reset_index(drop = True , inplace=True)
            self.X_test_numerical.reset_index(drop=True, inplace=True)
            self.X_train_categorical.reset_index(drop=True, inplace=True)
            self.X_test_categorical.reset_index(drop=True, inplace=True)

            self.final_training_data = pd.concat([self.X_train_numerical , self.X_train_categorical],axis=1)
            self.final_testing_data = pd.concat([self.X_test_numerical , self.X_test_categorical],axis=1)

            logger.info("=====================================================")
            logger.info(f"===========Final Train Data============================= : \n : {self.final_training_data.shape} : \n : {self.final_training_data.columns} : \n : {self.final_training_data.head(10)} : \n : {self.final_training_data.isnull().sum()}")
            logger.info(
                f"===========Final Test Data============================= : \n : {self.final_testing_data.shape} : \n : {self.final_testing_data.columns} : \n : {self.final_testing_data.head(10)} : \n : {self.final_testing_data.isnull().sum()}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def data_balancing(self):
        try:
            logger.info(f"Total Final Training data shape : {self.y_train.shape}")
            logger.info(f"Number of Rows for Good :{1} : Class : {sum(self.y_train==1)}")
            logger.info(f"Number of Rows for Bad :{0} : Class : {sum(self.y_train == 0)}")
            sm_obj = SMOTE(random_state=42)
            self.X_train_bal,self.y_train_bal = sm_obj.fit_resample(self.final_training_data , self.y_train)
            logger.info(f"After Balancing Training data shape : {self.y_train_bal.shape}")
            logger.info(f"Number of Rows for Good :{1} : Class : {sum(self.y_train_bal == 1)}")
            logger.info(f"Number of Rows for Bad :{0} : Class : {sum(self.y_train_bal == 0)}")
            # lets scale down the values
            sc = StandardScaler()
            sc.fit(self.X_train_bal)
            self.X_train_bal_scaled = sc.transform(self.X_train_bal)
            self.X_test_scaled = sc.transform(self.final_testing_data)

            # know we can give the indepdent data and dependent data to ML Algorithms
            # Training_variables : (self.X_train_bal_scaled , self.y_train_bal)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def train_all_models(self):
        try:
            common(self.X_train_bal_scaled , self.y_train_bal , self.X_test_scaled,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")


if __name__ == "__main__":
    try:
        obj = CREDIT_CARD_PROJECT("creditcard.csv")
        obj.handle_missing_values()
        obj.spliting_data_clearlly()
        obj.vt_outliers()
        obj.feature_selection()
        obj.cat_to_numerical()
        obj.data_balancing()
        obj.train_all_models()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")






