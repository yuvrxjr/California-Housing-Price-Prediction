import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from feature_engineering import FeatureEngineer
from preprocessing import build_pipeline

import joblib

MODEL_FILE = "models/housing_model.pkl"

if not os.path.exists(MODEL_FILE):
    
    housing = pd.read_csv("housing.csv")

    # 2. Stratified spilt

    housing['income_cat'] = pd.cut(housing['median_income'] , bins=[0.0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])


    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index, test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop('income_cat', axis = 1).to_csv("test_data.csv",index = False)
        housing = housing.loc[train_index].drop('income_cat', axis =1)

    train_test = housing.copy()
    housing_labels = train_test['median_house_value']
    housing_features = train_test.drop('median_house_value',axis=1)

    fe = FeatureEngineer()
    housing_prepared = fe.transform(housing_features)

    num_attribs = housing_features.drop('ocean_proximity', axis=1).columns.to_list()
    cat_attribs = ['ocean_proximity']

    # 5. Transforming the Data or Preprocessing
    pipeline = build_pipeline(num_attribs,cat_attribs)

    full_model = Pipeline([
        ('Feature_engineering',FeatureEngineer()),
        ('preprocessor',pipeline),
        ('model', RandomForestRegressor(random_state=42))
        
    ])

    full_model.fit(housing_features,housing_labels)

    joblib.dump(full_model,MODEL_FILE)

    print("Model trained and Saved")