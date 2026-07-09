import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

def calculate_rmse(model, x ,y):
    predictions = model.predict(x)
    return root_mean_squared_error(y,predictions)

def cross_validate_model(model,x,y,cv=10):
    scores = -cross_val_score(model,x,y,scoring="neg_root_mean_squared_error",cv=cv)

    return pd.Series(scores).describe()