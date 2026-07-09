# California Housing Price Prediction

This project predicts California housing prices using machine learning. I built it to practice the complete machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, and making predictions on new data.

## What I learned from this project

- Data cleaning and preprocessing using Scikit-Learn pipelines
- Creating custom transformers for feature engineering
- Comparing multiple machine learning models
- Cross-validation and hyperparameter tuning
- Saving and loading models with Joblib
- Building an end-to-end prediction pipeline

## Dataset

The project uses the California Housing dataset.

Target variable:

- `median_house_value`

Features include:

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- ocean_proximity

## Feature Engineering

I created three additional features:

- `rooms_per_household`
- `population_per_household`
- `bedrooms_per_room`

These features are generated automatically during both training and inference using a custom transformer.

## Models Tried

I experimented with three different models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Cross-Validation Results

| Model | RMSE |
|-------|-------|
| Linear Regression | 68,439 |
| Decision Tree | 71,268 |
| Random Forest | 50,209 |

The Random Forest model performed the best and was selected as the final model.

## Hyperparameter Tuning

I also experimented with `GridSearchCV` and tried different values of:

- `n_estimators`
- `max_depth`
- `min_samples_split`

Interestingly, the default Random Forest performed slightly better than the tuned versions, so I kept the default model as the final choice.

## Project Structure

```text
California-Housing-Price-Prediction/
│
├── data/
├── models/
├── src/
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How it works

```text
input.csv
    ↓
Feature Engineering
    ↓
Preprocessing Pipeline
    ↓
Random Forest Model
    ↓
Predictions
    ↓
output.csv
```

## Running the project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Make predictions:

```bash
python src/predict.py
```

## Future Improvements

Some things I'd like to add in the future:

- Feature importance visualization
- A simple web app using Streamlit
- Better experiment tracking
- Trying additional models and features

## Author

**Yuvraj Raghav**

GitHub: https://github.com/Yuvrxjr
