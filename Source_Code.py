import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import BorderlineSMOTE


# === 1. Load and combine datasets ===
def load_weather_data():
    files = [
        'Predict-weather-using-random-forest/datasets/hanoi 2016-2018.csv',
        'Predict-weather-using-random-forest/datasets/hanoi 2018-2020.csv',
        'Predict-weather-using-random-forest/datasets/hanoi 2020-2022.csv',
        'Predict-weather-using-random-forest/datasets/hanoi 2022-2025.csv'
    ]
    df_list = [pd.read_csv(file) for file in files]
    df = pd.concat(df_list, ignore_index=True)

    # Convert datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day

    fill_columns = ['windgust', 'solarradiation', 'solarenergy', 'uvindex']
    for col in fill_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in df.columns:
        if df[col].dtype != object and col != 'datetime':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df


# === 3. Main Model Training Function ===
def model_training():
    df = load_weather_data()

    # Define feature and target columns
    target_column = 'conditions'
    feature_columns = [
        'temp', 'humidity', 'precip', 'precipcover', 'windspeed', 'cloudcover', 'dew',
        'sealevelpressure', 'solarradiation', 'visibility',
        'year', 'month', 'day'
    ]

    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(df[target_column])

    # Feature matrix
    X = df[feature_columns]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to training data
    Borderline = BorderlineSMOTE(random_state=42, kind='borderline-1')
    X_train_res, y_train_res = Borderline.fit_resample(X_train, y_train)

    # 4. Train Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_res, y_train_res)
    y_pred = rf.predict(X_test)

    # Return useful components
    return rf, feature_columns, target_column, le, X_test, y_test
