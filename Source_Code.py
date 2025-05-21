import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE

# 1. Load and combine datasets
def load_weather_data():
    df1 = pd.read_csv('Predict-weather-using-random-forest/datasets-main/hanoi 2016-2018.csv')
    df2 = pd.read_csv('Predict-weather-using-random-forest/datasets-main/hanoi 2018-2020.csv')
    df3 = pd.read_csv('Predict-weather-using-random-forest/datasets-main/hanoi 2020-2022.csv')
    df4 = pd.read_csv('Predict-weather-using-random-forest/datasets-main/hanoi 2022-2025.csv')
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    # Convert datetime column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Extract time-based features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    
    return df

def model_training():
    # Load the data
    df = load_weather_data()

    # 2. Preprocessing
    # Select features and target
    target_column = 'conditions'  # Predicting weather conditions
    feature_columns = [
        'temp', 'humidity', 'precip', 'precipcover', 'windspeed', 'cloudcover',
        'sealevelpressure', 'solarradiation', 'visibility',
        'year', 'month', 'day'  # Include time-based features here
    ]

    # Handle missing values
    df = df.dropna(subset=feature_columns + [target_column])

    # Encode categorical target variable
    le = LabelEncoder()
    y = le.fit_transform(df[target_column])

    # Prepare features
    X = df[feature_columns]

    # 3. Train-test split
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

    return rf, feature_columns, target_column, le, X_test, y_test

# Only run the model training if this file is run directly
if __name__ == "__main__":
    rf, feature_columns, target_column, le, X_test, y_test = model_training()

    # 5. Predictions & Evaluation
    y_pred = rf.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 6. Feature importances
    importances = pd.Series(rf.feature_importances_, index=feature_columns)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=importances.index)
    plt.title('Feature Importances in Weather Condition Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
