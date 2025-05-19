import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from imblearn.over_sampling import BorderlineSMOTE

# 1. Load and combine datasets
def load_weather_data():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, 'datasets-main', 'datasets-main')
    
    # List of dataset files
    files = [
        os.path.join(datasets_dir, 'hanoi 2016-2018.csv'),
        os.path.join(datasets_dir, 'hanoi 2018-2020.csv'),
        os.path.join(datasets_dir, 'hanoi 2020-2022.csv'),
        os.path.join(datasets_dir, 'hanoi 2022-2025.csv')
    ]
    
    # Read and combine all datasets
    dfs = []
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
        else:
            print(f"Warning: File not found: {file}")
    
    if not dfs:
        raise FileNotFoundError("No dataset files found. Please check the dataset directory.")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert datetime column to datetime type
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
    
    # Extract time-based features
    combined_df['month'] = combined_df['datetime'].dt.month
    combined_df['day'] = combined_df['datetime'].dt.day
    combined_df['hour'] = combined_df['datetime'].dt.hour
    
    return combined_df

# Only run the model training if this file is run directly
if __name__ == "__main__":
    # Load the data
    df = load_weather_data()

    # 2. Preprocessing
    # Select features and target
    target_column = 'conditions'  # Predicting weather conditions
    feature_columns = [
        'temp', 'humidity', 'precip', 'windspeed', 'cloudcover', 
        'sealevelpressure', 'solarradiation', 'uvindex',
        'month', 'day', 'hour'  # Include time-based features here
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
    from sklearn.metrics import confusion_matrix
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

# 7. Demo
# import ipywidgets as widgets
# from IPython.display import display, clear_output

# condition_classes = le.classes_

# input_widgets = {}
# for col in feature_columns:
#     if col in ['year', 'month', 'day']:
#         input_widgets[col] = widgets.IntText(
#             description=f'{col}:',
#             disabled=False
#         )
#     else:
#         input_widgets[col] = widgets.FloatText(
#             description=f'{col}:',
#             disabled=False
#         )

# predict_button = widgets.Button(description="Predict Weather")

# output_widget = widgets.Output()

# def on_predict_button_clicked(b):
#     with output_widget:
#         clear_output(wait=True)
#         try:
#             input_data = {}
#             for col in feature_columns:
#                 input_data[col] = input_widgets[col].value

#             input_df = pd.DataFrame([input_data])

#             input_df = input_df[feature_columns]

#             prediction_encoded = rf.predict(input_df)

#             prediction_decoded = le.inverse_transform(prediction_encoded)

#             print(f"Predicted Weather Condition: {prediction_decoded[0]}")

#         except Exception as e:
#             print(f"An error occurred: {e}")

# predict_button.on_click(on_predict_button_clicked)

# # Arrange widgets in a VBox
# input_box = widgets.VBox(list(input_widgets.values()))
# ui = widgets.VBox([input_box, predict_button, output_widget])

# # Display the UI
# display(ui)
