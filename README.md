# Predict weather pattern using Random Forest
A simple Machine Learning model using Random Forest to predict weather (Random Forest Classifier)

## Overview

This project aims to develop a machine learning model that can predict weather patterns using the Random Forest algorithm and is trained on historical weather data in Hanoi city.

## Features

- Data preprocessing: duplicate handling, missing value handling, outlier detection
- Data visualization: time plot, target label analysis, correlation analysis
- Imbalanced class handling using Borderline SMOTE
- Model training and evaluation
- Streamlit application and interactive demonstration

## Files

- `Group9_Weather_Prediction.ipynb`: The main analysis and model building notebook, containing the pipeline our group has followed.
- `Source_Code.py`: A shortened and simplified version of the notebook for easier understanding.
- `main.py`: Launches the streamlit demo
- `requirements.txt`: Python libraries and packages required to run the model
- `datasets-main`: Contains the datasets we have used to train our model

## How to run
1. Open terminal and clone our GitHub repository: https://github.com/haops27/Predict-weather-using-random-forest.git and go to the file containing our repo: `cd path/to/Predict-weather-using-random-forest`
2. Install the required packages: `pip install -r requirements.txt`. Make sure that you have installed Python 3.13 on your computer.
3. Run the Streamlit application: `streamlit run main.py`. If you cannot run it, try `python -m streamlit run main.py`.

## Group 9
 - Nguyễn Huy Diễn – 20235910	Group leader, overseeing group progress an coordination, managing code implementation and integration
- Nguyễn Tiến Đạt – 20239715	Data collection and preprocessing, demo UI support
- Trần Trung Hiếu – 20235934	Data visualization, preprocessing support
- Phạm Song Hào – 20235930	    Model research, building, training, demo UI building
- Nguyễn Nhật Anh – 20235892	Model evaluation

## References

https://en.wikipedia.org/wiki/Weather_forecasting
https://www.visualcrossing.com/resources/documentation/weather-data/weather-data-documentation/
https://www.geeksforgeeks.org/decision-tree/
https://www.youtube.com/watch?v=7P6yYhcSuPc
https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/
https://www.youtube.com/watch?v=BmoNAptI1nI
https://users.soict.hust.edu.vn/khoattq/ml-dm-course/L7-Random-forests.pdf
https://www.geeksforgeeks.org/smote-for-imbalanced-classification-with-python/
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html