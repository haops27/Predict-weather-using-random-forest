import streamlit as st
import pandas as pd
from Source_Code import model_training

# Load model once
@st.cache_resource
def load_model():
    rf, feature_columns, target_column, le, _, _ = model_training()
    return rf, feature_columns, le

rf, feature_columns, le = load_model()

# Title
st.title("🌤️ Hanoi Weather Condition Predictor")
st.markdown("Upload a CSV or fill in values manually to get weather condition predictions.")

# Upload option
uploaded_file = st.file_uploader("📁 Upload CSV file with weather data", type=["csv"])

# If file uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Extract year/month/day if datetime exists
    if 'datetime' in df.columns:
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

    # Check required columns
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
    else:
        st.success("✅ File looks good! Generating predictions...")

        predictions = rf.predict(df[feature_columns])
        decoded = le.inverse_transform(predictions)
        df['Predicted Condition'] = decoded

        # ✅ Toggle display mode
        view_mode = st.radio("📊 Choose display mode:", ["Full Table", "Only Date + Prediction"])

        if view_mode == "Full Table":
            st.dataframe(df)
        else:
            if 'datetime' in df.columns:
                df_display = df[["datetime", "Predicted Condition", "conditions"]].copy()
                df_display["datetime"] = df_display["datetime"].dt.strftime("%Y-%m-%d")  # Format date only
                st.dataframe(df_display)
            else:
                st.warning("⚠️ 'datetime' column not found. Showing predictions only.")
                st.dataframe(df[["Predicted Condition"]])

        # ✅ Download button
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Predictions as CSV", csv, "weather_predictions.csv", "text/csv")

else:
    # Manual entry form
    st.subheader("🔢 Or enter values manually")
    with st.form("manual_form"):
        inputs = {}
        for col in feature_columns:
            if col == 'year':
                inputs[col] = st.number_input("Year", min_value=0, max_value=2025, value=0)
            elif col == 'month':
                inputs[col] = st.number_input("Month", min_value=0, max_value=12, value=0)
            elif col == 'day':
                inputs[col] = st.number_input("Day", min_value=0, max_value=31, value=0)
            else:
                inputs[col] = st.number_input(f"{col.title()}", value=0.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([inputs])[feature_columns]
        prediction_encoded = rf.predict(input_df)
        prediction_decoded = le.inverse_transform(prediction_encoded)
        st.success(f"🌦️ Predicted Weather Condition: **{prediction_decoded[0]}**")
