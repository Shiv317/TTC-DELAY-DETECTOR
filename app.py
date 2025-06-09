import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import pydeck as pdk
import plotly.express as px
import xgboost as xgb
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("model.pkl")
df = pd.read_csv("ttc_final_dataset.csv")

# Define model-trained features
feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_peak'] + \
               [col for col in df.columns if col.startswith('line_') or col.startswith('cause_category_')]

# Set up Streamlit UI
st.title("🚇 TTC Subway Delay Risk App")
st.subheader("Predict Subway Delays + Visualize Risk + Understand Causes")

# UI inputs
line = st.selectbox("Select Line", sorted([col.replace('line_', '') for col in df.columns if col.startswith('line_')]))
hour = st.slider("Hour of Day", 0, 23, 8)
day = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Feature processing
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
           'Friday': 4, 'Saturday': 5, 'Sunday': 6}
day_of_week = day_map[day]
is_peak = 1 if hour in list(range(7, 11)) + list(range(16, 20)) else 0
is_weekend = 1 if day_of_week in [5, 6] else 0

# Create input vector with all zeros first
input_data = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)
input_data.at[0, 'hour'] = hour
input_data.at[0, 'day_of_week'] = day_of_week
input_data.at[0, 'is_weekend'] = is_weekend
input_data.at[0, 'is_peak'] = is_peak
if f'line_{line}' in input_data.columns:
    input_data.at[0, f'line_{line}'] = 1

# Prediction
prob = model.predict_proba(input_data)[0][1]
st.markdown(f"### 🚦 Predicted Delay Risk: **{prob:.2%}**")

# SHAP Explanation
st.subheader("🔍 Explanation of Prediction")
explainer = shap.Explainer(model)
shap_values = explainer(input_data)
shap.plots.bar(shap_values[0], show=False)
st.pyplot(plt.gcf(), clear_figure=True)

# 🧠 Natural Language Explanation from SHAP
st.subheader("🧠 Reason Behind This Risk")
impact_scores = shap_values[0].values
top_index = np.argmax(np.abs(impact_scores))
top_feature = input_data.columns[top_index]
impact = impact_scores[top_index]

# Convert feature name to readable format
readable_name = top_feature.replace("_", " ").replace("cause category", "cause").replace("line", "Line").title()

# Generate summary
if impact > 0:
    st.success(f"➕ The delay risk increased primarily due to: **{readable_name}**")
else:
    st.info(f"➖ The delay risk decreased primarily due to: **{readable_name}**")

# Map Visualization
st.subheader("🗺️ Risk Visualization on Map")
coords = {
    "YU": [43.6629, -79.3957],
    "BD": [43.6512, -79.3832],
    "SHP": [43.7315, -79.2622],
    "SRT": [43.7680, -79.4144],
}
if line in coords:
    lat, lon = coords[line]
    map_data = pd.DataFrame([{
        'lat': lat,
        'lon': lon,
        'risk': prob * 100
    }])

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=map_data,
                get_position='[lon, lat]',
                get_fill_color='[255, 0, 0, 160]',
                get_radius='risk * 100',
            )
        ]
    ))
else:
    st.warning("No map coordinates available for this line.")

# Delay Reasons Chart
st.subheader("📊 Most Common Delay Causes (Top 5)")
if 'delay_reason' in df.columns:
    delay_counts = df['delay_reason'].value_counts().head(5)
    fig = px.bar(delay_counts, x=delay_counts.index, y=delay_counts.values,
                 labels={'x': 'Cause', 'y': 'Count'},
                 title="Top 5 Delay Reasons")
    st.plotly_chart(fig)
