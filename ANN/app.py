import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

model = tf.keras.models.load_model(
    "models/breast_cancer_ann.keras"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction Using ANN")

st.markdown(
"""
Enter the patient's medical measurements and click Predict.
"""
)

tab1, tab2, tab3 = st.tabs([
    "Mean Measurements",
    "Error Measurements",
    "Worst Measurements"
])

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        mean_radius = st.number_input("Mean Radius", min_value=0.0, value=0.0)
        mean_texture = st.number_input("Mean Texture", min_value=0.0, value=0.0)
        mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, value=0.0)
        mean_area = st.number_input("Mean Area", min_value=0.0, value=0.0)
        mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, value=0.0)

    with col2:
        mean_compactness = st.number_input("Mean Compactness", min_value=0.0, value=0.0)
        mean_concavity = st.number_input("Mean Concavity", min_value=0.0, value=0.0)
        mean_concave_points = st.number_input("Mean Concave Points", min_value=0.0, value=0.0)
        mean_symmetry = st.number_input("Mean Symmetry", min_value=0.0, value=0.0)
        mean_fractal_dimension = st.number_input("Mean Fractal Dimension", min_value=0.0, value=0.0)

with tab2:

    col1, col2 = st.columns(2)

    with col1:
        radius_error = st.number_input("Radius Error", min_value=0.0, value=0.0)
        texture_error = st.number_input("Texture Error", min_value=0.0, value=0.0)
        perimeter_error = st.number_input("Perimeter Error", min_value=0.0, value=0.0)
        area_error = st.number_input("Area Error", min_value=0.0, value=0.0)
        smoothness_error = st.number_input("Smoothness Error", min_value=0.0, value=0.0)

    with col2:
        compactness_error = st.number_input("Compactness Error", min_value=0.0, value=0.0)
        concavity_error = st.number_input("Concavity Error", min_value=0.0, value=0.0)
        concave_points_error = st.number_input("Concave Points Error", min_value=0.0, value=0.0)
        symmetry_error = st.number_input("Symmetry Error", min_value=0.0, value=0.0)
        fractal_dimension_error = st.number_input("Fractal Dimension Error", min_value=0.0, value=0.0)

with tab3:

    col1, col2 = st.columns(2)

    with col1:
        worst_radius = st.number_input("Worst Radius", min_value=0.0, value=0.0)
        worst_texture = st.number_input("Worst Texture", min_value=0.0, value=0.0)
        worst_perimeter = st.number_input("Worst Perimeter", min_value=0.0, value=0.0)
        worst_area = st.number_input("Worst Area", min_value=0.0, value=0.0)
        worst_smoothness = st.number_input("Worst Smoothness", min_value=0.0, value=0.0)

    with col2:
        worst_compactness = st.number_input("Worst Compactness", min_value=0.0, value=0.0)
        worst_concavity = st.number_input("Worst Concavity", min_value=0.0, value=0.0)
        worst_concave_points = st.number_input("Worst Concave Points", min_value=0.0, value=0.0)
        worst_symmetry = st.number_input("Worst Symmetry", min_value=0.0, value=0.0)
        worst_fractal_dimension = st.number_input("Worst Fractal Dimension", min_value=0.0, value=0.0)

st.markdown("---")

if st.button("🔍 Predict Cancer Type", use_container_width=True):

    input_data = [[
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        mean_compactness,
        mean_concavity,
        mean_concave_points,
        mean_symmetry,
        mean_fractal_dimension,
        radius_error,
        texture_error,
        perimeter_error,
        area_error,
        smoothness_error,
        compactness_error,
        concavity_error,
        concave_points_error,
        symmetry_error,
        fractal_dimension_error,
        worst_radius,
        worst_texture,
        worst_perimeter,
        worst_area,
        worst_smoothness,
        worst_compactness,
        worst_concavity,
        worst_concave_points,
        worst_symmetry,
        worst_fractal_dimension
    ]]

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data, verbose=0)[0][0]

    st.markdown("## Prediction Result")

    confidence = max(
        prediction,
        1 - prediction
    ) * 100

    if prediction > 0.5:

        st.success(
            f"""
            ### ✅ Benign Tumor
            
            **Confidence:** {confidence:.2f}%
            """
        )

    else:

        st.error(
            f"""
            ### ⚠️ Malignant Tumor
            
            **Confidence:** {confidence:.2f}%
            """
        )

    st.progress(float(confidence / 100))