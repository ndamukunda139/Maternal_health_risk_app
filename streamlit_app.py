import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_shap import st_shap

# Set page configuration
st.set_page_config(
    page_title="Maternal health risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e63946;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1d3557;
        margin-bottom: 0.5rem;
    }
    .info-text {
        background-color: #f1faee;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #a8dadc;
        color: #1d3557;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        color: #856404;
    }
    .stButton>button {
        background-color: #457b9d;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1d3557;
    }
    /* Improve general text visibility */
    p, h1, h2, h3, label {
        color: #1d3557;
    }
    /* Improve form field visibility */
    .stNumberInput, .stSelectbox {
        background-color: #f1faee !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model from disk"""
    try:
        with open('maternal_health_risk_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'maternal_health_risk_model.pkl' is in the current directory.")
        return None

@st.cache_data
def load_feature_info():
    """Return information about features for the prediction form"""
    return {
        'Age': {'type': 'number', 'min': 20, 'max': 100, 'help': 'Patient age in years'},
        'SystolicBP': {'type': 'number', 'min': 60, 'max': 200, 'help': 'Systolic blood pressure in mm Hg'},
        'DiastolicBP': {'type': 'number', 'min': 40, 'max': 120, 'help': 'Diastolic blood pressure in mm Hg'},
        'BS': {'type': 'number', 'min': 4, 'max': 20, 'help': 'Blood glucose in mmol/L'},
        'BodyTemp': {'type': 'number', 'min': 95, 'max': 105, 'help': 'Body temperature in F'},
        'HeartRate': {'type': 'number', 'min': 5, 'max': 100, 'help': 'Heart rate in bpm'}
}

def preprocess_input(input_data, model):
    """Process input data to match model's expected format"""
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Add missing columns that the model expects
    input_df['BS'] = 0 
    input_df['BodyTemp'] = 0
    input_df['HeartRate'] = 0 # Default value for prediction target

    # Create engineered features similar to training

    # Create age groups
    input_df['Age_Group'] = pd.cut(input_df['Age'],
        bins=[0, 20, 30, 40, 50, 60, 70, 100],
        labels=['<20','20-30', '30-40', '40-50', '50-60', '60-70', '70+'])

    return input_df

def get_prediction_probability(model, input_df):
    """Get prediction and probability from the model"""
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    result = {
        'prediction': 'high risk' if prediction == 0 else 'mid risk' if prediction == 1 else 'low risk',
        'probability': probability[2] if prediction == 2 else probability[1] if prediction == 1 else probability[0],
        'probability_high_risk': probability[0],
        'probability_mid_risk': probability[1],
        'probability_row_risk': probability[2]
    }

    return result

def generate_random_patient():
    """Generate random but realistic patient data"""
    import random

    # Get feature info to use the appropriate options
    feature_info = load_feature_info()

    # Generate random patient data
    random_patient = {
        'Age': random.randint(30, 85),
        'SystolicBP': random.randint(60, 200),
        'DiastolicBP': random.randint(40, 120),
        'BS': random.randint(4, 20),
        'BodyTemp': random.randint(95, 105),
        'HeartRate': random.randint(0, 100)
    }

    return random_patient

def generate_sample_data():
    """Generate sample dataset for EDA visualizations"""
    import random

    # Create a sample dataset with 100 patients
    sample_size = 100
    feature_info = load_feature_info()

    data = {
        'Age': [random.randint(30, 85) for _ in range(sample_size)],
        'SystolicBP': [random.randint(60, 200) for _ in range(sample_size)],
        'DiastolicBP': [random.randint(40, 120) for _ in range(sample_size)],
        'BS': [random.randint(4, 20) for _ in range(sample_size)],
        'BodyTemp': [random.randint(95, 105) for _ in range(sample_size)],
        'HeartRate': [random.randint(0, 100) for _ in range(sample_size)],
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def main():
    """Main function to run the Streamlit app"""
    st.markdown("<h1 class='main-header'>Maternal health risk Prediction</h1>", unsafe_allow_html=True)

    # Create tabs for different app sections
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "EDA", "Information", "About"])

    with tab1:
        st.markdown("<div class='sub-header'>Mother Information</div>", unsafe_allow_html=True)

        # Load model
        model = load_model()
        if model is None:
            st.stop()

        # Load feature info
        feature_info = load_feature_info()

        # Add sample patient button outside the form
        if st.button("Load Random Sample Mother Info"):
            st.session_state.input_data = generate_random_patient()
            st.rerun()

        # Create form for user input
        with st.form(key='prediction_form'):
            cols = st.columns(3)

            # Form for collecting patient data
            input_data = {}

            # If we have stored session state data, use it
            if hasattr(st.session_state, 'input_data'):
                input_data = st.session_state.input_data

            for i, (feature, info) in enumerate(feature_info.items()):
                col_idx = i % 3

                with cols[col_idx]:
                    if info['type'] == 'number':
                        default_value = input_data.get(feature, info.get('default', info.get('min', 0)))
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            min_value=info.get('min', 0),
                            max_value=info.get('max', 100),
                            value=default_value,
                            help=info.get('help', '')
                        )
                    elif info['type'] == 'select':
                        default_index = 0
                        if feature in input_data:
                            try:
                                default_index = info['options'].index(input_data[feature])
                            except ValueError:
                                default_index = 0

                        input_data[feature] = st.selectbox(
                            f"{feature}",
                            options=info['options'],
                            index=default_index,
                            help=info.get('help', '')
                        )

            # Submit button
            submit_button = st.form_submit_button(label="Predict maternal risk")

        # Process the form
        if submit_button:
            with st.spinner('Processing...'):
                # Preprocess input data
                processed_input = preprocess_input(input_data, model)

                # Get prediction
                result = get_prediction_probability(model, processed_input)

                # Display result
                st.markdown("<div class='sub-header'>Prediction Result</div>", unsafe_allow_html=True)

                # Create a two-column layout for the results
                col1, col2 = st.columns([1, 1])

                with col1:
                    if result['prediction'] == 'row risk':
                        st.markdown(f"""
                        <div class='success-box'>
                            <h3>Prediction: {result['prediction']}</h3>
                            <p>The model predicts that the patient has low maternal health risk.</p>
                            <p>Confidence: {result['probability_row_risk']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    if result['prediction'] == 'mid risk':
                        st.markdown(f"""
                        <div class='success-box'>
                            <h3>Prediction: {result['prediction']}</h3>
                            <p>The model predicts that the patient has mid maternal health risk.</p>
                            <p>Confidence: {result['probability_mid_risk']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='warning-box'>
                            <h3>Prediction: {result['prediction']}</h3>
                            <p>The model predicts a higher risk of maternal health.</p>
                            <p>Confidence: {result['probability_high_risk']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    # Create a gauge chart showing probability with improved colors
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=float(result['probability_high_risk']),
                        title={'text': "Mortality Risk", 'font': {'color': '#1d3557'}},
                        gauge={
                            'axis': {'range': [0, 1], 'tickcolor': '#1d3557'},
                            'bar': {'color': "rgba(0, 0, 0, 0)"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "#a8dadc"},
                                {'range': [0.3, 0.7], 'color': "#457b9d"},
                                {'range': [0.7, 1], 'color': "#e63946"}
                            ],
                            'threshold': {
                                'line': {'color': "#1d3557", 'width': 4},
                                'thickness': 0.75,
                                'value': float(result['probability_high_risk'])
                            }
                        }
                    ))

                    fig.update_layout(height=250, font={'color': '#1d3557'})
                    st.plotly_chart(fig, use_container_width=True)

                # Display warning disclaimer
                st.markdown("""
                <div class='info-text'>
                    <p><strong>Disclaimer:</strong> This prediction is based on a machine learning model and should be
                    used for informational purposes only. Always consult with healthcare professionals for
                    medical decisions.</p>
                </div>
                """, unsafe_allow_html=True)


    with tab3:
        st.markdown("<div class='sub-header'>Maternal Health risk Information</div>", unsafe_allow_html=True)
        st.markdown("""
        ### About Maternal health

        Maternal health risk is one of the most common issue diagnosed in women. Several factors can influence the high risk
        including:

        - **Age**: Patient's age at diagnosis
        - **Tumor Size**: The size of the tumor in millimeters
        - **Lymph Node Status**: Whether cancer has spread to lymph nodes
        - **Hormone Receptor Status**: Whether cancer cells have estrogen or progesterone receptors
        - **Cancer Stage**: The overall stage of cancer progression
        - **Grade**: How abnormal the cancer cells look under a microscope

        ### Features Used in Prediction

        Our model uses the following key features to make predictions:

        - **Patient Demographics**: Age, race, and marital status
        - **Cancer Characteristics**: Tumor size, grade, stage, differentiation
        - **Biological Markers**: Estrogen status, progesterone status
        - **Lymph Node Status**: Number of nodes examined and positive nodes

        ### Model Performance

        The machine learning model was trained on historical patient data with known outcomes. The model achieves:

        - Accuracy: ~88%
        - Precision: ~89%
        - Recall: ~88%
        - ROC-AUC: ~97%

        These metrics indicate good but not perfect predictive ability. Always consult healthcare professionals for
        medical decisions.
        """)

        # Add visualizations
        st.markdown("### Key Survival Factors")

        # Example visualization (would use actual data in production)
        col1, col2 = st.columns(2)

        
        with col2:
            # Dummy data for visualization
            data = pd.DataFrame({
                'Factor': ['ER+/PR+', 'ER+/PR-', 'ER-/PR+', 'ER-/PR-'],
                'Survival Rate': [0.90, 0.70, 0.65, 0.50]
            })

            fig = px.bar(data, x='Factor', y='Survival Rate',
                      color='Survival Rate', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig)

    with tab4:
        st.markdown("<div class='sub-header'>About This Project</div>", unsafe_allow_html=True)
        st.markdown("""
        ### Project Overview

        This web application was developed to help predict breast cancer survival based on patient and
        tumor characteristics. The underlying model was trained using machine learning techniques on
        historical patient data.

        ### How It Works

        1. **Data Collection**: Enter patient information in the form
        2. **Preprocessing**: Data is processed to match the format expected by the model
        3. **Prediction**: The model analyzes the data and provides a survival prediction
        4. **Interpretation**: Feature importance analysis helps explain what factors influenced the prediction

        ### Data Sources

        The model was trained on breast cancer patient data with known outcomes. This includes patient
        demographics, tumor characteristics, and treatment information.

        ### Limitations

        - The model makes predictions based on historical data patterns and may not account for recent medical advances
        - Individual cases can vary significantly from statistical patterns
        - This tool should be used as a supplement to, not a replacement for, professional medical advice

        ### Contact Information

        For questions or feedback about this application, please contact:

        - Email: example@example.com
        - GitHub: [github.com/username/breast-cancer-prediction](<https://github.com>)
        """)

if __name__ == "__main__":
    main()

