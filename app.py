# symptom_app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .symptom-count {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    h1 {
        color: #2d3748;
        font-weight: 700;
    }
    h3 {
        color: #4a5568;
    }
    </style>
    """, unsafe_allow_html=True)

# Paths
MODEL_PATH = "models/disease_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
TRAIN_PATH = "data/Training.csv"


# Load model and encoder with error handling
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        label_encoder = pickle.load(open(ENCODER_PATH, "rb"))
        return model, label_encoder
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure the model files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()


@st.cache_data
def load_symptoms():
    try:
        df = pd.read_csv(TRAIN_PATH)
        if "Unnamed: 133" in df.columns:
            df = df.drop(["Unnamed: 133"], axis=1)
        return df.columns[:-1].tolist()
    except FileNotFoundError:
        st.error("❌ Training data not found. Please ensure the training.csv file is in the data directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading training data: {str(e)}")
        st.stop()


model, label_encoder = load_models()
symptoms = load_symptoms()

# Header with icon and description
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🏥 Disease Prediction System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #718096; font-size: 1.1rem;'>AI-powered symptom analysis for preliminary health assessment</p>",
        unsafe_allow_html=True)

st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.markdown("""
    <div class='info-card'>
    <b>How to use:</b>
    <ol>
        <li>Select symptoms from the dropdown</li>
        <li>Add multiple symptoms for accurate prediction</li>
        <li>Click 'Analyze Symptoms' to get results</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "This tool is for educational purposes only. Always consult a healthcare professional for medical advice.")

    st.markdown("### 📊 System Info")
    st.info(f"**Available Symptoms:** {len(symptoms)}\n\n**Diseases in Database:** {len(label_encoder.classes_)}")

# Main content area
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 🔍 Select Your Symptoms")
    selected_symptoms = st.multiselect(
        "Search and select symptoms you're experiencing:",
        symptoms,
        help="Start typing to search for symptoms. You can select multiple symptoms."
    )

    # Display selected symptoms count
    if selected_symptoms:
        st.markdown(f"""
        <div class='symptom-count'>
            <h4 style='margin: 0; color: #2c5282;'>Selected Symptoms: {len(selected_symptoms)}</h4>
        </div>
        """, unsafe_allow_html=True)

        # Show selected symptoms as pills
        st.markdown("**Currently Selected:**")
        cols = st.columns(3)
        for idx, symptom in enumerate(selected_symptoms):
            with cols[idx % 3]:
                st.markdown(f"✓ {symptom}")

with col_right:
    st.markdown("### 💡 Tips")
    st.markdown("""
    <div class='info-card'>
    <ul style='margin: 0; padding-left: 1.2rem;'>
        <li>Be specific with symptoms</li>
        <li>Include all relevant symptoms</li>
        <li>More symptoms = better accuracy</li>
        <li>Consider symptom duration</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction button and results
col_button = st.columns([1, 2, 1])
with col_button[1]:
    predict_button = st.button("🔬 Analyze Symptoms", use_container_width=True)

if predict_button:
    if not selected_symptoms:
        st.error("⚠️ Please select at least one symptom to proceed with the analysis")
    else:
        with st.spinner("🔄 Analyzing symptoms..."):
            # Create input vector
            input_data = np.zeros(len(symptoms))
            for symptom in selected_symptoms:
                idx = symptoms.index(symptom)
                input_data[idx] = 1

            # Reshape for model
            input_data = input_data.reshape(1, -1)

            # Predict
            prediction = model.predict(input_data)
            predicted_disease = label_encoder.inverse_transform(prediction)[0]

            # Display result
            st.markdown(f"""
            <div class='prediction-box'>
                <h2 style='margin: 0; color: white;'>🎯 Prediction Result</h2>
                <h1 style='margin: 1rem 0; color: white; font-size: 2.5rem;'>{predicted_disease}</h1>
                <p style='margin: 0; color: #e6e6e6;'>Based on {len(selected_symptoms)} symptom(s)</p>
            </div>
            """, unsafe_allow_html=True)

            # Additional information
            col_info1, col_info2 = st.columns(2)

            with col_info1:
                st.markdown("### 📝 Next Steps")
                st.info("""
                1. Consult a healthcare professional
                2. Share these symptoms with your doctor
                3. Consider getting a proper diagnosis
                4. Follow medical advice
                """)

            with col_info2:
                st.markdown("### 🏥 Recommendations")
                st.success("""
                - Schedule a doctor's appointment
                - Keep track of symptom changes
                - Maintain a symptom diary
                - Follow up regularly
                """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 1rem;'>
    <p>Made with ❤️ using Streamlit | For Educational Purposes Only</p>
    <p style='font-size: 0.9rem;'>⚠️ This is not a substitute for professional medical advice, diagnosis, or treatment</p>
</div>
""", unsafe_allow_html=True)