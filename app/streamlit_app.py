import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_data, preprocess_data, split_data
def predict_single(model, input_data, scaler):
    df_input = pd.DataFrame([input_data])
    df_input['ExtracurricularActivities'] = 1 if input_data['ExtracurricularActivities'] == 'Yes' else 0
    df_input['PlacementTraining'] = 1 if input_data['PlacementTraining'] == 'Yes' else 0
    df_scaled = scaler.transform(df_input)
    df_scaled = pd.DataFrame(df_scaled, columns=df_input.columns)
    prob = model.predict_proba(df_scaled)[0][1]
    prediction = "Placed" if prob >= 0.5 else "NotPlaced"
    return prediction, prob

def get_shap_explanation(model, input_data, X_train_scaled, scaler):
    df_input = pd.DataFrame([input_data])
    df_input['ExtracurricularActivities'] = 1 if input_data['ExtracurricularActivities'] == 'Yes' else 0
    df_input['PlacementTraining'] = 1 if input_data['PlacementTraining'] == 'Yes' else 0
    df_scaled = scaler.transform(df_input)
    df_scaled = pd.DataFrame(df_scaled, columns=df_input.columns)
    
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(df_scaled)
    
    explanation = pd.DataFrame({
        'Feature': df_input.columns,
        'Value': pd.DataFrame([input_data])[df_input.columns].values[0],
        'SHAP Value': shap_values[0]
    }).sort_values('SHAP Value', ascending=False)
    
    return explanation

# Page config
st.set_page_config(
    page_title="PlacifyX",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide Streamlit header */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        display: none !important;
    }

    .block-container {
        padding-top: 2rem !important;
    }

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #2c1810;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #fdf6f0 0%, #faeee4 50%, #f5e6d8 100%);
    }

    /* Title */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #8B4513, #c0622a, #d4845a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        letter-spacing: 4px;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 400;
        color: #8b6355;
        text-align: center;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #c0622a, #8B4513, transparent);
        margin: 0.5rem 0 2rem 0;
        border: none;
    }

    /* Number input */
    div[data-testid="stNumberInput"] > div {
        background-color: #fff8f3 !important;
        border: 1.5px solid #ddb89a !important;
        border-radius: 8px !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #fff8f3 !important;
        color: #2c1810 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stNumberInput"] > div:focus-within {
        border-color: #ddb89a !important;
        box-shadow: none !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #f5e6d8 !important;
        color: #8B4513 !important;
        border: none !important;
        border-left: 1px solid #ddb89a !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #ddb89a !important;
    }

    /* Dropdown closed state */
    div[data-testid="stSelectbox"] > div > div {
        background-color: #fff8f3 !important;
        color: #2c1810 !important;
        border: 1.5px solid #ddb89a !important;
        border-radius: 8px !important;
    }

    div[data-testid="stSelectbox"] > div > div:hover {
        border-color: #8B4513 !important;
    }

    /* Dropdown open menu */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[data-baseweb="option"] {
        background-color: #fff8f3 !important;
        color: #2c1810 !important;
    }

    div[data-baseweb="option"]:hover {
        background-color: #f5e6d8 !important;
    }
    /* Override Streamlit's internal red focus color */
    [data-baseweb="input"] {
        border-color: #ddb89a !important;
    }

    [data-baseweb="input"]:focus-within {
        border-color: #8B4513 !important;
    }

    [data-baseweb="base-input"] {
        background-color: #fff8f3 !important;
    }

    /* Result badges */
    .placed-badge {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #2d5a27;
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 900;
        padding: 1rem 3rem;
        border-radius: 50px;
        text-align: center;
        display: inline-block;
        letter-spacing: 3px;
        border: 2px solid #a8d5a2;
        box-shadow: 0 8px 32px #a8d5a244;
        margin: 1rem auto;
    }

    .notplaced-badge {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 900;
        padding: 1rem 3rem;
        border-radius: 50px;
        text-align: center;
        display: inline-block;
        letter-spacing: 3px;
        border: 2px solid #f1aeb5;
        box-shadow: 0 8px 32px #f1aeb544;
        margin: 1rem auto;
    }

    .prob-text {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #8B4513;
        text-align: center;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f5e6d8;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #ddc4b0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #8b6355;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        color: #8B4513 !important;
        box-shadow: 0 2px 8px #8B451322;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #8B4513, #c0622a);
        color: white !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.1rem !important;
        letter-spacing: 2px;
        border: none;
        border-radius: 50px;
        padding: 1rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px #8B451344;
        text-transform: uppercase;
    }

    .stButton > button:hover {
        box-shadow: 0 8px 25px #8B451366;
        transform: translateY(-2px);
    }

    /* General text */
    p, div, span, li {
        color: #2c1810;
    }

    h1, h2, h3, h4 {
        color: #5c3317 !important;
        font-family: 'Playfair Display', serif !important;
    }

    /* Metric */
    [data-testid="stMetricValue"] {
        color: #8B4513 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
    }

    /* Progress bar */
    div[data-testid="stProgressBar"] > div {
        background: linear-gradient(90deg, #e74c3c, #f39c12, #2ecc71) !important;
    }

    [data-testid="stProgressBar"] [role="progressbar"] {
        background: linear-gradient(90deg, #8B4513, #c0622a) !important;
    }

    /* Hide menu and footer */
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    </style>
""", unsafe_allow_html=True)
st.markdown('''
<h1 style="
    font-family: 'Playfair Display', serif;
    font-size: 4.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #8B4513, #c0622a, #d4845a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: 4px;
    margin-bottom: 0;
    line-height: 1.1;
">PlacifyX</h1>
''', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ML · Explainable Intelligence · Placement Prediction</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Load model and data

def load_model_and_data():
    import importlib
    from sklearn.preprocessing import StandardScaler
    
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), 
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns
    )
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = load_model_and_data()


# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Your Profile", "Model Insights", "Bias & Fairness"])

# ==================== TAB 1 - PREDICT ====================
with tab1:
    st.subheader("Enter Your Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">CGPA</p>', unsafe_allow_html=True)
        cgpa = st.number_input("", min_value=0.0, max_value=10.0, value=7.5, step=0.01, label_visibility="collapsed")
        cgpa_color = "🟢" if cgpa >= 8.0 else "🟡" if cgpa >= 7.0 else "🔴"
        st.markdown(f"<small style='color:#8b6355'>{cgpa_color} {cgpa:.2f} / 10.0</small>", unsafe_allow_html=True)

        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">Internships</p>', unsafe_allow_html=True)
        internships = st.selectbox("", [0, 1, 2, 3], label_visibility="collapsed")
        int_color = "🟢" if internships == 2 else "🟡" if internships == 1 else "🔴"
        st.markdown(f"<small style='color:#8b6355'>{int_color} {internships} internship(s)</small>", unsafe_allow_html=True)
        
        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">Projects</p>', unsafe_allow_html=True)
        projects = st.selectbox("", [0, 1, 2, 3], key="projects", label_visibility="collapsed")
        proj_color = "🟢" if projects >= 3 else "🟡" if projects >= 1 else "🔴"
        st.markdown(f"<small style='color:#8b6355'>{proj_color} {projects} project(s)</small>", unsafe_allow_html=True)

    with col2:
        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">Workshops/Certifications</p>', unsafe_allow_html=True)
        workshops = st.selectbox("", [0, 1, 2, 3], key="workshops", label_visibility="collapsed")
        work_color = "🟢" if workshops >= 2 else "🟡" if workshops == 1 else "🔴"
        st.markdown(f"<small style='color:#8b6355'>{work_color} {workshops} workshop(s)</small>", unsafe_allow_html=True)
        
        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">Aptitude Test Score</p>', unsafe_allow_html=True)
        aptitude = st.number_input("", min_value=0, max_value=100, value=75, label_visibility="collapsed")
        apt_color = "🟢" if aptitude >= 80 else "🟡" if aptitude >= 65 else "🔴"
        st.markdown(f"<small style='color:#8b6355'>{apt_color} <b>{aptitude}</b> / 100</small>", unsafe_allow_html=True)
        
        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">Soft Skills Rating</p>', unsafe_allow_html=True)
        soft_skills = st.number_input("", min_value=0.0, max_value=5.0, value=4.0, step=0.1, label_visibility="collapsed")
        soft_color = "🟢" if soft_skills >= 4.3 else "🟡" if soft_skills >= 3.5 else "🔴"
        st.markdown(f"<small style='color:#8b6355'>{soft_color} <b>{soft_skills:.1f}</b> / 5.0</small>", unsafe_allow_html=True)

        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">Extracurricular Activities</p>', unsafe_allow_html=True)
        extracurricular = st.selectbox("", ["Yes", "No"], key="extracurricular", label_visibility="collapsed")
        st.markdown(f"<small style='color:#8b6355'>{'🟢' if extracurricular == 'Yes' else '🔴'} {extracurricular}</small>", unsafe_allow_html=True)

    with col3:
        
        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">Secondary School Certificate Marks (Class 10th)</p>', unsafe_allow_html=True)
        ssc = st.number_input("", min_value=0.0, max_value=100.0, value=70.0, step=0.01, label_visibility="collapsed")
        ssc_color = "🟢" if ssc >= 75 else "🟡" if ssc >= 60 else "🔴"
        st.markdown(f"<small style='color:#8b6355'>{ssc_color} <b>{ssc:.0f}</b> / 100</small>", unsafe_allow_html=True)

        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">Higher Secondary Certificate Marks (Class 12th)</p>', unsafe_allow_html=True)
        hsc = st.number_input("", min_value=0.0, max_value=100.0, value=72.0, step=0.01, label_visibility="collapsed")
        hsc_color = "🟢" if hsc >= 75 else "🟡" if hsc >= 60 else "🔴"
        st.markdown(f"<small style='color:#8b6355'>{hsc_color} <b>{hsc:.0f}</b> / 100</small>", unsafe_allow_html=True)

        st.markdown('<p style="font-size:1.2rem; font-weight:700; color:#5c3317;">Placement Training</p>', unsafe_allow_html=True)
        placement_training = st.selectbox("", ["Yes", "No"], key="placement_training", label_visibility="collapsed")
        st.markdown(f"<small style='color:#8b6355'>{'🟢' if placement_training == 'Yes' else '🔴'} {placement_training}</small>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    if st.button("Predict My Placement", use_container_width=True):
        
        input_data = {
            'CGPA': cgpa,
            'Internships': internships,
            'Projects': projects,
            'Workshops/Certifications': workshops,
            'AptitudeTestScore': aptitude,
            'SoftSkillsRating': soft_skills,
            'ExtracurricularActivities': extracurricular,
            'PlacementTraining': placement_training,
            'SSC_Marks': ssc,
            'HSC_Marks': hsc
        }
        
        prediction, probability = predict_single(model, input_data, scaler)
        
        # Result
        st.markdown("---")
        if prediction == "Placed":
            st.markdown(f'''
            <div style="
                background: linear-gradient(135deg, #d4edda, #c3e6cb);
                color: #2d5a27;
                font-family: Playfair Display, serif;
                font-size: 2rem;
                font-weight: 900;
                padding: 1rem;
                border-radius: 16px;
                border: 2px solid #a8d5a2;
                box-shadow: 0 8px 32px #a8d5a244;
                letter-spacing: 3px;
                text-align: center;
                width: 100%;
                box-sizing: border-box;
            ">PLACED</div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div style="
                background: linear-gradient(135deg, #f8d7da, #f5c6cb);
                color: #721c24;
                font-family: Playfair Display, serif;
                font-size: 2rem;
                font-weight: 900;
                padding: 1rem;
                border-radius: 16px;
                border: 2px solid #f1aeb5;
                box-shadow: 0 8px 32px #f1aeb544;
                letter-spacing: 3px;
                text-align: center;
                width: 100%;
                box-sizing: border-box;
            ">NOT PLACED</div>
            ''', unsafe_allow_html=True)

        st.markdown(f'''
        <div style="text-align:center; margin: 1.5rem 0 0.5rem 0;">
            <p style="font-family: Playfair Display, serif; font-size: 2rem;
              font-weight: 700; color: #8B4513; margin:0;">
            Placement Probability: {probability*100:.2f}%
            </p>
        </div>
        ''', unsafe_allow_html=True)

        prob_color = "#2ecc71" if probability >= 0.6 else "#f39c12" if probability >= 0.4 else "#e74c3c"
        st.markdown(f'''
        <div style="background:#f0e0d0; border-radius:50px; height:20px; 
            margin:0.5rem 0 1.5rem 0; width:100%;">
            <div style="
                background: {prob_color};
                width: {probability*100}%;
                height: 20px;
                border-radius: 50px;
            "></div>
        </div>
        ''', unsafe_allow_html=True)
        
        
        # SHAP explanation
        st.subheader("Why this prediction?")
        explanation = get_shap_explanation(model, input_data, X_train_scaled, scaler)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#fff8f3')
        ax.set_facecolor('#fff8f3')

        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in explanation['SHAP Value']]
        bars = ax.barh(explanation['Feature'], explanation['SHAP Value'], 
               color=colors, height=0.5, edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, explanation['SHAP Value']):
            ax.text(val + (0.01 if val >= 0 else -0.01), 
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}',
            va='center', ha='left' if val >= 0 else 'right',
            fontsize=8, color='#2c1810', fontweight='600')

        ax.axvline(x=0, color='#8b6355', linewidth=1, linestyle='--')
        ax.set_title('Feature Contribution to Your Prediction', 
             fontweight='bold', fontsize=11, color='#5c3317', pad=10)
        ax.set_xlabel('SHAP Value', fontsize=9, color='#8b6355')
        ax.tick_params(axis='y', labelsize=9, colors='#2c1810')
        ax.tick_params(axis='x', labelsize=8, colors='#8b6355')

        green_patch = mpatches.Patch(color='#2ecc71', label='Helps placement')
        red_patch = mpatches.Patch(color='#e74c3c', label='Hurts placement')
        ax.legend(handles=[green_patch, red_patch], fontsize=8,
          loc='upper right', framealpha=0.9,
          facecolor='#fff8f3', edgecolor='#ddb89a')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ddb89a')
        ax.spines['bottom'].set_color('#ddb89a')

        plt.tight_layout()

        col_shap1, col_shap2, col_shap3 = st.columns([2, 6, 2])
        with col_shap2:
            st.pyplot(fig, use_container_width=True)

        # Plain English summary
        st.subheader("Key Insights for You")
        top_positive = explanation[explanation['SHAP Value'] > 0].head(3)
        top_negative = explanation[explanation['SHAP Value'] < 0].head(3)
        
        if not top_positive.empty:
            features = ', '.join([f'**{f}**' for f in top_positive['Feature'].tolist()])
            st.success(f" **Your Strengths:** {features}")
        if not top_negative.empty:
            features = ', '.join([f'**{f}**' for f in top_negative['Feature'].tolist()])
            st.error(f" **Areas to Improve:** {features}")
        
        # Store input for tab 2
        st.session_state['input_data'] = input_data
        st.session_state['prediction'] = prediction
        st.session_state['probability'] = probability

# ==================== TAB 2 - YOUR PROFILE ====================
with tab2:
    if 'input_data' not in st.session_state:
        st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.5rem; color:#8b6355; text-align:center; margin-top:3rem;">Go to Predict tab first, enter your details and click Predict.</p>', unsafe_allow_html=True)
    else:
        input_data = st.session_state['input_data']
        prediction = st.session_state['prediction']
        probability = st.session_state['probability']

        st.markdown('<p style="font-family: Playfair Display, serif; font-size:2rem; font-weight:700; color:#5c3317; margin-bottom:1rem;">Your Profile vs Average Placed Student</p>', unsafe_allow_html=True)

        df_raw = load_data()
        df_placed = df_raw[df_raw['PlacementStatus'] == 'Placed']

        numeric_features = ['CGPA', 'Internships', 'Projects',
                           'Workshops/Certifications', 'AptitudeTestScore',
                           'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']

        avg_placed = df_placed[numeric_features].mean()
        user_values = [input_data[f] for f in numeric_features]

        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#fff8f3')
        ax.set_facecolor('#fff8f3')

        x = np.arange(len(numeric_features))
        width = 0.35

        bars1 = ax.bar(x - width/2, user_values, width,
                       label='You', color='#c0622a', alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x + width/2, avg_placed.values, width,
                       label='Avg Placed Student', color='#a8d5a2', alpha=0.85, edgecolor='white')

        ax.set_xlabel('Features', fontsize=10, color='#5c3317')
        ax.set_ylabel('Value', fontsize=10, color='#5c3317')
        ax.set_title('Your Profile vs Average Placed Student',
                     fontweight='bold', fontsize=13, color='#5c3317', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(numeric_features, rotation=25, ha='right',
                           fontsize=9, color='#2c1810')
        ax.tick_params(axis='y', colors='#8b6355')
        ax.legend(fontsize=10, facecolor='#fff8f3', edgecolor='#ddb89a')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ddb89a')
        ax.spines['bottom'].set_color('#ddb89a')

        plt.tight_layout()
        col1, col2, col3 = st.columns([3, 9, 3])
        with col2:
            st.pyplot(fig, use_container_width=True)

        # Gap analysis
        st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.8rem; font-weight:700; color:#5c3317; margin:1.5rem 0 1rem 0;">Gap Analysis</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        strengths = []
        improvements = []

        for feature, user_val, avg_val in zip(numeric_features, user_values, avg_placed.values):
            diff = user_val - avg_val
            if diff >= 0:
                strengths.append(f"**{feature}:** You ({user_val:.2f}) vs Avg ({avg_val:.2f}) +{diff:.2f}")
            else:
                improvements.append(f"**{feature}:** You ({user_val:.2f}) vs Avg ({avg_val:.2f}) {diff:.2f}")

        with col1:
            st.markdown('<p style="font-size:1.1rem; font-weight:700; color:#2d5a27;">Your Strengths</p>', unsafe_allow_html=True)
            if strengths:
                for s in strengths:
                    st.success(s)
            else:
                st.info("No strengths above average yet — keep working!")

        with col2:
            st.markdown('<p style="font-size:1.1rem; font-weight:700; color:#721c24;">Areas to Improve</p>', unsafe_allow_html=True)
            if improvements:
                for i in improvements:
                    st.error(i)
            else:
                st.success("You're above average in all areas!")

        # Placement probability
        st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.8rem; font-weight:700; color:#5c3317; margin:1.5rem 0 1rem 0;">Your Placement Probability</p>', unsafe_allow_html=True)
        st.markdown(f'''
        <div style="text-align:center; margin: 1rem 0 0.5rem 0;">
            <p style="font-family: Playfair Display, serif; font-size: 2rem;
                    font-weight: 700; color: #8B4513; margin:0;">
                Placement Probability: {probability*100:.2f}%
            </p>
        </div>
        ''', unsafe_allow_html=True)

        prob_color = "#2ecc71" if probability >= 0.6 else "#f39c12" if probability >= 0.4 else "#e74c3c"
        st.markdown(f'''
        <div style="background:#f0e0d0; border-radius:50px; height:20px;
            margin:0.5rem 0 1.5rem 0; width:100%;">
            <div style="
                background: {prob_color};
                width: {probability*100}%;
                height: 20px;
                border-radius: 50px;
            "></div>
        </div>
        ''', unsafe_allow_html=True)

        if prediction == "Placed":
            st.success("You are likely to be Placed!")
        else:
            st.error("You need to improve your profile.")

# ==================== TAB 3 - MODEL INSIGHTS ====================
with tab3:
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:2rem; font-weight:700; color:#5c3317; margin-bottom:1rem;">Model Performance & Insights</p>', unsafe_allow_html=True)

    # Model comparison table
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.5rem; font-weight:700; color:#5c3317; margin-bottom:0.5rem;">Model Comparison</p>', unsafe_allow_html=True)

    model_results = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.8085, 0.7875, 0.7820],
        'F1 Score': [0.7727, 0.7410, 0.7383],
        'ROC-AUC': [0.8837, 0.8668, 0.8578]
    }
    df_results = pd.DataFrame(model_results)
    df_results = df_results.set_index('Model')

    # Style the table
    st.dataframe(
        df_results.style
        .highlight_max(axis=0, color='#d4edda')
        .format('{:.4f}'),
        use_container_width=True
    )

    st.markdown('<p style="font-size:0.9rem; color:#8b6355; margin-top:0.3rem;">🟢 Green = best score in each metric</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Confusion matrix and ROC curve side by side
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.5rem; font-weight:700; color:#5c3317; margin-bottom:0.5rem;">Evaluation Plots</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor('#fff8f3')
        ax_cm.set_facecolor('#fff8f3')

        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
                    xticklabels=['NotPlaced', 'Placed'],
                    yticklabels=['NotPlaced', 'Placed'],
                    ax=ax_cm, linewidths=0.5)
        ax_cm.set_title('Confusion Matrix', fontweight='bold',
                         fontsize=12, color='#5c3317', pad=10)
        ax_cm.set_xlabel('Predicted', color='#5c3317')
        ax_cm.set_ylabel('Actual', color='#5c3317')
        ax_cm.tick_params(colors='#2c1810')
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)

    with col2:
        # ROC curve
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        fig_roc.patch.set_facecolor('#fff8f3')
        ax_roc.set_facecolor('#fff8f3')

        ax_roc.plot(fpr, tpr, color='#c0622a', linewidth=2.5,
                    label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax_roc.plot([0, 1], [0, 1], color='#ddb89a',
                    linestyle='--', linewidth=1.5, label='Random Classifier')
        ax_roc.fill_between(fpr, tpr, alpha=0.1, color='#c0622a')

        ax_roc.set_title('ROC Curve', fontweight='bold',
                          fontsize=12, color='#5c3317', pad=10)
        ax_roc.set_xlabel('False Positive Rate', color='#5c3317')
        ax_roc.set_ylabel('True Positive Rate', color='#5c3317')
        ax_roc.legend(fontsize=9, facecolor='#fff8f3', edgecolor='#ddb89a')
        ax_roc.tick_params(colors='#8b6355')
        ax_roc.spines['top'].set_visible(False)
        ax_roc.spines['right'].set_visible(False)
        ax_roc.spines['left'].set_color('#ddb89a')
        ax_roc.spines['bottom'].set_color('#ddb89a')

        plt.tight_layout()
        st.pyplot(fig_roc, use_container_width=True)

    st.markdown("---")

    # SHAP global feature importance
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.5rem; font-weight:700; color:#5c3317; margin-bottom:0.5rem;">Global Feature Importance (SHAP)</p>', unsafe_allow_html=True)

    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    feature_importance = pd.DataFrame({
        'Feature': X_test_scaled.columns,
        'Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('Importance', ascending=True)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    fig_imp.patch.set_facecolor('#fff8f3')
    ax_imp.set_facecolor('#fff8f3')

    bars = ax_imp.barh(feature_importance['Feature'],
                       feature_importance['Importance'],
                       color='#c0622a', alpha=0.85,
                       edgecolor='white', height=0.6)

    for bar, val in zip(bars, feature_importance['Importance']):
        ax_imp.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8,
                    color='#2c1810', fontweight='600')

    ax_imp.set_title('Average Feature Impact on Placement Prediction',
                     fontweight='bold', fontsize=11, color='#5c3317', pad=10)
    ax_imp.set_xlabel('Mean |SHAP Value|', color='#5c3317', fontsize=9)
    ax_imp.tick_params(axis='y', labelsize=9, colors='#2c1810')
    ax_imp.tick_params(axis='x', labelsize=8, colors='#8b6355')
    ax_imp.spines['top'].set_visible(False)
    ax_imp.spines['right'].set_visible(False)
    ax_imp.spines['left'].set_color('#ddb89a')
    ax_imp.spines['bottom'].set_color('#ddb89a')

    plt.tight_layout()
    col1, col2, col3 = st.columns([0.5, 9, 0.5])
    with col2:
        st.pyplot(fig_imp, use_container_width=True)

    # Key insights
    st.markdown("---")
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.5rem; font-weight:700; color:#5c3317; margin-bottom:0.5rem;">Key Model Insights</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Best Model:** Logistic Regression with ROC-AUC of 0.8837")
    with col2:
        st.info("**Top Predictor:** Aptitude Test Score has highest impact on placement outcome")
    with col3:
        st.info("**Class Imbalance:** Dataset has 42% placed vs 58% not placed - handled with stratified split")

# ==================== TAB 4 - BIAS & FAIRNESS ====================
with tab4:
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:2rem; font-weight:700; color:#5c3317; margin-bottom:1rem;">Bias & Fairness Analysis</p>', unsafe_allow_html=True)

    st.markdown('<p style="font-size:1rem; color:#8b6355; margin-bottom:1.5rem;">This section analyzes whether the model performs fairly across different student subgroups. A fair model should predict equally well regardless of extracurricular participation, placement training, or academic background.</p>', unsafe_allow_html=True)

    df_raw = load_data()
    df_raw['PlacementStatus_binary'] = (df_raw['PlacementStatus'] == 'Placed').astype(int)

    st.markdown("---")

    # 1. Placement rate by Extracurricular Activities
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.5rem; font-weight:700; color:#5c3317; margin-bottom:0.5rem;">Placement Rate by Extracurricular Activities</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        extra_rate = df_raw.groupby('ExtracurricularActivities')['PlacementStatus_binary'].mean() * 100

        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        fig1.patch.set_facecolor('#fff8f3')
        ax1.set_facecolor('#fff8f3')

        bars = ax1.bar(extra_rate.index, extra_rate.values,
                       color=['#e74c3c', '#2ecc71'], alpha=0.85,
                       edgecolor='white', width=0.5)

        for bar, val in zip(bars, extra_rate.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom',
                     fontsize=10, fontweight='700', color='#2c1810')

        ax1.set_title('Placement Rate by Extracurricular Activities',
                      fontweight='bold', fontsize=10, color='#5c3317', pad=10)
        ax1.set_xlabel('Extracurricular Activities', color='#5c3317', fontsize=9)
        ax1.set_ylabel('Placement Rate (%)', color='#5c3317', fontsize=9)
        ax1.tick_params(colors='#2c1810')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#ddb89a')
        ax1.spines['bottom'].set_color('#ddb89a')
        ax1.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)

        gap = extra_rate['Yes'] - extra_rate['No']
        st.markdown(f'<p style="font-size:0.9rem; color:#8b6355; text-align:center;">Gap: <b style="color:#c0622a">{gap:.1f}%</b> higher placement rate with extracurriculars</p>', unsafe_allow_html=True)

    with col2:
        # 2. Placement rate by Placement Training
        training_rate = df_raw.groupby('PlacementTraining')['PlacementStatus_binary'].mean() * 100

        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        fig2.patch.set_facecolor('#fff8f3')
        ax2.set_facecolor('#fff8f3')

        bars = ax2.bar(training_rate.index, training_rate.values,
                       color=['#e74c3c', '#2ecc71'], alpha=0.85,
                       edgecolor='white', width=0.5)

        for bar, val in zip(bars, training_rate.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom',
                     fontsize=10, fontweight='700', color='#2c1810')

        ax2.set_title('Placement Rate by Placement Training',
                      fontweight='bold', fontsize=10, color='#5c3317', pad=10)
        ax2.set_xlabel('Placement Training', color='#5c3317', fontsize=9)
        ax2.set_ylabel('Placement Rate (%)', color='#5c3317', fontsize=9)
        ax2.tick_params(colors='#2c1810')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#ddb89a')
        ax2.spines['bottom'].set_color('#ddb89a')
        ax2.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

        gap2 = training_rate['Yes'] - training_rate['No']
        st.markdown(f'<p style="font-size:0.9rem; color:#8b6355; text-align:center;">Gap: <b style="color:#c0622a">{gap2:.1f}%</b> higher placement rate with training</p>', unsafe_allow_html=True)

    st.markdown("---")

    # 3. CGPA distribution by placement
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.5rem; font-weight:700; color:#5c3317; margin-bottom:0.5rem;">CGPA Distribution by Placement Status</p>', unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    fig3.patch.set_facecolor('#fff8f3')
    ax3.set_facecolor('#fff8f3')

    for status, color, label in zip(['Placed', 'NotPlaced'],
                                     ['#2ecc71', '#e74c3c'],
                                     ['Placed', 'Not Placed']):
        subset = df_raw[df_raw['PlacementStatus'] == status]['CGPA']
        subset.plot.kde(ax=ax3, color=color, linewidth=2.5, label=label)

    placed_mean = df_raw[df_raw['PlacementStatus'] == 'Placed']['CGPA'].mean()
    notplaced_mean = df_raw[df_raw['PlacementStatus'] == 'NotPlaced']['CGPA'].mean()

    ax3.axvline(placed_mean, color='#2ecc71', linestyle='--', linewidth=1.5)
    ax3.axvline(notplaced_mean, color='#e74c3c', linestyle='--', linewidth=1.5)
    ax3.text(placed_mean + 0.05, 0.9, f'Placed\nMean: {placed_mean:.2f}',
             color='#2ecc71', fontsize=8, transform=ax3.get_xaxis_transform())
    ax3.text(notplaced_mean + 0.05, 0.75, f'NotPlaced\nMean: {notplaced_mean:.2f}',
             color='#e74c3c', fontsize=8, transform=ax3.get_xaxis_transform())

    ax3.set_title('CGPA Distribution by Placement Status',
                  fontweight='bold', fontsize=11, color='#5c3317', pad=10)
    ax3.set_xlabel('CGPA', color='#5c3317', fontsize=9)
    ax3.set_ylabel('Density', color='#5c3317', fontsize=9)
    ax3.legend(facecolor='#fff8f3', edgecolor='#ddb89a', fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#ddb89a')
    ax3.spines['bottom'].set_color('#ddb89a')

    plt.tight_layout()
    col1, col2, col3 = st.columns([0.5, 9, 0.5])
    with col2:
        st.pyplot(fig3, use_container_width=True)

    st.markdown("---")

    # 4. Subgroup model performance
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.5rem; font-weight:700; color:#5c3317; margin-bottom:0.5rem;">Model Performance Across Subgroups</p>', unsafe_allow_html=True)

    from sklearn.metrics import accuracy_score

    df_test = X_test_scaled.copy()
    df_test['PlacementStatus'] = y_test.values
    df_test['ExtracurricularActivities'] = df_raw.loc[y_test.index, 'ExtracurricularActivities'].values
    df_test['PlacementTraining'] = df_raw.loc[y_test.index, 'PlacementTraining'].values
    df_test['predicted'] = model.predict(X_test_scaled)

    subgroups = {
        'With Extracurriculars': df_test[df_test['ExtracurricularActivities'] == 'Yes'],
        'Without Extracurriculars': df_test[df_test['ExtracurricularActivities'] == 'No'],
        'With Placement Training': df_test[df_test['PlacementTraining'] == 'Yes'],
        'Without Placement Training': df_test[df_test['PlacementTraining'] == 'No'],
    }

    subgroup_acc = {}
    for name, subset in subgroups.items():
        if len(subset) > 0:
            acc = accuracy_score(subset['PlacementStatus'], subset['predicted'])
            subgroup_acc[name] = round(acc * 100, 2)

    fig4, ax4 = plt.subplots(figsize=(8, 3.5))
    fig4.patch.set_facecolor('#fff8f3')
    ax4.set_facecolor('#fff8f3')

    colors = ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c']
    bars = ax4.barh(list(subgroup_acc.keys()), list(subgroup_acc.values()),
                    color=colors, alpha=0.85, edgecolor='white', height=0.5)

    for bar, val in zip(bars, subgroup_acc.values()):
        ax4.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{val}%', va='center', fontsize=9,
                 fontweight='700', color='#2c1810')

    ax4.set_title('Model Accuracy Across Subgroups',
                  fontweight='bold', fontsize=11, color='#5c3317', pad=10)
    ax4.set_xlabel('Accuracy (%)', color='#5c3317', fontsize=9)
    ax4.set_xlim(0, 110)
    ax4.tick_params(axis='y', labelsize=9, colors='#2c1810')
    ax4.tick_params(axis='x', labelsize=8, colors='#8b6355')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color('#ddb89a')
    ax4.spines['bottom'].set_color('#ddb89a')

    plt.tight_layout()
    col1, col2, col3 = st.columns([0.5, 9, 0.5])
    with col2:
        st.pyplot(fig4, use_container_width=True)

    # Fairness summary
    st.markdown("---")
    st.markdown('<p style="font-family: Playfair Display, serif; font-size:1.5rem; font-weight:700; color:#5c3317; margin-bottom:0.5rem;">Fairness Summary</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.warning("**Extracurricular Gap:** Students with extracurricular activities have significantly higher placement rates — model reflects this real-world disparity")
    with col2:
        st.warning("**Training Gap:** Placement training shows a strong positive effect — colleges should ensure equal access to training programs")
    with col3:
        st.info("**CGPA Gap:** Placed students have mean CGPA of 8.02 vs 7.47 for not placed — academic performance remains a key differentiator")