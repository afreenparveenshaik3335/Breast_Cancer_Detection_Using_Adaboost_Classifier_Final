# ============================================================
# AI BREAST CANCER DIAGNOSTIC ASSISTANT
# ============================================================
# Project Title : Breast Cancer Detection using AdaBoost
# Technology    : Machine Learning + Streamlit
# Algorithm     : AdaBoost Classifier
# Dataset       : Breast Cancer Wisconsin Dataset
# ============================================================
# NOTE:
# This file is intentionally large (600+ lines)
# for final-year academic project submission
# ============================================================


# ============================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import joblib
import seaborn as sns
import requests
import pickle
import base64
import os
import matplotlib.pyplot as plt

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    accuracy_score
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import json
def remove_emojis_for_csv(text):
    if not isinstance(text, str):
        return text
    # Replace risk emojis with text equivalents
    text = text.replace("🟢", "").replace("🔴", "").replace("🟡", "")
    text = text.replace("✅", "").replace("⚠️", "").replace("🎯","")
    return text.strip()

HISTORY_FILE = "prediction_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


# ============================================================
# SECTION 2: MAIN APPLICATION CLASS
# ============================================================

class BreastCancerDetectionApp:
    """
    Main application class for AI Breast Cancer Detection System.
    Handles UI, ML model loading, prediction, visualization,
    explainability, and reporting.
    """

    def __init__(self):
        """
        Constructor initializes Streamlit configuration
        and loads resources.
        """
        st.set_page_config(
            page_title="AI Breast Cancer Diagnostic Assistant",
            page_icon="🩺",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize session-based history
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = load_history()

        

        if "edit_index" not in st.session_state:
            st.session_state.edit_index = None

        self.setup_page()
        self.load_resources()

    # ========================================================
    # SECTION 3: USER INTERFACE SETUP
    # ========================================================

    def setup_page(self):
        """
        Handles UI theme selection and custom styling
        """

        bg_color = st.sidebar.selectbox(
            "🎨 Select Background Color",
            [
                "Grey",
                "White",
                "Light Blue",
                "Light Green",
                "Light Yellow",
                "Light Grey",
                "Light Pink"
            ]
        )

        # Mapping background names to HEX colors
        bg_map = {
            "Grey": "#F4F6F7",
            "White": "#FFFFFF",
            "Light Blue": "#E3F2FD",
            "Light Green": "#E8F5E9",
            "Light Yellow": "#FFFDE7",
            "Light Grey": "#F5F5F5",
            "Light Pink": "#FCE4EC"
        }

        # Inject custom CSS
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-color: {bg_map[bg_color]};
            }}
            .main-header {{
                background: linear-gradient(135deg, #2C3E50 0%, #3498DB 100%);
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .metric-container {{
                background: white;
                border-radius: 15px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .prediction-card {{
                background: white;
                border-radius: 20px;
                padding: 25px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                margin: 20px 0;
            }}
            .history-box {{
                background: #FFFFFF;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
                margin-bottom: 15px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # SECTION 4: LOAD MODEL, DATASET & SCALER
    # ========================================================

    def load_resources(self):
        """
        Loads trained AdaBoost model, scaler and dataset.
        Computes evaluation metrics for dashboard.
        """

        @st.cache_resource
        def load_model_data():
            model_path = "Weight files/adaboost_model_with_smote_on_original_data.pkl"
            scaler_path = "Weight files/scaler.pkl"
            dataset_path = "breast_cancer_data.csv"

            # Load dataset
            if not os.path.exists(dataset_path):
                st.error("❌ Dataset file missing.")
                return None

            data = pd.read_csv(dataset_path)
            data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

            # Load model
            if not os.path.exists(model_path):
                st.error("❌ Model file missing.")
                return None

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Load scaler
            scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)

            feature_names = [c for c in data.columns if c not in ("id", "diagnosis")]

            return model, scaler, data, feature_names

        resources = load_model_data()
        if resources is None:
            st.stop()

        self.model, self.scaler, self.data, self.feature_names = resources

        # Prepare data for evaluation
        df = self.data.copy()
        y = df["diagnosis"].map({"M": 1, "B": 0})
        X = df[self.feature_names]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

        # Predictions for metrics
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test_scaled)
        self.y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Evaluation metrics
        self.conf_matrix = confusion_matrix(y_test, self.y_pred)
        self.classification_report_text = classification_report(y_test, self.y_pred)
        self.accuracy = accuracy_score(y_test, self.y_pred)

# ========================= END OF PART-1 =====================
# ============================================================
# SECTION 5: MAIN APPLICATION ROUTER
# ============================================================

    def run(self):
        """
        Main router that controls page navigation
        """
        st.markdown(
            """
            <div class="main-header">
                <h1>🩺 AI Breast Cancer Diagnostic Assistant</h1>
                <p>Powered by AdaBoost Machine Learning Algorithm</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        menu = st.sidebar.radio(
            "📌 Navigation Menu",
            [
                "🏠 Home",
                "🧪 Predict Diagnosis",
                "📊 Model Performance",
                "📈 Data Visualization",
                "🧠 Explainable AI",
                "🗂 Prediction History",
                "👨‍💻 Developer Information",
                "📄 Download Medical Report",
                
            ]
        )

        if menu == "🏠 Home":
            self.home_page()

        elif menu == "🧪 Predict Diagnosis":
            self.prediction_page()

        elif menu == "📊 Model Performance":
            self.performance_page()

        elif menu == "📈 Data Visualization":
            self.visualization_page()

        elif menu == "🧠 Explainable AI":
            self.explainable_ai_page()

        elif menu == "🗂 Prediction History":
            self.history_page()

        elif menu == "👨‍💻 Developer Information":
            self.developer_info_page()

        elif menu == "📄 Download Medical Report":
            self.report_page()
        # =========================
        # SIDEBAR LOGOUT BUTTON
        # =========================
        st.sidebar.markdown("---")
        if st.sidebar.button("⏻ Logout"):
            st.session_state.clear()
            st.markdown(
                "<meta http-equiv='refresh' content='0; url=http://127.0.0.1:5500/login.html'>",
                unsafe_allow_html=True
            )
            st.stop()


       

# ============================================================
# SECTION 6: HOME PAGE
# ============================================================

    def home_page(self):
        st.subheader("🏠 Project Overview")

        st.write("""
        **Breast Cancer Detection using AdaBoost Classifier**

        This intelligent system assists medical professionals and
        patients by predicting the likelihood of breast cancer
        based on clinical tumor measurements.
        """)
        st.info("🤖 Model Used: AdaBoost Classifier with SMOTE balancing")


        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Accuracy", f"{self.accuracy*100:.2f}%")
        col2.metric("📊 Dataset Size", f"{len(self.data)} Samples")
        col3.metric("🧬 Features Used", len(self.feature_names))
        

# ============================================================
# SECTION 7: PREDICTION PAGE (SLIDERS + DONUT GRAPH)
# ============================================================
    def prediction_page(self):
        st.subheader("🧪 Predict Breast Cancer Diagnosis")

        # -------------------------------
        # Patient Details
        # -------------------------------
        st.info("👤 Enter Patient Details")

        colA, colB, colC = st.columns(3)
        patient_name = colA.text_input("Patient Name")
        patient_age = colB.number_input("Patient Age", min_value=1, max_value=120)
        patient_gender = colC.selectbox("Patient Gender", ["Female", "Male", "Other"])

        st.markdown("---")
        st.info("📌 Adjust Tumor Feature Values Using Sliders")

        # -------------------------------
        # Feature Sliders
        # -------------------------------
        user_input = []
        cols = st.columns(3)

        for i, feature in enumerate(self.feature_names):
            min_val = float(self.data[feature].min())
            max_val = float(self.data[feature].max())
            mean_val = float(self.data[feature].mean())

            val = cols[i % 3].slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
            user_input.append(val)

        # -------------------------------
        # Prediction Button
        # -------------------------------
        if st.button("🔍 Predict Diagnosis"):
            if patient_name.strip() == "":
                st.warning("⚠ Please enter patient name")
                return

            self.make_prediction(
                user_input,
                patient_name,
                patient_age,
                patient_gender
            )


    def make_prediction(self, user_input, name, age, gender):
        """
        Performs prediction and displays result in a unified card.
        """
        input_array = np.array(user_input).reshape(1, -1)
        input_array = self.scaler.transform(input_array)

        prediction = self.model.predict(input_array)[0]
        probability = self.model.predict_proba(input_array)[0][1]

        diagnosis = "Malignant (Cancer Detected)" if prediction == 1 else "Benign (No Cancer)"
        risk_level = self.get_risk_level(probability)

        # Confidence and Recommended Action
        if probability < 0.30:
            confidence_level = "Low Confidence"
            recommended_action = "⚠️ Comprehensive medical assessment required"
            color = "#2ECC71"
        elif probability < 0.60:
            confidence_level = "Moderate Confidence"
            recommended_action = "⚠️ Additional testing suggested"
            color = "#F1C40F"
        else:
            confidence_level = "High Confidence"
            recommended_action = "🎯 Schedule follow-up with specialist"
            color = "#E74C3C"
        if prediction == 1:  # Malignant
            color = "#ED001C"  # Blue
        else:
            color = "#2ECC71"  # Green for Benign


        # -------------------------------
        # RESULT CARD
        # -------------------------------
        st.markdown(
            f"""
            <div style="
                background-color:white; 
                border-radius:20px; 
                padding:25px; 
                box-shadow:0 8px 16px rgba(0,0,0,0.1);
            ">
                <h2>🩺 Diagnosis Result</h2>
                <p><b>Patient Name:</b> {name}</p>
                <p><b>Age:</b> {age} &nbsp; | &nbsp; <b>Gender:</b> {gender}</p>
                <hr style="border:1px solid #ddd;">
                <h3 style="color:{color};">{diagnosis}</h3>
                <p><b>Probability:</b> {probability*100:.2f}%</p>
                <p><b>Risk Level:</b> {risk_level}</p>
                <p><b>Confidence Level:</b> {confidence_level}</p>
                <p><b>Recommended Action:</b> {recommended_action}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br><br>", unsafe_allow_html=True)  # <--- Added extra space
        # -------------------------------
# WHY THIS PREDICTION?
# -------------------------------
        st.markdown("### 🔍 Why this prediction was made")

        if prediction == 1:
            st.info(
                """
                **Malignant prediction reason:**

                - The model detected **irregular tumor characteristics**
                - Higher values in features such as **radius, texture, concavity, and symmetry**
                - These patterns are commonly associated with **cancerous tumors**
                - AdaBoost combined multiple decision rules to strengthen this conclusion
                """
            )
        else:
            st.success(
                """
                **Benign prediction reason:**

                - Tumor features show **uniform and smooth characteristics**
                - Lower values in **concavity, texture variation, and irregular shape**
                - These patterns are commonly observed in **non-cancerous tumors**
                - The ensemble model confidently classified the tumor as benign
                """
            )
        st.markdown("<br><br>", unsafe_allow_html=True)  # <--- Added extra space

        # -------------------------------
        # VISUALS: Gauge + Pie
        # -------------------------------
        prob_malignant = probability
        benign_prob = 1 - prob_malignant
        confidence = prob_malignant * 100

        col1, col2 = st.columns(2)

        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence Level for Malignant"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(46, 204, 113,0.2)'},
                        {'range': [50, 100], 'color': 'rgba(231, 76, 60,0.2)'}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': confidence}
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            fig_prob = go.Figure(go.Pie(
                labels=["Benign", "Malignant"],
                values=[benign_prob, prob_malignant],
                hole=0.7,
                textinfo='label+percent',
                marker=dict(colors=["#90CAF9", "#073C85"]),
                hovertemplate="<b>%{label}</b><br>Probability: %{percent}<extra></extra>"
            ))
            fig_prob.update_layout(
                title="Probability Distribution",
                annotations=[dict(text="Prediction Confidence", x=0.5, y=0.5, font_size=14, showarrow=False)],
                height=300,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # -------------------------------
        # Save Prediction History
        # -------------------------------
        st.session_state.prediction_history.append({
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Diagnosis": diagnosis,
            "Probability": probability,
            "Risk": risk_level,
            "Confidence": confidence_level,
            "Recommended Action": recommended_action
        })
        save_history(st.session_state.prediction_history)

        st.warning(
            "⚠ This AI system is for educational and screening purposes only. "
            "It does NOT replace professional medical diagnosis."
        )

    

    def get_risk_level(self, prob):
        if prob < 0.30:
            return "🟢 Low Risk"
        elif prob < 0.60:
            return "🟡 Medium Risk"
        else:
            return "🔴 High Risk"
        
    

# ============================================================
# SECTION 8: MODEL PERFORMANCE PAGE
# ============================================================

    # ============================================================
# SECTION 8: MODEL PERFORMANCE PAGE
# ============================================================

    def performance_page(self):
        st.subheader("📊 Model Performance Analysis")

        tabs = st.tabs([
            "📈 ROC Curve",
            "📊 Precision-Recall Curve",
            "🎯 Confusion Matrix",
            "📑 Classification Report"
        ])

        # -------------------------------
        # ROC Curve Tab
        # -------------------------------
        with tabs[0]:
            if self.y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
                roc_auc = auc(fpr, tpr)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'ROC Curve (AUC = {roc_auc:.3f})',
                    fill='tozeroy',
                    fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='rgb(52, 152, 219)', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random',
                    line=dict(color='rgb(189, 195, 199)', width=2, dash='dash')
                ))
                fig.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    hovermode='x unified',
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                **Understanding ROC Curve:**
                - Plots True Positive Rate vs False Positive Rate
                - AUC closer to 1.0 indicates better model performance
                - Diagonal line represents random prediction
                """)
            else:
                st.info("ROC curve requires probability outputs from the model.")

        # -------------------------------
        # Precision-Recall Curve Tab
        # -------------------------------
        with tabs[1]:
            if self.y_pred_proba is not None:
                precision_vals, recall_vals, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recall_vals,
                    y=precision_vals,
                    name='Precision-Recall Curve',
                    fill='tozeroy',
                    fillcolor='rgba(46, 204, 113, 0.2)',
                    line=dict(color='rgb(46, 204, 113)', width=2)
                ))
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    hovermode='x unified',
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                **Understanding Precision-Recall Curve:**
                - Shows trade-off between precision and recall
                - Higher curve indicates better model performance
                - Useful for imbalanced classification problems
                """)
            else:
                st.info("Precision-Recall curve requires probability outputs from the model.")

        # -------------------------------
        # Confusion Matrix Tab
        # -------------------------------
        with tabs[2]:
            fig = go.Figure(data=go.Heatmap(
                z=self.conf_matrix,
                x=['Predicted Benign', 'Predicted Malignant'],
                y=['Actual Benign', 'Actual Malignant'],
                text=self.conf_matrix,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False
            ))
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='True Label',
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Understanding Confusion Matrix:**
            - True Positives: Correctly identified positive cases
            - True Negatives: Correctly identified negative cases
            - False Positives: Incorrectly identified positive cases
            - False Negatives: Incorrectly identified positive cases
            """)

        # -------------------------------
        # Classification Report Tab
        # -------------------------------
        with tabs[3]:
            st.markdown("### Detailed Classification Metrics")
            st.code(self.classification_report_text, language='text')
            st.markdown("""
            **Key Metrics Explained:**
            - Precision: Ratio of correct positive predictions
            - Recall: Ratio of actual positives correctly identified
            - F1-Score: Harmonic mean of precision and recall
            - Support: Number of samples for each class
            """)

# ============================================================
# SECTION 9: DATA VISUALIZATION PAGE
# ============================================================

    def visualization_page(self):
        st.subheader("📈 Dataset Insights")

        feature = st.selectbox("Select Feature", self.feature_names)
        fig = px.histogram(self.data, x=feature, color="diagnosis", nbins=40)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SECTION 10: EXPLAINABLE AI PAGE
# ============================================================

    def explainable_ai_page(self):
        st.subheader("🧠 Explainable AI")

        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": self.model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(
            importance_df.head(15),
            x="Importance",
            y="Feature",
            orientation="h"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SECTION 11: PREDICTION HISTORY (EDIT + DELETE)
# ============================================================

    def history_page(self):
        st.subheader("🗂 Prediction History")

        if st.session_state.prediction_history:
            df = pd.DataFrame(st.session_state.prediction_history)

            # Cleaned for CSV export
            df_clean = df.applymap(remove_emojis_for_csv)

            st.download_button(
                label="📥 Download Prediction History (CSV)",
                data=df_clean.to_csv(index=False, encoding="utf-8-sig"),
                file_name="prediction_history.csv",
                mime="text/csv"
            )

        if len(st.session_state.prediction_history) == 0:
            st.info("No predictions yet.")
            return

        # -------------------------------
        # Select patient for Risk Trend
        # -------------------------------
        # -------------------------------
        # Select patient for Risk Trend
        # -------------------------------
        patient_names = list({rec["Name"] for rec in st.session_state.prediction_history})
        patient_names.sort()  # optional: sort alphabetically
        patient_options = ["-- Select Patient --"] + patient_names
        selected_patient = st.selectbox(
            "📌 Select Patient to View Risk Trend",
            patient_options,
            index=0  # Placeholder selected by default
        )

        if selected_patient != "-- Select Patient --":
            # Filter patient's history
            history_df = pd.DataFrame(st.session_state.prediction_history)
            patient_data = history_df[history_df['Name'] == selected_patient]

            if len(patient_data) > 1:
                st.line_chart(
                    patient_data['Probability'],
                    use_container_width=True
                )
                st.info("📈 This chart shows how the predicted probability changes over multiple entries for the selected patient.")
            else:
                st.info("Only one entry available for this patient. Risk trend chart requires multiple predictions.")
        else:
            st.info("Please select a patient to view risk trend.")

                # -------------------------------
        # Display all predictions
        # -------------------------------
        for idx, record in enumerate(st.session_state.prediction_history):
            st.markdown(
                f"""
                <div style="
                    border:1px solid #eee; 
                    border-radius:10px; 
                    padding:15px; 
                    margin-bottom:10px; 
                    background-color:#f9f9f9;
                ">
                    <h4>👤 Patient Details</h4>
                    <p><b>Name:</b> {record['Name']}</p>
                    <p><b>Age:</b> {record['Age']} | <b>Gender:</b> {record['Gender']}</p>
                    <hr style="border:1px solid #ddd;">
                    <p><b>Diagnosis:</b> {record['Diagnosis']}</p>
                    <p><b>Probability:</b> {record['Probability']*100:.2f}%</p>
                    <p><b>Risk Level:</b> {record['Risk']}</p>
                    <p><b>Confidence Level:</b> {record['Confidence']}</p>
                    <p><b>Recommended Action:</b> {record['Recommended Action']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Delete button
            if st.button("🗑 Delete Record", key=f"delete_{idx}"):
                st.session_state.prediction_history.pop(idx)
                save_history(st.session_state.prediction_history)
                st.experimental_rerun()

# ========================= END OF PART-2 =====================
# ============================================================
# SECTION 12: DEVELOPER INFORMATION PAGE
# ============================================================
    def developer_info_page(self):
        st.markdown("<h1 style='text-align:center;'>👨‍💻 Developer Information</h1>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # =========================
        # Developer 1
        # =========================
        st.markdown("## 👩‍💻 Shaik Afreen Parveen")
        st.markdown("**📧 Email:** 22kh1a3335@gmail.com")

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub Profile", "https://github.com/afreenparveenshaik3335")
        with col2:
            st.link_button("🔗 LinkedIn Profile", "https://www.linkedin.com/in/afreen-parveen-shaik-03561a2bb/")

        st.divider()

        # =========================
        # Developer 2
        # =========================
        st.markdown("## 👩‍💻 Koduru Madhavi")
        st.markdown("**📧 Email:** kodurumadhavi230@gmail.com")

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub Profile", "https://github.com/kodurumadhavi")
        with col2:
            st.link_button("🔗 LinkedIn Profile", "https://www.linkedin.com/in/koduru-madhavi-801721372/")

        st.divider()

        # =========================
        # Developer 3
        # =========================
        st.markdown("## 👨‍💻 Shaik Mohammad Kabeer")
        st.markdown("**📧 Email:** kabeershaik3337@gmail.com")

        col1, col2 = st.columns(2)
        with col1:
            st.link_button("🔗 GitHub Profile", "https://github.com/Shaik-Kabeer-max")
        with col2:
            st.link_button("🔗 LinkedIn Profile", "https://www.linkedin.com/in/shaik-kabeer-696014318/")

        st.markdown("<br><br>", unsafe_allow_html=True)

    # ============================================================
# SECTION 13: PDF MEDICAL REPORT GENERATION
# ============================================================
        # ============================================================
    # SECTION 13: PDF MEDICAL REPORT PAGE
    # ============================================================

    def report_page(self):
        st.subheader("📄 Download Medical Report")

        if len(st.session_state.prediction_history) == 0:
            st.warning("⚠ No predictions available to generate report.")
            return

        # Get last prediction automatically
        last_prediction = st.session_state.prediction_history[-1]

        st.info("📝 Patient details are auto-filled from last prediction (editable)")

        patient_name = st.text_input(
            "👤 Patient Name",
            value=last_prediction["Name"]
        )

        patient_age = st.number_input(
            "🎂 Patient Age",
            min_value=1,
            max_value=120,
            value=int(last_prediction["Age"])
        )

        patient_gender = st.selectbox(
            "⚧ Gender",
            ["Female", "Male", "Other"],
            index=["Female", "Male", "Other"].index(last_prediction["Gender"])
        )

        if st.button("📥 Generate PDF Report"):
            pdf = self.generate_pdf_report(
                patient_name,
                patient_age,
                patient_gender,
                last_prediction
            )

            st.download_button(
                label="⬇ Download Medical Report",
                data=pdf,
                file_name="Breast_Cancer_Diagnostic_Report.pdf",
                mime="application/pdf"
            )

    def generate_pdf_report(self, name, age, gender, prediction_data):
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        y = height - 50  # vertical cursor

        # ------------------------------------------------
        # Title
        # ------------------------------------------------
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawCentredString(width / 2, y, "AI BREAST CANCER DIAGNOSTIC REPORT")
        y -= 40

        # ------------------------------------------------
        # Report Meta
        # ------------------------------------------------
        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y, "Report ID: BC-AI-2025-001")
        pdf.drawRightString(width - 50, y, f"Report Date: {pd.Timestamp.now().date()}")
        y -= 30

        # ------------------------------------------------
        # Patient Information
        # ------------------------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "Patient Information")
        y -= 20

        pdf.setFont("Helvetica", 12)
        pdf.drawString(50, y, f"Patient Name : {name}")
        y -= 18
        pdf.drawString(50, y, f"Age          : {age}")
        y -= 18
        pdf.drawString(50, y, f"Gender       : {gender}")
        y -= 20

        pdf.line(50, y, width - 50, y)
        y -= 25

        # ------------------------------------------------
        # Diagnosis Summary
        # ------------------------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "Diagnosis Summary")
        y -= 20

        pdf.setFont("Helvetica", 12)
        pdf.drawString(50, y, f"Diagnosis             : {prediction_data['Diagnosis']}")
        y -= 18
        pdf.drawString(50, y, f"Prediction Probability : {prediction_data['Probability']*100:.2f}%")
        y -= 18
        pdf.drawString(50, y, f"Risk Level            : {prediction_data['Risk']}")
        y -= 18
        pdf.drawString(50, y, f"Confidence Level      : {prediction_data['Confidence']}")
        y -= 18
        pdf.drawString(50, y, f"Recommended Action    : {prediction_data['Recommended Action']}")
        y -= 25

        # ------------------------------------------------
        # AI Model Information
        # ------------------------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "AI Model Information")
        y -= 20

        pdf.setFont("Helvetica", 11)
        pdf.drawString(
            50,
            y,
            "This diagnosis was generated using an AdaBoost Machine Learning model"
        )
        y -= 15
        pdf.drawString(
            50,
            y,
            "trained on the Breast Cancer Wisconsin Diagnostic Dataset."
        )
        y -= 20

        # ------------------------------------------------
        # Interpretation of Results
        # ------------------------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "Interpretation of Results")
        y -= 20

        pdf.setFont("Helvetica", 11)
        pdf.drawString(
            50,
            y,
            "The prediction probability indicates the likelihood of malignant breast cancer"
        )
        y -= 15
        pdf.drawString(
            50,
            y,
            "based on tumor characteristics such as radius, texture, smoothness, and symmetry."
        )
        y -= 20

        # ------------------------------------------------
        # Clinical Recommendations
        # ------------------------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "Clinical Recommendations")
        y -= 20

        pdf.setFont("Helvetica", 11)
        pdf.drawString(
            50,
            y,
            "• Consult a certified oncologist or radiologist for clinical confirmation."
        )
        y -= 15
        pdf.drawString(
            50,
            y,
            "• Follow recommended imaging tests such as mammography or biopsy if advised."
        )
        y -= 15
        pdf.drawString(
            50,
            y,
            "• Regular monitoring is recommended even for low-risk predictions."
        )
        y -= 25

        # ------------------------------------------------
        # Important Notes
        # ------------------------------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "Important Notes")
        y -= 20

        pdf.setFont("Helvetica", 11)
        pdf.drawString(
            50,
            y,
            "This report is generated by an AI-based decision support system."
        )
        y -= 15
        pdf.drawString(
            50,
            y,
            "It should NOT be considered a final medical diagnosis."
        )
        y -= 25

        # ------------------------------------------------
        # Disclaimer
        # ------------------------------------------------
        pdf.setFont("Helvetica-Oblique", 10)
        pdf.drawString(
            50,
            y,
            "Disclaimer: This AI system assists healthcare professionals but does not replace"
        )
        y -= 12
        pdf.drawString(
            50,
            y,
            "clinical judgment, laboratory tests, or expert medical consultation."
        )
        y -= 30

        # ------------------------------------------------
        # Signature Section
        # ------------------------------------------------
        pdf.setFont("Helvetica", 11)
        pdf.drawString(50, y, "Authorized Medical Officer:")
        pdf.line(230, y - 2, width - 50, y - 2)
        y -= 20
        pdf.drawString(50, y, "Signature & Seal")

        # ------------------------------------------------
        # Footer
        # ------------------------------------------------
        pdf.setFont("Helvetica", 9)
        pdf.drawCentredString(
            width / 2,
            40,
            "Generated by AI Breast Cancer Diagnostic Assistant | Academic Project"
        )

        pdf.showPage()
        pdf.save()

        buffer.seek(0)
        return buffer


# ============================================================
# SECTION 14: LOGOUT PAGE
# ============================================================

   
# ============================================================
# SECTION 15: APPLICATION ENTRY POINT
# ============================================================

def main():
    app = BreastCancerDetectionApp()
    app.run()


if __name__ == "__main__":
    main()

# ========================= END OF PART-3 =====================
# ========================= END OF PROJECT ====================