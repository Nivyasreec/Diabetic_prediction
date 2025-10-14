import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import traceback # Import traceback for detailed error logging

# --- Page Configuration & CSS ---
st.set_page_config(layout="wide", page_title="AI-Powered Diabetes Risk & Nutrition Advisor", page_icon="🍎")
def load_css():
    st.markdown("""
    <style>
        .stApp { background: #f0f2f6; }
        h1, h2, h3 { color: #1a73e8; font-weight: bold; }
        .suggestion-card { background-color: #ffffff; border-left: 6px solid #1a73e8; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .urgent-card { border-left-color: #d93025; } /* Red for High Risk */
        .low-risk-card { border-left-color: #1e8e3e; } /* Green for Low Risk */
        .meal-plan-card { border-left-color: #fbbc05; } /* Yellow for Moderate Risk */
        div[data-testid="stButton"] > button[kind="primary"] { background-color: #1a73e8; color: white; border: none; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)
load_css()

# --- Backend Database Connection (SQLite) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(script_dir, 'predictions.db')

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def save_to_db(data_dict):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        columns = '`, `'.join(data_dict.keys())
        columns = f"`{columns}`"
        placeholders = ', '.join(['?'] * len(data_dict))
        sql = f"INSERT INTO predictions ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, list(data_dict.values()))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"🔴 A SQLite error occurred while saving! Error: {e}")
        st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during DB save! Error: {e}")
        st.code(traceback.format_exc())
    finally:
        if conn:
            conn.close()

def fetch_from_db():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching from database: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_food_list():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM food_items ORDER BY name ASC")
        food_list = [row[0] for row in cursor.fetchall()]
        conn.close()
        return food_list
    except:
        return ["Apple", "Pizza", "Salad", "Samosa", "Chicken Curry"]

# --- Model Loading & Helper Functions ---
@st.cache_resource
def get_artifacts():
    artifacts = {}
    MODELS_DIR = os.path.join(script_dir, 'model1')
    if not os.path.isdir(MODELS_DIR):
        st.error(f"Fatal Error: Model directory not found at {MODELS_DIR}")
        return None
    files_to_load = {
        'diabetes_model': 'diabetes_xgboost_model.pkl',
        'diabetes_scaler': 'diabetes_scaler.pkl',
        'model_features': 'diabetes_model_features.pkl',
        'calorie_model': 'calorie_model.pkl',
        'sugar_model': 'sugar_model.pkl',
        'tfidf_vectorizer': 'tfidf_vectorizer.pkl'
    }
    for key, filename in files_to_load.items():
        path = os.path.join(MODELS_DIR, filename)
        try:
            if os.path.exists(path):
                artifacts[key] = joblib.load(path)
            else:
                st.error(f"File Not Found: '{filename}' in '{MODELS_DIR}'")
                return None
        except Exception as e:
            st.error(f"Error loading '{filename}': {e}")
            return None
    st.success("All models loaded successfully!")
    return artifacts
artifacts = get_artifacts()

def suggestion_card(title, text, icon, card_class=""):
    st.markdown(f'<div class="suggestion-card {card_class}"><h3>{icon} {title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

def predict_nutrition(food_list, vectorizer, cal_model, sug_model):
    if not food_list or not all([vectorizer, cal_model, sug_model]): return 0, 0, []
    total_calories, total_sugar, breakdown = 0, 0, []
    for item in food_list:
        try:
            food_vector = vectorizer.transform([item['name']]).toarray()
            est_calories = max(0, cal_model.predict(food_vector)[0] * item['quantity'])
            est_sugar = max(0, sug_model.predict(food_vector)[0] * item['quantity'])
            total_calories += est_calories
            total_sugar += est_sugar
            breakdown.append({"Item": f"{item['quantity']}x {item['name']}", "Calories": f"{est_calories:.1f}", "Sugar": f"{est_sugar:.1f}"})
        except:
            pass
    return total_calories, total_sugar, breakdown

def get_recommendations(risk_level, nutrition_data):
    recommendations = {}
    sugar_intake = nutrition_data.get('sugar', 0)

    if risk_level == "High Risk":
        recommendations['urgent'] = ("Consult a Doctor Immediately", "Your HIGH risk level requires professional medical attention. Please schedule an appointment to discuss these results.", "🩺", "urgent-card")
        recommendations['lifestyle'] = "### 🏃 Lifestyle Suggestions\n- **Monitor Blood Sugar Regularly** as advised by a professional.\n- **Engage in at least 150 minutes of moderate exercise** per week (e.g., brisk walking, cycling).\n- **Prioritize stress management and 7-8 hours of sleep** nightly."
        recommendations['food'] = "### 🥗 Food Intake Recommendations\n- **Strictly Limit Sugars & Refined Carbs** like white bread, pastries, and sugary drinks.\n- **Focus on a diet rich in fiber** from vegetables, whole grains, and lean proteins."
    elif risk_level == "Moderate Risk":
        recommendations['urgent'] = ("Focus on Proactive Changes", "Your MODERATE risk level is a crucial warning sign. Making positive changes now can significantly reduce your future risk.", "🏃", "meal-plan-card")
        recommendations['lifestyle'] = "### 🏃 Lifestyle Suggestions\n- **Increase Physical Activity:** Aim for 30 minutes of activity most days.\n- **Incorporate strength training** 2 times a week to build muscle."
        recommendations['food'] = "### 🥗 Food Intake Recommendations\n- **Adopt a Balanced Diet:** Fill half your plate with vegetables, a quarter with lean protein, and a quarter with whole grains.\n- **Control portion sizes** and be mindful of your carbohydrate intake."
    else: # Low Risk
        recommendations['urgent'] = ("Maintain Your Excellent Health", "Congratulations on your LOW risk! Continue your healthy habits.", "🌟", "low-risk-card")
        recommendations['lifestyle'] = "### 🏃 Lifestyle Suggestions\n- **Stay active** and maintain a balanced diet.\n- **Continue with regular check-ups** to monitor your health."
        recommendations['food'] = "### 🥗 Food Intake Recommendations\n- **Focus on a nutrient-dense diet** with plenty of fruits, vegetables, and whole grains."
    
    if sugar_intake > 40:
        sugar_warning = f"Your estimated sugar intake is **{sugar_intake:.1f}g**, which is high and should be reduced."
        title, text, icon, card_class = recommendations.get('urgent')
        recommendations['urgent'] = (title, f"{text} {sugar_warning}", icon, card_class)
    
    return recommendations

# --- Main App ---
st.title("AI-Powered Diabetes Risk & Nutrition Advisor")
if not artifacts: st.stop()
if 'risk_calculated' not in st.session_state: st.session_state['risk_calculated'] = False
if 'food_list' not in st.session_state: st.session_state['food_list'] = []
if 'history' not in st.session_state: st.session_state['history'] = []

tab1, tab2, tab3, tab4 = st.tabs(["Step 1: Your Profile & Food Log", "Step 2: Your Results", "📜 Session History", "☁️ Backend Data"])

with tab1:
    st.header("Enter Your Information")
    form_col, food_col = st.columns(2)
    with form_col:
        with st.expander("👤 **Enter Your Health Metrics**", expanded=True):
            age = st.slider("Age", 1, 100, 30, key="age_input")
            hypertension = 1 if st.radio("Do you have hypertension?", ["No", "Yes"], key="hyper_input") == "Yes" else 0
            heart_disease = 1 if st.radio("Do you have heart disease?", ["No", "Yes"], key="heart_input") == "Yes" else 0
            height = st.number_input("Height (cm)", 50, 250, 170, key="height_input")
            weight = st.number_input("Weight (kg)", 10, 300, 70, key="weight_input")
            hba1c_level = st.slider("HbA1c Level (%)", 3.0, 9.0, 5.7, 0.1, key="hba1c_input")
            blood_glucose_level = st.slider("Last Blood Glucose Level (mg/dL)", 50, 300, 100, key="glucose_input")
            height_m = height / 100
            bmi = weight / (height_m ** 2) if height_m > 0 else 0
            st.metric("Your Body Mass Index (BMI)", f"{bmi:.2f}")
    with food_col:
        with st.expander("🥗 **Log Your Daily Food Intake**", expanded=True):
            food_options = fetch_food_list()
            with st.form("food_form"):
                food_name = st.selectbox("Food Item", options=food_options, index=None, placeholder="Search or select a food item...")
                quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
                if st.form_submit_button("Add Food Item") and food_name:
                    st.session_state.food_list.append({"name": food_name.strip(), "quantity": quantity})
                    st.rerun()
            if st.session_state.food_list:
                st.write("**Today's Food Log:**")
                for i, item in enumerate(st.session_state.food_list):
                    col1, col2 = st.columns([4, 1])
                    col1.write(f"- {item['quantity']}x {item['name']}")
                    if col2.button("X", key=f"remove_{i}", help="Remove item"):
                        st.session_state.food_list.pop(i)
                        st.rerun()
    st.markdown("---")
    
    if st.button("Analyze My Risk & Get Action Plan", type="primary", use_container_width=True, disabled=not st.session_state.food_list):
        with st.spinner("Analyzing your risk..."):
            try:
                input_data_from_ui = {
                    'age': age,
                    'bmi': bmi,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'HbA1c_level': hba1c_level,
                    'blood_glucose_level': blood_glucose_level
                }
                
                expected_features = artifacts['model_features']
                input_df = pd.DataFrame([input_data_from_ui])[expected_features]
                
                input_scaled = artifacts['diabetes_scaler'].transform(input_df)
                pred_proba = artifacts['diabetes_model'].predict_proba(input_scaled)[:, 1][0]
                
                total_calories, total_sugar, breakdown = predict_nutrition(st.session_state.food_list, artifacts['tfidf_vectorizer'], artifacts['calorie_model'], artifacts['sugar_model'])
                nutrition_data = {"calories": total_calories, "sugar": total_sugar, "breakdown": breakdown}
                
                food_log_str = ", ".join([f"{d['quantity']}x {d['name']}" for d in st.session_state.food_list])
                db_data = {
                    "risk_score": float(pred_proba * 100), "age": age, "bmi": float(bmi),
                    "hypertension": "Yes" if hypertension else "No", "heart_disease": "Yes" if heart_disease else "No",
                    "hba1c_level": hba1c_level, "blood_glucose": blood_glucose_level,
                    "total_calories": total_calories, "total_sugar": total_sugar, "food_log": food_log_str
                }
                save_to_db(db_data)
                
                history_entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inputs": db_data, "food_log": pd.DataFrame(st.session_state.food_list), "nutrition": nutrition_data, "prediction": {"proba": pred_proba}}
                st.session_state.history.append(history_entry)
                st.session_state.latest_results = history_entry
                st.session_state.risk_calculated = True
                st.success("Analysis Complete! Click 'Step 2' for results.")
                
            except KeyError as e:
                st.error(f"🔴 **Feature Mismatch Error!**")
                st.error(f"The model expected a feature named `{e}` but it was not found in the data you provided from the UI.")
                st.error("Please check that the feature names in your training script and Streamlit app match exactly.")
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                st.code(traceback.format_exc())

# --- TAB 2: RESULTS (WITH ENHANCED BUSINESS LOGIC) ---
with tab2:
    st.header("Your Latest Personalized Results")
    if not st.session_state.risk_calculated:
        st.info("Please fill out your profile in 'Step 1' and click the 'Analyze' button.")
    else:
        latest_results = st.session_state.latest_results
        proba = latest_results["prediction"]["proba"]
        user_inputs = latest_results["inputs"] 
        
        # --- ⭐ UPDATED LOGIC BLOCK FOR BETTER ACCURACY AND USER EXPERIENCE ⭐ ---
        display_score = proba
        
        # Initialize risk level based on AI model prediction first
        if proba >= 0.60:
            risk_level, color = "High Risk", "red"
        elif proba >= 0.20:
            risk_level, color = "Moderate Risk", "orange"
        else:
            risk_level, color = "Low Risk", "green"

        # Apply expert adjustment for high-risk individuals misclassified as low-risk
        # This also triggers if the score is high in the "Low" range (e.g., 15-20%)
        if risk_level == "Low Risk" and (user_inputs.get('hba1c_level', 0) > 6.0 or user_inputs.get('bmi', 0) > 35 or proba > 0.15):
            risk_level = "Moderate Risk"
            color = "orange"
            # Adjust the displayed score to be at least the minimum for the new category
            if display_score < 0.20:
                display_score = 0.20 
            st.warning("")
        
        # The gauge chart now uses the potentially adjusted 'display_score' for its value
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=display_score * 100,
            title={'text': f"Diabetes Risk Score: {risk_level}"}, number={'suffix': '%'},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color},
                   'steps': [
                       {'range': [0, 20], 'color': 'lightgreen'},
                       {'range': [20, 60], 'color': 'lightyellow'},
                       {'range': [60, 100], 'color': 'lightcoral'}
                   ]}
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🍽️ **Today's Nutrition Snapshot**", expanded=True):
            nd = latest_results["nutrition"]
            st.metric("Total Estimated Calories", f"{nd['calories']:.0f} kcal")
            st.metric("Total Estimated Sugar", f"{nd['sugar']:.1f} g")
            if nd.get('breakdown'): st.dataframe(pd.DataFrame(nd['breakdown']), use_container_width=True)
                
        st.subheader("💡 Your Recommended Actions")
        recommendations = get_recommendations(risk_level, latest_results["nutrition"])
        if 'urgent' in recommendations: suggestion_card(*recommendations['urgent'])
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1: st.markdown(recommendations.get('lifestyle', ''))
        with rec_col2: st.markdown(recommendations.get('food', ''))

# --- TAB 3: SESSION HISTORY (WITH INCREASED PRECISION) ---
with tab3:
    st.header("Your Session History")
    if not st.session_state.history:
        st.info("Your past entries from this session will appear here.")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            # --- ⭐ UPDATED FORMATTING FOR MORE PRECISION ⭐ ---
            with st.expander(f"**Entry from {entry['timestamp']}** (AI Risk Score: {entry['prediction']['proba']:.2%})", expanded=(i==0)):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Your Metrics:**")
                    st.json({k: v for k, v in entry['inputs'].items() if k not in ['food_log']})
                    st.write("**Nutrition:**")
                    st.metric("Total Estimated Calories", f"{entry['nutrition']['calories']:.0f} kcal")
                    st.metric("Total Estimated Sugar", f"{entry['nutrition']['sugar']:.1f} g")
                with col2:
                    st.write("**Food Log:**")
                    if not entry['food_log'].empty: st.dataframe(entry['food_log'])
                    else: st.write("No food was logged for this entry.")

# --- TAB 4: BACKEND DATA ---
with tab4:
    st.header("Fetch All Saved Data from Database")
    st.write("This table shows all data that has been saved to the local SQLite database from all users.")
    if st.button("🔄 Refresh Data from Database"): st.cache_data.clear()
    
    @st.cache_data(ttl=60)
    def get_backend_data(): return fetch_from_db()
        
    history_df = get_backend_data()
    if history_df.empty:
        st.info("No data has been saved to the database yet.")
    else:
        display_df = history_df.drop(columns=['id'], errors='ignore')
        st.dataframe(display_df, use_container_width=True)
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download All Data as CSV", data=csv, file_name='all_diabetes_risk_data.csv', mime='text/csv')

st.markdown("---")
st.markdown("<div style='text-align: center;'><strong>Disclaimer:</strong> This tool is for informational purposes only and does not constitute medical advice. Consult with a healthcare professional for any health concerns.</div>", unsafe_allow_html=True)