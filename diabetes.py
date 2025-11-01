import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Diabetes Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom white background + style
st.markdown("""
    <style>
    .main { background-color: white; }
    h1, h2, h3 { color: #1a1a1a; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        width: 100%;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Set default page in session_state
if "page" not in st.session_state:
    st.session_state.page = "About"

# Sidebar buttons as navigation
st.sidebar.title("🔍 Navigation")
if st.sidebar.button("About"):
    st.session_state.page = "About"
if st.sidebar.button("Prediction"):
    st.session_state.page = "Prediction"
if st.sidebar.button("Evaluation"):
    st.session_state.page = "Evaluation"

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes.csv')
    scaled_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    scaler = MinMaxScaler()
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])
    return data, scaler

data, scaler = load_data()
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'Age', 'DiabetesPedigreeFunction']
target = 'Outcome'
x = data[features]
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# ROUTING BASED ON SESSION STATE
if st.session_state.page == "About":
    st.title("🧠 Diabetes Prediction App")
    st.markdown("""
    ### What is Diabetes?
    Diabetes is a chronic condition that affects how your body turns food into energy. It occurs when your blood glucose is too high, either due to insufficient insulin production or ineffective insulin usage.

    ### Dataset Used
    **Pima Indians Diabetes Dataset** containing:
    - Health metrics like glucose, insulin, BMI, age
    - A binary outcome (0 = No diabetes, 1 = Diabetes)

    ### Why Machine Learning?
    - 💉 Over 400 million people worldwide have diabetes.
    - 🧬 Genetics and lifestyle factors influence diabetes risk.
    - 📈 Machine Learning models like Random Forest help predict the likelihood of diabetes.
    """)

elif st.session_state.page == "Prediction":
    st.title("🔬 Diabetes Risk Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 100)
        blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
    with col2:
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    with col3:
        age = st.number_input("Age", 1, 120, 30)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

    if st.button("🧠 Predict"):
        user_input = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'Age': [age],
            'DiabetesPedigreeFunction': [dpf]
        })

        scaled_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
        user_input[scaled_columns] = scaler.transform(user_input[scaled_columns])
        prediction = model.predict(user_input)

        st.success("✅ Prediction Complete!")
        st.subheader("📋 Result")
        st.write("Diabetes Outcome (0 = No, 1 = Yes):", int(prediction[0]))

elif st.session_state.page == "Evaluation":
    st.title("📊 Model Performance")

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.metric("✅ Model Accuracy", f"{accuracy * 100:.2f}%")
    st.subheader("📋 Classification Report")
    st.code(report)

    st.subheader("📌 Feature Importance")
    feature_importances = pd.Series(model.feature_importances_, index=features)
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importances.sort_values().plot(kind='barh', color='skyblue', ax=ax)
    st.pyplot(fig)
