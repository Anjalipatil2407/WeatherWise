
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="WeatherWise | Smart Weather Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).with_name(
    "WeatherWise_RandomForest_Model_fixed.pkl"
)

# ==========================================
# 2. LOAD MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# ==========================================
# 3. CUSTOM DARK ROSE CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #17151B;
    color: #F5EAF0;
}

/* Main header */

[data-testid="stHeader"] {
    background-color: #17151B;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #211C25;
}

[data-testid="stSidebar"] * {
    color: #F5EAF0;
}

/* Main title */

.main-title {
    color: #E8A0BA;
    font-size: 45px;
    font-weight: 800;
    letter-spacing: 1px;
}

.subtitle {
    color: #B8AAB5;
    font-size: 17px;
}

/* Section headings */

.section-title {
    color: #E8A0BA;
    font-size: 25px;
    font-weight: bold;
}

h1, h2, h3 {
    color: #F5EAF0;
}

/* Normal text */

p, label {
    color: #F5EAF0;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: #27232D;
    border: 1px solid #403542;
    padding: 20px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #B8AAB5;
}

[data-testid="stMetricValue"] {
    color: #E8A0BA;
}

/* Input form */

[data-testid="stForm"] {
    background-color: #211C25;
    border: 1px solid #403542;
    padding: 22px;
    border-radius: 15px;
}

/* Input fields */

[data-baseweb="select"] > div {
    background-color: #27232D;
    color: #F5EAF0;
    border-color: #594454;
}

[data-testid="stNumberInput"] input {
    background-color: #27232D;
    color: #F5EAF0;
    border-color: #594454;
}

/* Buttons */

div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #A94E75;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
    transition: 0.3s;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #C46B91;
    color: white;
    border: none;
}

/* Dividers */

hr {
    border-color: #403542;
}

/* Progress bar */

[data-testid="stProgressBar"] > div > div {
    background-color: #E8A0BA;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border: 1px solid #403542;
    border-radius: 10px;
}

/* Footer */

.footer {
    text-align: center;
    color: #B8AAB5;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.markdown(
    "<h1 style='color:#E8A0BA;'>🌦️ WeatherWise</h1>",
    unsafe_allow_html=True
)

st.sidebar.caption("SMART WEATHER & RAIN PREDICTION")

st.sidebar.divider()

st.sidebar.subheader("About WeatherWise")

st.sidebar.write("""
WeatherWise is a Machine Learning project that
predicts rainfall using weather conditions.

It uses a trained Random Forest Classifier
to generate rain and no-rain predictions.
""")

st.sidebar.divider()

st.sidebar.subheader("Project Information")

st.sidebar.write("🤖 Algorithm: Random Forest")
st.sidebar.write("📊 Features: 10")
st.sidebar.write("🏙️ Cities: 5")

st.sidebar.divider()

st.sidebar.subheader("Supported Cities")

st.sidebar.write("""
- Bengaluru
- Chennai
- Delhi
- Kolkata
- Mumbai
""")

st.sidebar.divider()

st.sidebar.caption("Powered by Python & Scikit-learn")


# ==========================================
# 5. MAIN DASHBOARD HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🌦️ WeatherWise</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart weather monitoring and rainfall prediction '
    'using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ==========================================
# 6. DASHBOARD OVERVIEW
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Supported Cities",
        value="05"
    )

with col2:
    st.metric(
        label="Input Features",
        value="10"
    )

with col3:
    st.metric(
        label="ML Algorithm",
        value="Random Forest"
    )

st.divider()


# ==========================================
# 7. WEATHER INPUT SECTION
# ==========================================

st.markdown(
    '<div class="section-title">'
    '🌤️ Weather Information'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the weather conditions below to predict rainfall."
)


cities = [
    "Bengaluru",
    "Chennai",
    "Delhi",
    "Kolkata",
    "Mumbai"
]

months = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}


with st.form("weather_form"):

    st.subheader("Location Details")

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox(
            "Select City",
            cities
        )

    with col2:
        month_name = st.selectbox(
            "Select Month",
            list(months.keys())
        )

    st.divider()

    st.subheader("Temperature Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        temp_mean = st.number_input(
            "Mean Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=28.0
        )

    with col2:
        temp_max = st.number_input(
            "Maximum Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=32.0
        )

    with col3:
        temp_min = st.number_input(
            "Minimum Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=24.0
        )

    st.divider()

    st.subheader("Atmospheric Conditions")

    col1, col2 = st.columns(2)

    with col1:

        humidity = st.number_input(
            "Mean Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        pressure = st.number_input(
            "Mean Pressure (hPa)",
            min_value=800.0,
            max_value=1100.0,
            value=1010.0
        )

        wind_speed = st.number_input(
            "Maximum Wind Speed (km/h)",
            min_value=0.0,
            max_value=300.0,
            value=15.0
        )

    with col2:

        cloud_cover = st.number_input(
            "Mean Cloud Cover (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=1000.0,
            value=0.0
        )

    st.write("")

    submitted = st.form_submit_button(
        "🌧️ Predict Weather",
        use_container_width=True
    )


# ==========================================
# 8. MACHINE LEARNING PREDICTION
# ==========================================

if submitted:

    if not (temp_min <= temp_mean <= temp_max):

        st.error(
            "Please check the temperatures. "
            "Minimum ≤ Mean ≤ Maximum is required."
        )

    else:

        input_data = pd.DataFrame([{
            "City": city,
            "Month": months[month_name],
            "Temperature_Mean": temp_mean,
            "Temperature_Max": temp_max,
            "Temperature_Min": temp_min,
            "Humidity_Mean": humidity,
            "Pressure_Mean": pressure,
            "Wind_Speed_Max": wind_speed,
            "Cloud_Cover_Mean": cloud_cover,
            "Rainfall_mm": rainfall
        }])

        try:

            prediction = model.predict(input_data)[0]

            probabilities = model.predict_proba(input_data)[0]

            classes = list(model.classes_)

            rain_probability = (
                probabilities[classes.index(1)]
                if 1 in classes else 0.0
            )

            no_rain_probability = (
                probabilities[classes.index(0)]
                if 0 in classes else 0.0
            )

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '📊 Prediction Results'
                '</div>',
                unsafe_allow_html=True
            )

            if prediction == 1:

                st.success(
                    "🌧️ Rain is predicted by WeatherWise!"
                )

            else:

                st.info(
                    "☀️ No Rain is predicted by WeatherWise!"
                )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Rain Probability",
                    f"{rain_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "No Rain Probability",
                    f"{no_rain_probability * 100:.2f}%"
                )

            st.subheader("Rain Probability")

            st.progress(float(rain_probability))

            st.caption(
                "This bar represents the model's estimated probability of rain."
            )

            st.divider()

            st.subheader("Weather Details Submitted")

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ==========================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ==========================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '📈 Weather Feature Importance'
    '</div>',
    unsafe_allow_html=True
)

st.write("""
Understand which weather conditions contribute
most to the Random Forest model's predictions.
""")

try:

    preprocessing = model.named_steps["preprocessing"]

    classifier = model.named_steps["model"]

    feature_names = preprocessing.get_feature_names_out()

    importance_values = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    def clean_feature_name(name):

        name = name.replace("num__", "")
        name = name.replace("cat__", "")
        name = name.replace("City_", "City: ")
        name = name.replace("_", " ")

        return name

    importance_df["Feature"] = (
        importance_df["Feature"].apply(clean_feature_name)
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.head(10).set_index("Feature"),
        horizontal=True
    )

    st.caption(
        "Higher importance means the feature contributed "
        "more to the trained model's decisions. "
        "It does not establish causation."
    )

except Exception as error:

    st.warning(
        f"Unable to display feature importance: {error}"
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.markdown("""
<div class="footer">

<h2 style="color:#E8A0BA;">🌦️ WeatherWise</h2>

<p>Smart Weather & Rain Prediction</p>

<p>Powered by Python | Pandas | Scikit-learn | Streamlit</p>

<p>Machine Learning Project</p>

</div>
""", unsafe_allow_html=True)
