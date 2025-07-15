import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# --- Config page ---
st.set_page_config(page_title="Telecom Customer Retention Dashboard - Huawei", layout="wide")

# --- Thème dynamique CSS (clair/sombre selon navigateur) ---
dynamic_theme = """
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    text-align: center;
    transition: background-color 0.4s ease, color 0.4s ease;
}

/* Thème clair */
@media (prefers-color-scheme: light) {
    body {
        background-color: #f9f9f9;
        color: #222;
    }
    h1, h2, h3 {
        color: #0078d7;
        margin-top: 25px;
    }
    .stMetric {
        background: #e1eaff;
        border-radius: 12px;
        padding: 15px;
        color: #222;
        text-align: center;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f0f0;
    }
    .custom-header {
        background-color: #dbe9f4;
        color: #0078d7;
    }
    .menu-container {
        background-color: #f9f9f9 !important;
    }
    .menu-icon {
        color: #0078d7 !important;
    }
    .menu-selected {
        background-color: #cce4f7 !important;
    }
}

/* Thème sombre */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3 {
        color: #ffc107;
        margin-top: 25px;
    }
    .stMetric {
        background: #1c1f26;
        border-radius: 12px;
        padding: 15px;
        color: white;
        text-align: center;
    }
    [data-testid="stSidebar"] {
        background-color: #1f2128;
    }
    .custom-header {
        background-color: #161a23;
        color: #ffc107;
    }
    .menu-container {
        background-color: #0e1117 !important;
    }
    .menu-icon {
        color: white !important;
    }
    .menu-selected {
        background-color: #2c313a !important;
    }
}

img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
</style>
"""
st.markdown(dynamic_theme, unsafe_allow_html=True)

# --- Logo dynamique clair/sombre ---
logo_css = """
<style>
.logo-light {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
}
.logo-dark {
    display: none;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
}

@media (prefers-color-scheme: dark) {
    .logo-light {
        display: none;
    }
    .logo-dark {
        display: block;
    }
}
</style>
"""

# Appliquer le CSS
st.sidebar.markdown(logo_css, unsafe_allow_html=True)

# Afficher les deux logos (le bon sera automatiquement visible selon le mode)
st.sidebar.markdown(
    """
    <img src="Images/Huawei.png" class="logo-light">
    <img src="Images/Huawei1.png" class="logo-dark">
    """,
    unsafe_allow_html=True
)


# --- Header centré et responsive ---
st.markdown("""
<div class="custom-header" style="padding:20px; border-radius:10px; margin-bottom:30px; text-align:center">
<h1 style="margin:0; font-size:2rem;">📡 Telecom Customer Retention Dashboard - Huawei</h1>
</div>
""", unsafe_allow_html=True)

# --- Menu horizontal avec classes CSS dynamiques ---
selected = option_menu(
    menu_title=None,
    options=["Accueil", "Exploration", "Prédiction", "Simulation", "À propos"],
    icons=["house", "bar-chart-line", "stars", "cpu", "person"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "class": "menu-container"},
        "icon": {"class": "menu-icon", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#6c757d"
        },
        "nav-link-selected": {"class": "menu-selected"},
    }
)


# --- Chargement modèle et données (cache) ---
@st.cache_resource
def load_model():
    return joblib.load("modelHW.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

model = load_model()
pipeline = joblib.load("preprocessing_pipeline.pkl")
data = load_data()

# --- Pages ---

if selected == "Accueil":
    st.subheader("Objectif du projet")
    st.markdown("""
    Ce dashboard simule une solution de Data Science appliquée au **secteur des télécommunications** en Afrique.

    💡 **Objectif** : Aider les opérateurs à anticiper les départs de clients afin d’optimiser leurs actions de fidélisation. Pensé comme un levier stratégique, il s’intègre parfaitement aux enjeux d’un acteur tel que **Huawei**, en renforçant sa capacité d’analyse client et de prise de décision.
    """)
    

elif selected == "Exploration":
    st.subheader("📊 Analyse exploratoire")
    
    st.markdown("### 🔍 Aperçu du dataset")
    st.dataframe(data.head())

    st.markdown("### 📈 Taux de churn par type de contrat")
    fig, ax = plt.subplots()
    sns.barplot(x='Contract', y='Churn', data=data.replace({'Churn': {'Yes': 1, 'No': 0}}), ax=ax)
    st.pyplot(fig)

    st.markdown("### 🔗 Corrélations entre variables")
    fig, ax = plt.subplots(figsize=(10, 5))
    corr = data.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif selected == "Prédiction":
    st.subheader("🔮 Prédire le churn d’un client")
    with st.form("form_churn"):
        col1, col2 = st.columns(2)

        gender = col1.selectbox("Genre", ["Female", "Male"])
        seniorcitizen = col2.selectbox("SeniorCitizen", ["Non", "Oui"])
        partner = col1.selectbox("Partenaire", ["Yes", "No"])
        dependents = col2.selectbox("Personnes à charge", ["Yes", "No"])
        tenure = col1.slider("Ancienneté (mois)", 0, 72, 12)
        monthly_charges = col2.slider("Facturation mensuelle (USD)", 0.0, 200.0, 50.0)
        phone_service = col1.selectbox("Téléphonie", ["Yes", "No"])
        multiple_lines = col2.selectbox("Lignes multiples", ["Yes", "No", "No phone service"])
        internet_service = col1.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        online_security = col2.selectbox("Sécurité", ["Yes", "No", "No internet service"])
        online_backup = col1.selectbox("Sauvegarde", ["Yes", "No", "No internet service"])
        device_protection = col2.selectbox("Protection", ["Yes", "No", "No internet service"])
        tech_support = col1.selectbox("Support technique", ["Yes", "No", "No internet service"])
        streaming_tv = col2.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = col1.selectbox("Streaming Films", ["Yes", "No", "No internet service"])
        contract = col2.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
        paperless_billing = col1.selectbox("Sans papier", ["Yes", "No"])
        payment_method = col2.selectbox("Paiement", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        total_charges = st.number_input("Total des charges (USD)", min_value=0.0)

        submit = st.form_submit_button("Prédire")

    if submit:
        input_df = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": seniorcitizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])
        input_processed = pipeline.transform(input_df)
        prediction = model.predict_proba(input_processed)[0][1]

        st.metric("Probabilité de churn", f"{prediction*100:.2f}%")
        if prediction > 0.5:
            st.warning("⚠️ Risque élevé de désabonnement.")
        else:
            st.success("✅ Client probablement fidèle.")

        # --- Résumé automatique ---
        def generate_insights(df):
            insights = []
            if df["tenure"].values[0] < 12:
                insights.append("L'ancienneté client est faible, ce qui augmente le risque.")
            if df["Contract"].values[0] == "Month-to-month":
                insights.append("Le client est en contrat mensuel, plus susceptible de churner.")
            if df["PaymentMethod"].values[0] == "Electronic check":
                insights.append("Méthode de paiement par chèque électronique, liée à un risque plus élevé.")
            if df["SeniorCitizen"].values[0] == "Oui":
                insights.append("Le client est senior, un segment à surveiller.")
            if df["OnlineSecurity"].values[0] == "No":
                insights.append("Pas de sécurité en ligne, ce qui peut être un facteur de churn.")
            if len(insights) == 0:
                insights.append("Le client présente un profil stable sans facteurs à risque majeurs.")
            return insights

        with st.expander("🧠 Résumé des insights sur ce client"):
            for i in generate_insights(input_df):
                st.write(f"- {i}")

elif selected == "Simulation":
    st.subheader("🧪 Simulation d’une action marketing")

    default_input = {
        "gender": "Female", "SeniorCitizen": "Non", "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0, "TotalCharges": 840.0
    }

    col1, col2 = st.columns(2)
    new_contract = col1.selectbox("📃 Type de contrat", ["Month-to-month", "One year", "Two year"])
    new_method = col2.selectbox("💳 Méthode de paiement", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    new_tenure = st.slider("📅 Ancienneté", 0, 72, default_input["tenure"])
    new_monthly = st.slider("💸 Facture mensuelle", 0.0, 200.0, default_input["MonthlyCharges"])

    default_input["Contract"] = new_contract
    default_input["PaymentMethod"] = new_method
    default_input["tenure"] = new_tenure
    default_input["MonthlyCharges"] = new_monthly

    sim_df = pd.DataFrame([default_input])
    sim_processed = pipeline.transform(sim_df)
    new_pred = model.predict_proba(sim_processed)[0][1]

    st.metric("🧮 Nouveau risque de churn", f"{new_pred*100:.2f}%")
    if new_pred > 0.5:
        st.warning("⚠️ Cette combinaison montre un risque élevé.")
    else:
        st.success("✅ Cette action semble bénéfique.")

elif selected == "À propos":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("Images/me.jpg")
    with col2:
        st.caption("Réalisé par :")
        st.markdown("👤 **ALOTSE Christy**")
        st.markdown("""
        **Data Scientist** spécialisée dans le traitement de données, la visualisation analytique et le développement de modèles de machine learning adaptés aux enjeux métier.
        
        💼 [LinkedIn](https://www.linkedin.com/in/christy-alotse)  
        📧 [Me contacter](mailto:alotsechristy@gmail.com)
        """)
