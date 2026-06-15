import streamlit as st
import pandas as pd
import joblib
import shap
import json
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ── Load model & artifacts ────────────────────────────────────
@st.cache_resource
def load_model():
    model     = joblib.load('../models/churn_model.pkl')
    explainer = joblib.load('../models/shap_explainer.pkl')
    with open('../models/feature_names.json') as f:
        features = json.load(f)
    return model, explainer, features

model, explainer, features = load_model()

# ── Sidebar — customer input ──────────────────────────────────
st.sidebar.header("Customer profile")

tenure          = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly charges ($)", 18, 120, 65)
contract        = st.sidebar.selectbox("Contract type",
    ["Month-to-month", "One year", "Two year"])
internet        = st.sidebar.selectbox("Internet service",
    ["DSL", "Fiber optic", "No"])
tech_support    = st.sidebar.selectbox("Tech support",
    ["No", "Yes", "No internet service"])
online_security = st.sidebar.selectbox("Online security",
    ["No", "Yes", "No internet service"])
streaming_tv    = st.sidebar.selectbox("Streaming TV",
    ["No", "Yes", "No internet service"])
streaming_movies= st.sidebar.selectbox("Streaming movies",
    ["No", "Yes", "No internet service"])
online_backup   = st.sidebar.selectbox("Online backup",
    ["No", "Yes", "No internet service"])
device_prot     = st.sidebar.selectbox("Device protection",
    ["No", "Yes", "No internet service"])
payment_method  = st.sidebar.selectbox("Payment method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"])
paperless       = st.sidebar.selectbox("Paperless billing", ["Yes", "No"])
gender          = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior          = st.sidebar.selectbox("Senior citizen", ["No", "Yes"])
partner         = st.sidebar.selectbox("Has partner", ["Yes", "No"])
dependents      = st.sidebar.selectbox("Has dependents", ["Yes", "No"])
phone_service   = st.sidebar.selectbox("Phone service", ["Yes", "No"])
multiple_lines  = st.sidebar.selectbox("Multiple lines",
    ["No", "Yes", "No phone service"])

total_charges = monthly_charges * tenure

# Build input DataFrame
input_data = {
    'gender': gender,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_prot,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'ChargesPerMonth': monthly_charges if tenure == 0 else total_charges / tenure,
    'IsNewCustomer': 1 if tenure < 12 else 0,
    'HasMultiServices': sum([
        streaming_tv == "Yes", streaming_movies == "Yes",
        online_backup == "Yes", device_prot == "Yes"
    ])
}
input_df = pd.DataFrame([input_data])

# ── Prediction ────────────────────────────────────────────────
prob  = model.predict_proba(input_df)[0][1]
risk  = "🔴 High" if prob > 0.6 else "🟡 Medium" if prob > 0.3 else "🟢 Low"

# ── Main page ─────────────────────────────────────────────────
st.title("📉 Customer Churn Predictor")
st.markdown("Adjust the customer profile in the sidebar to predict churn risk.")

col1, col2, col3 = st.columns(3)
col1.metric("Churn probability", f"{prob:.1%}")
col2.metric("Risk level", risk)
col3.metric("Monthly charges", f"${monthly_charges}")

st.markdown("---")

# SHAP waterfall explanation
st.subheader("Why this prediction?")
st.caption("The waterfall chart shows which factors push this customer toward or away from churning.")

X_transformed = model[:-1].transform(input_df)
sv = explainer.shap_values(X_transformed)

fig, ax = plt.subplots(figsize=(9, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=sv[0],
        base_values=explainer.expected_value,
        data=X_transformed[0],
        feature_names=features['all']
    ),
    show=False,
    max_display=12
)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Retention recommendations ─────────────────────────────────
st.subheader("Recommended actions")

if prob > 0.6:
    st.error("This customer is high-risk. Consider immediate outreach.")
    if contract == "Month-to-month":
        st.write("🔹 Offer a discount to upgrade to an annual contract")
    if tenure < 12:
        st.write("🔹 Assign a dedicated account manager for first-year customers")
    if tech_support == "No":
        st.write("🔹 Offer a free 3-month TechSupport trial")
elif prob > 0.3:
    st.warning("Medium risk — monitor this customer.")
    st.write("🔹 Include in next loyalty program campaign")
else:
    st.success("This customer is low risk. No immediate action needed.")

# ── Tab: At-risk customers (batch view) ──────────────────────
st.markdown("---")
st.subheader("Batch scoring — top at-risk customers")
st.caption("Upload a CSV of customers to score them all at once.")

uploaded = st.file_uploader("Upload customer CSV", type="csv")
if uploaded:
    batch_df = pd.read_csv(uploaded)
    # apply same feature engineering
    batch_df['TotalCharges']    = pd.to_numeric(batch_df['TotalCharges'], errors='coerce')
    batch_df['TotalCharges'].fillna(batch_df['TotalCharges'].median(), inplace=True)
    batch_df['ChargesPerMonth'] = batch_df['TotalCharges'] / (batch_df['tenure'] + 1)
    batch_df['IsNewCustomer']   = (batch_df['tenure'] < 12).astype(int)
    batch_df['HasMultiServices']= (
        (batch_df['StreamingTV']=='Yes').astype(int) +
        (batch_df['StreamingMovies']=='Yes').astype(int) +
        (batch_df['OnlineBackup']=='Yes').astype(int) +
        (batch_df['DeviceProtection']=='Yes').astype(int)
    )
    if 'customerID' in batch_df.columns:
        ids = batch_df['customerID']
        batch_df.drop(['customerID', 'Churn'], axis=1, errors='ignore', inplace=True)
    else:
        ids = pd.Series(range(len(batch_df)))

    probs = model.predict_proba(batch_df)[:, 1]
    results = pd.DataFrame({
        'Customer ID': ids,
        'Churn Probability': (probs * 100).round(1),
        'Risk Level': ['High' if p>0.6 else 'Medium' if p>0.3 else 'Low' for p in probs]
    }).sort_values('Churn Probability', ascending=False)

    st.dataframe(results.head(50), use_container_width=True)
    st.download_button("Download results CSV",
        results.to_csv(index=False), "churn_scores.csv", "text/csv")