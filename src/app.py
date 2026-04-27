import streamlit as st
import pandas as pd
import plotly.express as px

from model import train_models

st.set_page_config(page_title="Sales Dashboard", layout="wide")

# ---------------------------
# LOAD DATA (FIXED PATH ✅)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore.csv", encoding='latin1')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert date
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

    return df

df = load_data()

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "⚖️ Model Comparison"])

# ===========================
# 📊 DASHBOARD TAB
# ===========================
with tab1:
    st.title("📊 Interactive Sales Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{df['Sales'].sum():,.0f}")
    col2.metric("Total Profit", f"{df['Profit'].sum():,.0f}")
    col3.metric("Total Orders", df.shape[0])

    # Sales by Region
    region_sales = df.groupby("Region")["Sales"].sum().reset_index()

    fig1 = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        color="Region",
        title="Sales by Region",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    st.plotly_chart(fig1, width='stretch')

    # Monthly trend
    df['Month'] = df['Order Date'].dt.month
    monthly = df.groupby("Month")["Sales"].sum().reset_index()

    fig2 = px.line(
        monthly,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend",
        color_discrete_sequence=["#FF5733"]
    )

    st.plotly_chart(fig2, width='stretch')

# ===========================
# 🤖 PREDICTION TAB
# ===========================
with tab2:
    st.title("🤖 Sales Category Prediction")

    models, encoders, target_encoder = train_models()

    model_choice = st.selectbox("Select Model", list(models.keys()))
    model, accuracy = models[model_choice]

    st.write(f"Model Accuracy: {accuracy:.2f}")

    region_input = st.selectbox("Region", df['Region'].unique())
    category_input = st.selectbox("Category", df['Category'].unique())
    ship_input = st.selectbox("Ship Mode", df['Ship Mode'].unique())

    region_val = encoders['Region'].transform([region_input])[0]
    category_val = encoders['Category'].transform([category_input])[0]
    ship_val = encoders['Ship Mode'].transform([ship_input])[0]

    input_df = pd.DataFrame(
        [[region_val, category_val, ship_val]],
        columns=['Region', 'Category', 'Ship Mode']
    )

    if st.button("Predict"):
        prediction = model.predict(input_df)
        result = target_encoder.inverse_transform(prediction)
        st.success(f"Predicted Sales Category: {result[0]}")

# ===========================
# ⚖️ MODEL COMPARISON
# ===========================
with tab3:
    st.title("⚖️ Model Comparison")

    models, _, _ = train_models()

    names = []
    scores = []

    for name, (model, acc) in models.items():
        names.append(name)
        scores.append(acc)

    compare_df = pd.DataFrame({
        "Model": names,
        "Accuracy": scores
    })

    fig3 = px.bar(
        compare_df,
        x="Model",
        y="Accuracy",
        color="Model",
        title="Model Accuracy Comparison",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    st.plotly_chart(fig3, width='stretch')