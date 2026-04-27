import streamlit as st
import pandas as pd
import plotly.express as px

from model import train_models

st.set_page_config(page_title="Sales Dashboard", layout="wide")

# ---------------------------
# LOAD DATA (ROBUST FIX ✅)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore.csv", encoding='latin1')

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 🔍 DEBUG: see actual columns (important once)
    st.write("Columns in dataset:", df.columns)

    # ✅ Find correct date column automatically
    date_col = None
    for col in df.columns:
        if "date" in col:
            date_col = col
            break

    # If found → convert
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['month'] = df[date_col].dt.month
    else:
        st.error("No date column found in dataset!")

    return df


# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "⚖️ Model Comparison"])

# ===========================
# 📊 DASHBOARD
# ===========================
with tab1:
    st.title("📊 Interactive Sales Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{df['sales'].sum():,.0f}")
    col2.metric("Total Profit", f"{df['profit'].sum():,.0f}")
    col3.metric("Total Orders", df.shape[0])

    # Sales by Region
    region_sales = df.groupby("region")["sales"].sum().reset_index()

    fig1 = px.bar(
        region_sales,
        x="region",
        y="sales",
        color="region",
        title="Sales by Region",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    st.plotly_chart(fig1, width='stretch')

    # Monthly Trend
    df['month'] = df['order_date'].dt.month
    monthly = df.groupby("month")["sales"].sum().reset_index()

    fig2 = px.line(
        monthly,
        x="month",
        y="sales",
        markers=True,
        title="Monthly Sales Trend",
        color_discrete_sequence=["#FF5733"]
    )

    st.plotly_chart(fig2, width='stretch')

# ===========================
# 🤖 PREDICTION
# ===========================
with tab2:
    st.title("🤖 Sales Category Prediction")

    models, encoders, target_encoder = train_models()

    model_choice = st.selectbox("Select Model", list(models.keys()))
    model, accuracy = models[model_choice]

    st.write(f"Model Accuracy: {accuracy:.2f}")

    region_input = st.selectbox("Region", df['region'].unique())
    category_input = st.selectbox("Category", df['category'].unique())
    ship_input = st.selectbox("Ship Mode", df['ship_mode'].unique())

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