import streamlit as st
import pandas as pd
import plotly.express as px

from model import train_models

st.set_page_config(page_title="Sales Dashboard", layout="wide")

# ---------------------------
# LOAD DATA (ULTIMATE FIX ✅)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore.csv")

    # Clean columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 🔍 AUTO DETECT COLUMNS
    def find_col(keyword):
        for col in df.columns:
            if keyword in col:
                return col
        return None

    sales_col = find_col("sales")
    profit_col = find_col("profit")
    region_col = find_col("region")
    category_col = find_col("category")
    ship_col = find_col("ship")
    date_col = find_col("date")

    # Convert date
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df["month"] = df[date_col].dt.month

    return df, sales_col, profit_col, region_col, category_col, ship_col


df, sales_col, profit_col, region_col, category_col, ship_col = load_data()

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

    col1.metric("Total Sales", f"{df[sales_col].sum():,.0f}")
    col2.metric("Total Profit", f"{df[profit_col].sum():,.0f}")
    col3.metric("Total Orders", df.shape[0])

    # Sales by Region
    region_sales = df.groupby(region_col)[sales_col].sum().reset_index()

    fig1 = px.bar(
        region_sales,
        x=region_col,
        y=sales_col,
        color=region_col,
        title="Sales by Region",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    st.plotly_chart(fig1, width='stretch')

    # Monthly Trend
    if "month" in df.columns:
        monthly = df.groupby("month")[sales_col].sum().reset_index()

        fig2 = px.line(
            monthly,
            x="month",
            y=sales_col,
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

    region_input = st.selectbox("Region", df[region_col].unique())
    category_input = st.selectbox("Category", df[category_col].unique())
    ship_input = st.selectbox("Ship Mode", df[ship_col].unique())

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