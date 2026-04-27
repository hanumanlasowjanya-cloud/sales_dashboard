import streamlit as st
import pandas as pd
import plotly.express as px

from model import train_models

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("../data/superstore.csv", encoding='latin1')

    df.columns = df.columns.str.encode('ascii', 'ignore').str.decode('ascii')
    df.columns = df.columns.str.replace('.', ' ')
    df.columns = df.columns.str.strip()

    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df = df.dropna(subset=['Order Date'])

    return df

df = load_data()

# ---------------------------
# TITLE
# ---------------------------
st.title("📊 Sales Dashboard with ML")

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "⚖️ Model Comparison"])

# =========================================================
# 📊 DASHBOARD
# =========================================================
with tab1:
    st.subheader("📊 Interactive Sales Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Sales", f"{df['Sales'].sum():,.0f}")
    col2.metric("📈 Total Profit", f"{df['Profit'].sum():,.0f}")
    col3.metric("📦 Total Orders", df.shape[0])

    # Region chart (BLUE)
    region_sales = df.groupby('Region')['Sales'].sum().reset_index()
    fig1 = px.bar(
        region_sales,
        x='Region',
        y='Sales',
        color='Region',
        color_discrete_sequence=px.colors.sequential.Blues,
        title="Sales by Region"
    )
    st.plotly_chart(fig1, width="stretch")

    # Category chart (GREEN)
    category_sales = df.groupby('Category')['Sales'].sum().reset_index()
    fig2 = px.bar(
        category_sales,
        x='Category',
        y='Sales',
        color='Category',
        color_discrete_sequence=px.colors.sequential.Greens,
        title="Sales by Category"
    )
    st.plotly_chart(fig2, width="stretch")

    # Monthly trend (PURPLE)
    df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()

    fig3 = px.line(
        monthly_sales,
        x='Month',
        y='Sales',
        markers=True,
        title="Monthly Sales Trend"
    )
    fig3.update_traces(line_color="purple")
    st.plotly_chart(fig3, width="stretch")

# =========================================================
# 🤖 PREDICTION
# =========================================================
with tab2:
    st.subheader("🤖 Sales Prediction")

    models, encoders, target_encoder, feature_importance = train_models()

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
        pred = model.predict(input_df)
        result = target_encoder.inverse_transform(pred)
        st.success(f"Predicted Category: {result[0]}")

# =========================================================
# ⚖️ MODEL COMPARISON + FEATURE IMPORTANCE
# =========================================================
with tab3:
    st.subheader("⚖️ Model Comparison")

    models, _, _, feature_importance = train_models()

    model_names = []
    accuracies = []

    for name, (model, acc) in models.items():
        model_names.append(name)
        accuracies.append(acc)

    comparison_df = pd.DataFrame({
        "Model": model_names,
        "Accuracy": accuracies
    })

    st.dataframe(comparison_df)

    # Accuracy chart (ORANGE)
    fig4 = px.bar(
        comparison_df,
        x='Model',
        y='Accuracy',
        color='Model',
        color_discrete_sequence=px.colors.sequential.Oranges,
        title="Model Accuracy Comparison"
    )
    st.plotly_chart(fig4, width="stretch")

    # Feature importance (RED)
    st.subheader("📈 Feature Importance (Random Forest)")

    fig5 = px.bar(
        feature_importance,
        x='Feature',
        y='Importance',
        color='Feature',
        color_discrete_sequence=px.colors.sequential.Reds,
        title="Feature Importance"
    )
    st.plotly_chart(fig5, width="stretch")