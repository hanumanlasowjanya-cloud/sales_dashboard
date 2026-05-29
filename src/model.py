import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def train_models():

    # Load dataset
    df = pd.read_csv("data/superstore.csv", encoding="latin1")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # -----------------------------
    # AUTO DETECT COLUMNS
    # -----------------------------
    def find_col(keyword):
        for col in df.columns:
            if keyword in col:
                return col
        return None

    region_col = find_col("region")
    category_col = find_col("category")
    ship_col = find_col("ship")
    sales_col = find_col("sales")

    if not all([region_col, category_col, ship_col, sales_col]):
        raise Exception(
            f"Required columns not found.\nAvailable columns:\n{list(df.columns)}"
        )

    # Keep only required columns
    df = df[[region_col, category_col, ship_col, sales_col]].copy()

    # Remove nulls
    df = df.dropna()

    # Create target variable
    df["sales_category"] = pd.qcut(
        df[sales_col],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    # -----------------------------
    # ENCODING
    # -----------------------------
    encoders = {}

    for col in [region_col, category_col, ship_col]:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

    target_encoder = LabelEncoder()

    df["sales_category"] = target_encoder.fit_transform(
        df["sales_category"]
    )

    X = df[[region_col, category_col, ship_col]]

    y = df["sales_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {}

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    models["Decision Tree"] = (
        dt,
        dt.score(X_test, y_test)
    )

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    models["KNN"] = (
        knn,
        knn.score(X_test, y_test)
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)

    models["Random Forest"] = (
        rf,
        rf.score(X_test, y_test)
    )

    return (
        models,
        encoders,
        target_encoder,
        region_col,
        category_col,
        ship_col
    )