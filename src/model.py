import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def train_models():
    # ✅ FIXED PATH
    df = pd.read_csv("data/superstore.csv", encoding='latin1')

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Select required columns
    df = df[['region', 'category', 'ship_mode', 'sales']]

    # Create target
    df['sales_category'] = pd.qcut(df['sales'], q=3, labels=['Low', 'Medium', 'High'])

    # Encode
    encoders = {}
    for col in ['region', 'category', 'ship_mode']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col.capitalize() if col != 'ship_mode' else 'Ship Mode'] = le

    target_encoder = LabelEncoder()
    df['sales_category'] = target_encoder.fit_transform(df['sales_category'])

    X = df[['region', 'category', 'ship_mode']]
    y = df['sales_category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {}

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    models['Decision Tree'] = (dt, dt.score(X_test, y_test))

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    models['KNN'] = (knn, knn.score(X_test, y_test))

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    models['Random Forest'] = (rf, rf.score(X_test, y_test))

    return models, encoders, target_encoder