import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def train_models():

    df = pd.read_csv("../data/superstore.csv", encoding='latin1')

    # Clean columns
    df.columns = df.columns.str.encode('ascii', 'ignore').str.decode('ascii')
    df.columns = df.columns.str.replace('.', ' ')
    df.columns = df.columns.str.strip()

    # Target
    df['Sales Category'] = pd.qcut(df['Sales'], q=3, labels=['Low', 'Medium', 'High'])

    features = ['Region', 'Category', 'Ship Mode']
    df = df[features + ['Sales Category']].dropna()

    # Encode
    encoders = {}
    for col in features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    df['Sales Category'] = target_encoder.fit_transform(df['Sales Category'])

    X = df[features]
    y = df['Sales Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Models
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier()

    dt.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Accuracy
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    models = {
        "Decision Tree": (dt, dt_acc),
        "KNN": (knn, knn_acc),
        "Random Forest": (rf, rf_acc)
    }

    # Feature importance
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return models, encoders, target_encoder, feature_importance