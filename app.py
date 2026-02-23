import streamlit as st [cite: 549]
import pandas as pd [cite: 551]
from sklearn.model_selection import train_test_split [cite: 553]
from sklearn.linear_model import LogisticRegression [cite: 555]
from sklearn.preprocessing import StandardScaler [cite: 557]
from sklearn.metrics import accuracy_score [cite: 559]

st.set_page_config(page_title="Titanic Survival", layout="centered") [cite: 566]
st.title(" Titanic Survival Prediction") [cite: 566]

file = st.file_uploader("Upload Titanic CSV", type="csv") [cite: 567]
if file:
    df = pd.read_csv(file) [cite: 571]
    
    # Preprocessing
    df["Age"].fillna(df["Age"].median(), inplace=True) [cite: 574]
    df["Fare"].fillna(df["Fare"].median(), inplace=True) [cite: 576]
    df.dropna(subset=["Embarked"], inplace=True) [cite: 578]
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True) [cite: 581]
    
    features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"] [cite: 585]
    X = df[features] [cite: 586]
    y = df["Survived"] [cite: 588]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) [cite: 591, 594]
    
    scaler = StandardScaler() [cite: 598]
    X_train = scaler.fit_transform(X_train) [cite: 601]
    X_test = scaler.transform(X_test) [cite: 603]
    
    model = LogisticRegression(max_iter=1000) [cite: 606]
    model.fit(X_train, y_train) [cite: 611]
    pred = model.predict(X_test) [cite: 612]
    
    st.success(f"Accuracy: {accuracy_score(y_test, pred):.2f}") [cite: 613]
