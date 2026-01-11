import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Diabetes Classification with PCA", layout="wide")

st.title("🩺 Binary Classification with PCA & Model Deployment")

# ======================================================
# a) DATA LOADING, CLEANING & EXPLORATION
# ======================================================
st.header("a) Data Loading, Cleaning & Exploration")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, names=columns)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

st.subheader("Dataset Shape")
st.write("Features (X):", X.shape)
st.write("Target (y):", y.shape)

st.subheader("First 5 Rows")
st.dataframe(df.head())

st.subheader("Summary Statistics")
st.dataframe(df.describe())

st.subheader("Class Distribution")
st.write(y.value_counts())
st.info("Dataset is **imbalanced** (more Non-Diabetic samples)")

# ======================================================
# b) PREPROCESSING, SCALING & SPLIT
# ======================================================
st.header("b) Preprocessing, Scaling & Stratified Split")

invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[invalid_cols] = X[invalid_cols].replace(0, np.nan)
X.fillna(X.median(), inplace=True)

st.success("Invalid zero values replaced with median")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

st.write("Training Set Shape:", X_train.shape)
st.write("Testing Set Shape:", X_test.shape)

# ======================================================
# c) PCA ANALYSIS
# ======================================================
st.header("c) PCA Analysis")

pca_95 = PCA(n_components=0.95)
X_train_pca = pca_95.fit_transform(X_train)
X_test_pca = pca_95.transform(X_test)

pca_99 = PCA(n_components=0.99)
pca_99.fit(X_train)

st.subheader("Number of Components")
st.write("95% Variance:", pca_95.n_components_)
st.write("99% Variance:", pca_99.n_components_)

st.subheader("Explained Variance Ratio (95%)")
st.write(pca_95.explained_variance_ratio_)

# ======================================================
# d) MODEL TRAINING & EVALUATION
# ======================================================
st.header("d) Model Training, Evaluation & Comparison")

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel="rbf")
}

accuracies = {}
best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    st.subheader(f"{name}")
    st.write("Accuracy:", acc)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    if acc > best_acc:
        best_acc = acc
        best_model = model

st.success(f"🏆 Best Model: {best_model.__class__.__name__}")
st.write("Justification: Selected based on highest test accuracy.")

# ======================================================
# e) MODEL DEPLOYMENT (STREAMLIT)
# ======================================================
st.header("e) Model Deployment")

st.subheader("Enter Patient Details")

preg = st.number_input("Pregnancies", 0, 20, 1)
glu = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
ins = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict Diabetes"):
    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca_95.transform(input_scaled)
    prediction = best_model.predict(input_pca)[0]

    if prediction == 1:
        st.error("⚠️ Diabetic (1)")
    else:
        st.success("✅ Non-Diabetic (0)")
