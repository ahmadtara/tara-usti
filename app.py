import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Streamlit UI ===
st.title("ML App: EDA + Decision Tree & Naive Bayes")
st.write("Upload file Excel, lakukan preprocessing, training model, dan lihat hasil visualisasi.")

# === Upload File ===
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
if uploaded_file is not None:
    # Baca data
    df = pd.read_excel(uploaded_file)
    st.subheader("Preview Data")
    st.write(df.head())

    # === EDA ===
    st.subheader("Exploratory Data Analysis")
    st.write("**Shape Data:**", df.shape)
    st.write("**Info:**")
    st.write(df.describe())

    # Heatmap korelasi
    st.write("### Korelasi Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribusi kolom
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        col = st.selectbox("Pilih kolom untuk distribusi:", numeric_cols)
        fig2, ax2 = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax2)
        st.pyplot(fig2)

    # === Preprocessing ===
    st.subheader("Preprocessing Data")
    target_col = st.selectbox("Pilih kolom target:", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Label Encoding untuk target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Handling categorical di X
    X_encoded = pd.get_dummies(X)

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # === Training Models ===
    st.subheader("Training Models")

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    # === Evaluation ===
    st.write("### Akurasi:")
    st.write(f"Decision Tree: {accuracy_score(y_test, y_pred_dt):.2f}")
    st.write(f"Naive Bayes: {accuracy_score(y_test, y_pred_nb):.2f}")

    # Classification Report
    st.write("### Classification Report (Decision Tree)")
    st.text(classification_report(y_test, y_pred_dt))

    st.write("### Classification Report (Naive Bayes)")
    st.text(classification_report(y_test, y_pred_nb))

    # Confusion Matrix
    st.write("### Confusion Matrix (Decision Tree)")
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", ax=ax3)
    st.pyplot(fig3)

    st.write("### Confusion Matrix (Naive Bayes)")
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens", ax=ax4)
    st.pyplot(fig4)

else:
    st.info("Silakan upload file Excel terlebih dahulu.")
