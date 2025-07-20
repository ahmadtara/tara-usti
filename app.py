
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide")
st.title("Analisis Perbandingan Algoritma: C4.5 vs Naive Bayes Untuk Memprediksi Ketercapaian Target PO Dalam Membangun Project FTTH")

# Upload file
uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx", "csv"])
if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded_file)
    else:
        df_raw = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df_raw.head())

    # Preprocessing
    df = df_raw.dropna().copy()
    if "LABEL" not in df.columns:
        st.error("Kolom target 'LABEL' tidak ditemukan dalam data.")
    else:
        X = df.drop(columns=['LABEL'])
        y = df['LABEL']

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col])
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Split configurations
        splits = [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]
        models = {
            'C4.5 (Decision Tree)': DecisionTreeClassifier(criterion='entropy'),
            'Naive Bayes': GaussianNB()
        }

        results = []
        for train_size, test_size in splits:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, test_size=test_size, random_state=42
            )
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results.append({
                    'Model': model_name,
                    'Split': f"{int(train_size*100)}% - {int(test_size*100)}%",
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted'),
                })

        df_results = pd.DataFrame(results)

        # Layout seperti gambar
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Model Base Results")
            metric_base = st.selectbox("Select Metric to Display", ['Accuracy', 'Precision', 'Recall', 'F1 Score'], key="base")
            fig1 = plt.figure(figsize=(8, 5))
            sns.barplot(data=df_results, x="Split", y=metric_base, hue="Model")
            plt.ylim(0, 1.05)
            plt.title(f"Model Performance by {metric_base.lower()}")
            st.pyplot(fig1)

        with col2:
            st.markdown("### Hasil Evaluasi")
            metric_boosted = st.selectbox("Select Metric to Display", ['Accuracy', 'Precision', 'Recall', 'F1 Score'], key="boosted")
            fig2 = plt.figure(figsize=(8, 5))
            sns.barplot(data=df_results, x="Split", y=metric_boosted, hue="Model")
            plt.ylim(0, 1.05)
            plt.title(f"Model Performance by {metric_boosted.lower()}")
            st.pyplot(fig2)
