import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# === Konfigurasi Streamlit ===
st.title("📊 Evaluasi Model C4.5 vs Naive Bayes")
st.write("Upload file Excel berisi data PO My Republic.")

# === Upload File ===
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df_raw.head())

    # === Step 3: Data Selection & Preprocessing ===
    df = df_raw.rename(columns={
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster\n(SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    })

    df = df[['topologi', 'vendor', 'hp_cluster', 'status_po']]
    df = df.dropna()
    df['status_po'] = df['status_po'].str.lower().str.strip()
    df['label'] = df['status_po'].apply(lambda x: 1 if x == 'done' else 0)

    # Encoding & Normalisasi
    le_topologi = LabelEncoder()
    le_vendor = LabelEncoder()
    df['topologi_enc'] = le_topologi.fit_transform(df['topologi'].astype(str))
    df['vendor_enc'] = le_vendor.fit_transform(df['vendor'].astype(str))

    scaler = MinMaxScaler()
    df['hp_cluster_norm'] = scaler.fit_transform(df[['hp_cluster']])

    st.write("Data setelah preprocessing:")
    st.dataframe(df.head())

    # Pilihan Split
    split_option = st.selectbox("Pilih Split Ratio", ["80:20", "70:30", "90:10"])
    split_ratios = {"80:20": 0.2, "70:30": 0.3, "90:10": 0.1}
    split = split_ratios[split_option]

    X = df[['topologi_enc', 'vendor_enc', 'hp_cluster_norm']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=split, random_state=42)

    # === Model C4.5 ===
    model_c45 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model_c45.fit(X_train, y_train)
    y_pred_c45 = model_c45.predict(X_test)

    # === Model Naive Bayes ===
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)

    # === Evaluasi ===
    def evaluate_model(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)
        }

    c45_eval = evaluate_model(y_test, y_pred_c45)
    nb_eval = evaluate_model(y_test, y_pred_nb)

    st.subheader("Hasil Evaluasi")
    results = pd.DataFrame([
        {"Model": "C4.5", **c45_eval},
        {"Model": "Naive Bayes", **nb_eval}
    ])
    st.dataframe(results)

    # === Confusion Matrix ===
    st.subheader("Confusion Matrix")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_c45), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("C4.5")
    sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title("Naive Bayes")
    st.pyplot(fig)

    # === Grafik Perbandingan ===
    st.subheader("Perbandingan Akurasi")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=results, x='Model', y='Accuracy', palette='Set2', ax=ax2)
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

    # === Kesimpulan ===
    best_model = results.sort_values(by='Accuracy', ascending=False).iloc[0]
    st.success(f"Model terbaik: **{best_model['Model']}** dengan akurasi {best_model['Accuracy']:.4f}")
