import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Judul Aplikasi ===
st.title("📊 ML App: C4.5 & Naive Bayes")
st.write("Pilih sumber data (GitHub URL atau Upload File) → Preprocessing → Training → Evaluasi")

# === Input Pilihan Data ===
data_source = st.radio("Pilih sumber data:", ["Upload File", "GitHub URL"])

df = None
if data_source == "Upload File":
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file)
elif data_source == "GitHub URL":
    github_url = st.text_input("Masukkan URL RAW file Excel dari GitHub:")
    if github_url:
        try:
            xls = pd.ExcelFile(github_url)
        except Exception as e:
            st.error(f"Gagal mengambil file dari URL: {e}")
            st.stop()

# === Jika file tersedia ===
if 'xls' in locals():
    sheet_names = xls.sheet_names
    selected_sheet = st.selectbox("Pilih Sheet:", sheet_names)

    # Baca data
    df = pd.read_excel(xls, sheet_name=selected_sheet)
    st.subheader("Preview Data")
    st.write(df.head())

    st.write(f"**Jumlah Data:** {df.shape[0]} baris, {df.shape[1]} kolom")

    # Pilih kolom target
    target_col = st.selectbox("Pilih kolom target:", df.columns)

    # Drop NaN hanya di target
    df = df.dropna(subset=[target_col])
    if df.shape[0] == 0:
        st.error("Dataset kosong setelah membersihkan NaN di target.")
        st.stop()

    # Pisahkan fitur dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Label encoding target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # One-hot encoding fitur kategorikal
    X_encoded = pd.get_dummies(X).apply(pd.to_numeric, errors='coerce').fillna(0)

    if X_encoded.shape[1] == 0:
        st.error("Tidak ada fitur yang valid untuk model.")
        st.stop()

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # === Model C4.5 ===
    st.subheader("🌳 Model C4.5 (Decision Tree)")
    c45 = DecisionTreeClassifier(criterion="entropy", random_state=42)
    c45.fit(X_train, y_train)
    y_pred_c45 = c45.predict(X_test)

    st.write(f"**Akurasi C4.5:** {accuracy_score(y_test, y_pred_c45):.2f}")
    st.text("Classification Report (C4.5):")
    st.text(classification_report(y_test, y_pred_c45))

    cm_c45 = confusion_matrix(y_test, y_pred_c45)
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm_c45, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("Confusion Matrix - C4.5")
    st.pyplot(fig1)

    # === Model Naive Bayes ===
    st.subheader("🤖 Model Naive Bayes")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    st.write(f"**Akurasi Naive Bayes:** {accuracy_score(y_test, y_pred_nb):.2f}")
    st.text("Classification Report (Naive Bayes):")
    st.text(classification_report(y_test, y_pred_nb))

    cm_nb = confusion_matrix(y_test, y_pred_nb)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens", ax=ax2)
    ax2.set_title("Confusion Matrix - Naive Bayes")
    st.pyplot(fig2)
