import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

st.set_page_config(layout="wide")
st.title("📊 Analisis Prediksi PO My Republic 2024")

# Upload file
uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])
if uploaded_file:
    # Load data awal
    df_raw = pd.read_excel(uploaded_file)
    jumlah_awal = len(df_raw)

    # Preprocessing
    df = df_raw.rename(columns={
        'Topology': 'topologi',
        'Vendor': 'vendor',
        'HP Cluster\n(SND Wajib Isi)': 'hp_cluster',
        'Status PO Cluster (SND Wajib Isi)': 'status_po'
    })[['topologi', 'vendor', 'hp_cluster', 'status_po']].dropna()
    jumlah_setelah = len(df)

    # Info jumlah data awal & setelah preprocessing
    st.info(f"📋 Jumlah data awal: **{jumlah_awal} baris**\n🧹 Setelah preprocessing: **{jumlah_setelah} baris**")

    # Encode variabel
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('status_po_Tercapai', axis=1)
    y = df_encoded['status_po_Tercapai']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    # Info distribusi data train/test
    train_tercapai = int((y_train == 1).sum())
    train_tidak = int((y_train == 0).sum())
    test_tercapai = int((y_test == 1).sum())
    test_tidak = int((y_test == 0).sum())

    st.markdown("### 📌 Distribusi Data Training & Testing")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"**Training Set:** {len(y_train)} data  \n- Tercapai: {train_tercapai}  \n- Tidak: {train_tidak}")
    with col_info2:
        st.markdown(f"**Testing Set:** {len(y_test)} data  \n- Tercapai: {test_tercapai}  \n- Tidak: {test_tidak}")

    # Model
    model_c45 = DecisionTreeClassifier(random_state=42)
    model_nb = GaussianNB()

    model_c45.fit(X_train, y_train)
    model_nb.fit(X_train, y_train)

    # Prediksi testing
    y_pred_c45_test = model_c45.predict(X_test)
    y_pred_nb_test = model_nb.predict(X_test)

    # Prediksi training
    y_pred_c45_train = model_c45.predict(X_train)
    y_pred_nb_train = model_nb.predict(X_train)

    # Hitung jumlah tercapai/tidak (testing)
    c45_tercapai_test = int((y_pred_c45_test == 1).sum())
    c45_tidak_test = int((y_pred_c45_test == 0).sum())
    nb_tercapai_test = int((y_pred_nb_test == 1).sum())
    nb_tidak_test = int((y_pred_nb_test == 0).sum())

    # Hitung jumlah tercapai/tidak (training)
    c45_tercapai_train = int((y_pred_c45_train == 1).sum())
    c45_tidak_train = int((y_pred_c45_train == 0).sum())
    nb_tercapai_train = int((y_pred_nb_train == 1).sum())
    nb_tidak_train = int((y_pred_nb_train == 0).sum())

    # Layout hasil visualisasi
    colA, colB = st.columns(2)

    # Chart C4.5
    with colA:
        st.subheader("🔴 C4.5 Decision Tree")

        st.markdown("**Data Training:**")
        st.write(f"- Tercapai: {c45_tercapai_train}")
        st.write(f"- Tidak: {c45_tidak_train}")
        fig_c45_train, ax_c45_train = plt.subplots(figsize=(2.2, 2))
        sns.barplot(x=['Tercapai', 'Tidak'], y=[c45_tercapai_train, c45_tidak_train],
                    palette=['#4CAF50', '#E53935'], ax=ax_c45_train)
        st.pyplot(fig_c45_train)

        st.markdown("**Data Testing:**")
        st.write(f"- Tercapai: {c45_tercapai_test}")
        st.write(f"- Tidak: {c45_tidak_test}")
        fig_c45_test, ax_c45_test = plt.subplots(figsize=(2.2, 2))
        sns.barplot(x=['Tercapai', 'Tidak'], y=[c45_tercapai_test, c45_tidak_test],
                    palette=['#4CAF50', '#E53935'], ax=ax_c45_test)
        st.pyplot(fig_c45_test)

    # Chart Naive Bayes
    with colB:
        st.subheader("🔵 Naive Bayes")

        st.markdown("**Data Training:**")
        st.write(f"- Tercapai: {nb_tercapai_train}")
        st.write(f"- Tidak: {nb_tidak_train}")
        fig_nb_train, ax_nb_train = plt.subplots(figsize=(2.2, 2))
        sns.barplot(x=['Tercapai', 'Tidak'], y=[nb_tercapai_train, nb_tidak_train],
                    palette=['#4CAF50', '#E53935'], ax=ax_nb_train)
        st.pyplot(fig_nb_train)

        st.markdown("**Data Testing:**")
        st.write(f"- Tercapai: {nb_tercapai_test}")
        st.write(f"- Tidak: {nb_tidak_test}")
        fig_nb_test, ax_nb_test = plt.subplots(figsize=(2.2, 2))
        sns.barplot(x=['Tercapai', 'Tidak'], y=[nb_tercapai_test, nb_tidak_test],
                    palette=['#4CAF50', '#E53935'], ax=ax_nb_test)
        st.pyplot(fig_nb_test)
