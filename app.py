# ----------------- TABEL HASIL UNTUK SEMUA SPLIT -----------------
st.markdown("### 📑 Hasil Prediksi PO untuk Semua Split Data")
split_ratios = {"70:30": 0.3, "80:20": 0.2, "90:10": 0.1}
prediksi_list = []

for name, ratio in split_ratios.items():
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, stratify=y, test_size=ratio, random_state=42)

    # Model C4.5
    model_c45_s = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model_c45_s.fit(X_train_s, y_train_s)
    y_pred_c45_s = model_c45_s.predict(X_test_s)
    c45_tercapai_s = int((y_pred_c45_s == 1).sum())
    c45_tidak_s = int((y_pred_c45_s == 0).sum())

    # Model Naive Bayes
    model_nb_s = GaussianNB()
    model_nb_s.fit(X_train_s, y_train_s)
    y_pred_nb_s = model_nb_s.predict(X_test_s)
    nb_tercapai_s = int((y_pred_nb_s == 1).sum())
    nb_tidak_s = int((y_pred_nb_s == 0).sum())

    prediksi_list.append({"Split": name, "Model": "C4.5", "Tercapai": c45_tercapai_s, "Tidak Tercapai": c45_tidak_s})
    prediksi_list.append({"Split": name, "Model": "Naive Bayes", "Tercapai": nb_tercapai_s, "Tidak Tercapai": nb_tidak_s})

# Buat DataFrame
df_split_prediksi = pd.DataFrame(prediksi_list)

# Tampilkan tabel
st.dataframe(df_split_prediksi.style.set_properties(**{'text-align': 'center'}).highlight_max(subset=["Tercapai"], color="lightgreen"))

# ----------------- TABEL EVALUASI UNTUK SPLIT YANG DIPILIH -----------------
st.markdown("<h3 style='color:#81C784;'>📄 Tabel Evaluasi Lengkap (Split Pilihan)</h3>", unsafe_allow_html=True)
st.dataframe(df_eval.style.highlight_max(axis=0, color='lightgreen'))
