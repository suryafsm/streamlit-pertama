# ============================================================
# APLIKASI MACHINE LEARNING DENGAN STREAMLIT
# Regresi & Klasifikasi - Toy Datasets dari scikit-learn
# ============================================================

# --- Import library ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

# Scikit-learn: datasets
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score

# Scikit-learn: model regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Scikit-learn: model klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Scikit-learn: metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Filter warnings biar output bersih
warnings.filterwarnings("ignore")

# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="Regresi / Klasifikasi ML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FUNGSI: Load Dataset dari sklearn
# ============================================================
@st.cache_data
def load_dataset(nama_dataset, task):
    """
    Load dataset sklearn berdasarkan nama dan jenis task.
    Return: X (fitur), y (target), feature_names, target_names
    """
    if task == "Regresi":
        if nama_dataset == "Diabetes":
            data = datasets.load_diabetes()
        elif nama_dataset == "California Housing":
            data = datasets.fetch_california_housing()
        else:
            raise ValueError(f"Dataset regresi '{nama_dataset}' tidak dikenali.")

        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        target_names = ["target"]

    else:  # Klasifikasi
        if nama_dataset == "Iris":
            data = datasets.load_iris()
        elif nama_dataset == "Wine":
            data = datasets.load_wine()
        elif nama_dataset == "Breast Cancer":
            data = datasets.load_breast_cancer()
        elif nama_dataset == "Digits":
            data = datasets.load_digits()
        else:
            raise ValueError(f"Dataset klasifikasi '{nama_dataset}' tidak dikenali.")

        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        # Untuk Digits, target_names = angka 0-9
        if hasattr(data, "target_names"):
            target_names = list(data.target_names)
        else:
            target_names = [str(i) for i in np.unique(y)]

    return X, y, feature_names, target_names


# ============================================================
# FUNGSI: Ambil Model berdasarkan Nama
# ============================================================
def get_model(nama_model, task):
    """Return instance model sklearn sesuai nama."""
    if task == "Regresi":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        }
    else:  # Klasifikasi
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Support Vector Machine": SVC(probability=True, random_state=42),
        }

    return models.get(nama_model)


# ============================================================
# HEADER APLIKASI
# ============================================================
st.title("🤖 REGRESI / KLASIFIKASI MACHINE LEARNING")
st.caption("Aplikasi demo Machine Learning menggunakan dataset dari scikit-learn")

st.divider()


# ============================================================
# SIDEBAR: Pengaturan
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    st.caption("Atur parameter di bawah, lalu klik **Train Model**")

    # 1. Pilih task
    task = st.radio(
        "🎯 Pilih Jenis Model",
        options=["Regresi", "Klasifikasi"],
        horizontal=True,
    )

    # 2. Pilih dataset (berubah sesuai task)
    if task == "Regresi":
        dataset_options = ["Diabetes", "California Housing"]
    else:
        dataset_options = ["Iris", "Wine", "Breast Cancer", "Digits"]

    nama_dataset = st.selectbox("📊 Pilih Dataset", options=dataset_options)

    # 3. Pilih model
    if task == "Regresi":
        model_options = [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Decision Tree",
            "Random Forest",
        ]
    else:
        model_options = [
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine",
        ]

    nama_model = st.selectbox("🧠 Pilih Model", options=model_options)

    # 4. Test size slider
    test_size = st.slider(
        "📐 Test Size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Proporsi data untuk testing (0.2 = 20%)",
    )

    # 5. Random state
    random_state = st.number_input(
        "🎲 Random State",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    st.divider()

    # 6. Tombol Train
    train_button = st.button("🚀 Train Model", use_container_width=True, type="primary")

    st.divider()
    st.caption("💡 Tips: coba berbagai kombinasi dataset & model!")


# ============================================================
# SESSION STATE: simpan hasil training
# ============================================================
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "cv_score" not in st.session_state:
    st.session_state.cv_score = None
if "task" not in st.session_state:
    st.session_state.task = None
if "nama_dataset" not in st.session_state:
    st.session_state.nama_dataset = None
if "nama_model" not in st.session_state:
    st.session_state.nama_model = None
if "target_names" not in st.session_state:
    st.session_state.target_names = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "train_time" not in st.session_state:
    st.session_state.train_time = None


# ============================================================
# PROSES TRAINING (saat tombol diklik)
# ============================================================
if train_button:
    try:
        # Load dataset
        with st.spinner("📥 Memuat dataset..."):
            X, y, feature_names, target_names = load_dataset(nama_dataset, task)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Ambil model
        model = get_model(nama_model, task)
        if model is None:
            st.error(f"❌ Model '{nama_model}' tidak ditemukan.")
            st.stop()

        # Training
        with st.spinner(f"🧠 Training {nama_model}..."):
            t0 = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            train_time = time.time() - t0

        # Hitung metrics
        metrics = {}
        if task == "Regresi":
            metrics["MAE"] = mean_absolute_error(y_test, y_pred)
            metrics["MSE"] = mean_squared_error(y_test, y_pred)
            metrics["RMSE"] = np.sqrt(metrics["MSE"])
            metrics["R2"] = r2_score(y_test, y_pred)
            metrics["MAPE"] = mean_absolute_percentage_error(y_test, y_pred)
        else:
            metrics["Accuracy"] = accuracy_score(y_test, y_pred)
            metrics["Precision"] = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            metrics["Recall"] = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            metrics["F1-Score"] = f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            )

        # Cross-validation (5-fold)
        with st.spinner("🔄 Cross-validation 5-fold..."):
            cv_scores = cross_val_score(model, X, y, cv=5)
            cv_score = (cv_scores.mean(), cv_scores.std())

        # Simpan ke session_state
        st.session_state.is_trained = True
        st.session_state.model = model
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.metrics = metrics
        st.session_state.cv_score = cv_score
        st.session_state.task = task
        st.session_state.nama_dataset = nama_dataset
        st.session_state.nama_model = nama_model
        st.session_state.target_names = target_names
        st.session_state.feature_names = feature_names
        st.session_state.train_time = train_time

        st.success(f"✅ Model berhasil ditraining dalam {train_time:.3f} detik!")

    except Exception as e:
        st.error(f"❌ Terjadi error: {e}")
        st.stop()


# ============================================================
# BODY: Tampilkan hasil
# ============================================================
if not st.session_state.is_trained:
    # Belum training
    st.info(
        "👈 **Silakan atur parameter di sidebar, lalu klik tombol `🚀 Train Model`** "
        "untuk memulai training."
    )

    # Placeholder info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Dataset")
        st.write("Pilih dari toy datasets scikit-learn")
    with col2:
        st.markdown("### 🧠 Model")
        st.write("5 model regresi & 5 model klasifikasi")
    with col3:
        st.markdown("### 📈 Evaluasi")
        st.write("Metrics lengkap + visualisasi")

    st.stop()


# --- Info dataset di atas tab ---
task = st.session_state.task
nama_dataset = st.session_state.nama_dataset
nama_model = st.session_state.nama_model

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Task", task)
with col2:
    st.metric("📊 Dataset", nama_dataset)
with col3:
    st.metric("🧠 Model", nama_model)
with col4:
    st.metric("⏱️ Training Time", f"{st.session_state.train_time:.3f}s")

st.divider()

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Training & Visualisasi", "📈 Evaluation Metrics"])


# ============================================================
# TAB 1: Training & Visualisasi
# ============================================================
with tab1:
    st.subheader("📊 Hasil Visualisasi")

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    feature_names = st.session_state.feature_names
    target_names = st.session_state.target_names
    model = st.session_state.model

    # Info ringkas
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Data Training:** {X_train.shape[0]} sampel")
    with col2:
        st.info(f"**Data Testing:** {X_test.shape[0]} sampel")

    st.divider()

    # ============ VISUALISASI REGRESI ============
    if task == "Regresi":
        col_a, col_b = st.columns(2)

        # Grafik 1: Scatter Actual vs Predicted
        with col_a:
            st.markdown("#### 📈 Actual vs Predicted (Scatter)")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, y_pred, alpha=0.5, color="#1f77b4", edgecolor="k", s=40)

            # Garis diagonal (perfect prediction)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                lw=2,
                label="Perfect Prediction",
            )
            ax.set_xlabel("Nilai Aktual")
            ax.set_ylabel("Nilai Prediksi")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        # Grafik 2: Line per Index
        with col_b:
            st.markdown("#### 📉 Perbandingan per Index (50 pertama)")
            fig, ax = plt.subplots(figsize=(6, 5))
            n_show = min(50, len(y_test))
            ax.plot(
                range(n_show),
                np.array(y_test)[:n_show],
                marker="o",
                label="Actual",
                color="#1f77b4",
                linewidth=2,
            )
            ax.plot(
                range(n_show),
                np.array(y_pred)[:n_show],
                marker="x",
                label="Predicted",
                color="#ff7f0e",
                linewidth=2,
            )
            ax.set_xlabel("Index")
            ax.set_ylabel("Nilai")
            ax.set_title("Actual vs Predicted per Index")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        # Residual plot (bonus)
        st.markdown("#### 📊 Residual Plot")
        fig, ax = plt.subplots(figsize=(10, 4))
        residuals = np.array(y_test) - np.array(y_pred)
        ax.scatter(y_pred, residuals, alpha=0.5, color="#2ca02c", edgecolor="k")
        ax.axhline(y=0, color="red", linestyle="--", lw=2)
        ax.set_xlabel("Nilai Prediksi")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title("Residual Plot — Semakin acak, semakin baik")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ============ VISUALISASI KLASIFIKASI ============
    else:
        col_a, col_b = st.columns(2)

        # Grafik 1: Confusion Matrix
        with col_a:
            st.markdown("#### 🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names,
                ax=ax,
                cbar_kws={"label": "Jumlah"},
            )
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            plt.close(fig)

        # Grafik 2: Feature Importance (kalau model mendukung)
        with col_b:
            st.markdown("#### 🌟 Feature Importance")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                idx = np.argsort(importances)

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.barh(
                    range(len(idx)),
                    importances[idx],
                    color="#ff7f0e",
                    edgecolor="k",
                )
                ax.set_yticks(range(len(idx)))
                ax.set_yticklabels([feature_names[i] for i in idx])
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance")
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)
                plt.close(fig)
            elif hasattr(model, "coef_"):
                # Untuk Logistic Regression / SVM linear
                coef = np.abs(model.coef_)
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
                idx = np.argsort(coef)

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.barh(range(len(idx)), coef[idx], color="#ff7f0e", edgecolor="k")
                ax.set_yticks(range(len(idx)))
                ax.set_yticklabels([feature_names[i] for i in idx])
                ax.set_xlabel("|Coefficient|")
                ax.set_title("Feature Coefficient (abs)")
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info(
                    "ℹ️ Model ini tidak menyediakan feature importance "
                    "(contoh: KNN, SVM dengan kernel RBF)."
                )

        # Distribusi kelas
        st.markdown("#### 📊 Distribusi Kelas (Data Testing)")
        fig, ax = plt.subplots(figsize=(10, 3))
        unique, counts = np.unique(y_test, return_counts=True)
        labels = [target_names[i] if i < len(target_names) else str(i) for i in unique]
        ax.bar(labels, counts, color="#1f77b4", edgecolor="k")
        ax.set_xlabel("Kelas")
        ax.set_ylabel("Jumlah")
        ax.set_title("Distribusi Kelas di Data Testing")
        for i, v in enumerate(counts):
            ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)


# ============================================================
# TAB 2: Evaluation Metrics
# ============================================================
with tab2:
    st.subheader("📈 Evaluation Metrics")

    metrics = st.session_state.metrics
    task = st.session_state.task
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    cv_score = st.session_state.cv_score

    if task == "Regresi":
        # Metric cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📏 MAE", f"{metrics['MAE']:.4f}")
        with col2:
            st.metric("📐 MSE", f"{metrics['MSE']:.4f}")
        with col3:
            st.metric("📊 RMSE", f"{metrics['RMSE']:.4f}")
        with col4:
            st.metric("🎯 R² Score", f"{metrics['R2']:.4f}")
        with col5:
            st.metric("📉 MAPE", f"{metrics['MAPE']*100:.2f}%")

        st.divider()

        # Interpretasi R²
        r2 = metrics["R2"]
        if r2 >= 0.9:
            st.success(f"🌟 **R² = {r2:.4f}** — Model SANGAT BAIK (menjelaskan {r2*100:.1f}% variansi data)")
        elif r2 >= 0.7:
            st.info(f"✅ **R² = {r2:.4f}** — Model BAIK (menjelaskan {r2*100:.1f}% variansi data)")
        elif r2 >= 0.5:
            st.warning(f"⚠️ **R² = {r2:.4f}** — Model CUKUP (menjelaskan {r2*100:.1f}% variansi data)")
        else:
            st.error(f"❌ **R² = {r2:.4f}** — Model KURANG BAIK (hanya menjelaskan {r2*100:.1f}% variansi data)")

        # Penjelasan metrics
        with st.expander("📖 Penjelasan Metrics"):
            st.markdown(
                """
                | Metric | Penjelasan | Semakin |
                |--------|-----------|---------|
                | **MAE** | Rata-rata error absolut | Kecil = bagus |
                | **MSE** | Rata-rata error kuadrat | Kecil = bagus |
                | **RMSE** | Akar dari MSE (satuan sama dengan target) | Kecil = bagus |
                | **R²** | Proporsi variansi yang dijelaskan model | Mendekati 1 = bagus |
                | **MAPE** | Rata-rata error dalam persen | Kecil = bagus |
                """
            )

    else:  # Klasifikasi
        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['Accuracy']*100:.2f}%")
        with col2:
            st.metric("🎪 Precision", f"{metrics['Precision']*100:.2f}%")
        with col3:
            st.metric("🔍 Recall", f"{metrics['Recall']*100:.2f}%")
        with col4:
            st.metric("⚖️ F1-Score", f"{metrics['F1-Score']*100:.2f}%")

        st.divider()

        # Interpretasi akurasi
        acc = metrics["Accuracy"]
        if acc >= 0.95:
            st.success(f"🌟 **Akurasi = {acc*100:.2f}%** — Model SANGAT BAIK")
        elif acc >= 0.85:
            st.info(f"✅ **Akurasi = {acc*100:.2f}%** — Model BAIK")
        elif acc >= 0.70:
            st.warning(f"⚠️ **Akurasi = {acc*100:.2f}%** — Model CUKUP")
        else:
            st.error(f"❌ **Akurasi = {acc*100:.2f}%** — Model KURANG BAIK")

        # Classification Report
        st.markdown("### 📋 Classification Report")
        target_names = st.session_state.target_names
        report = classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(
            df_report.style.format("{:.4f}"),
            use_container_width=True,
        )

        # Penjelasan metrics
        with st.expander("📖 Penjelasan Metrics"):
            st.markdown(
                """
                | Metric | Penjelasan | Semakin |
                |--------|-----------|---------|
                | **Accuracy** | Proporsi prediksi benar dari total | Mendekati 1 = bagus |
                | **Precision** | Dari yang diprediksi positif, berapa yang benar | Mendekati 1 = bagus |
                | **Recall** | Dari yang aktual positif, berapa yang berhasil diprediksi | Mendekati 1 = bagus |
                | **F1-Score** | Harmonic mean Precision & Recall | Mendekati 1 = bagus |
                """
            )

    st.divider()

    # ============ Cross-Validation ============
    st.markdown("### 🔄 Cross-Validation (5-Fold)")
    cv_mean, cv_std = cv_score

    col1, col2 = st.columns(2)
    with col1:
        st.metric("CV Score (Mean)", f"{cv_mean:.4f}")
    with col2:
        st.metric("CV Score (Std)", f"±{cv_std:.4f}")

    st.caption(
        "Cross-validation membagi data jadi 5 bagian, training 5x, "
        "dan rata-rata hasilnya. Std kecil = model stabil."
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("🤖 Dibuat dengan ❤️ menggunakan Streamlit + scikit-learn")
st.caption("📚 Dataset: toy datasets dari scikit-learn")