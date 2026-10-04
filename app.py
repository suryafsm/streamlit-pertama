# ============================================================
# APLIKASI MACHINE LEARNING DENGAN STREAMLIT
# Regresi & Klasifikasi + MongoDB CRUD + Retrain
# ============================================================

# --- Import library ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from datetime import datetime

# Scikit-learn
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

# MongoDB
try:
    import pymongo
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

# Filter warnings
warnings.filterwarnings("ignore")

# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="Regresi / Klasifikasi ML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# KONEKSI MONGODB
# ============================================================
@st.cache_resource
def get_mongo_client():
    """Koneksi ke MongoDB Atlas."""
    if not MONGO_AVAILABLE:
        return None
    try:
        uri = st.secrets["MONGODB_URI"]
        client = MongoClient(uri)
        client.admin.command("ping")
        return client
    except Exception:
        return None


# ============================================================
# FUNGSI: Load Dataset dari sklearn
# ============================================================
@st.cache_data
def load_dataset(nama_dataset, task):
    """Load dataset sklearn sesuai task."""
    if task == "Regresi":
        if nama_dataset == "Diabetes":
            data = datasets.load_diabetes()
        elif nama_dataset == "California Housing":
            data = datasets.fetch_california_housing()
        else:
            raise ValueError(f"Dataset '{nama_dataset}' tidak dikenali.")
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_names = ["target"]
    else:
        if nama_dataset == "Iris":
            data = datasets.load_iris()
        elif nama_dataset == "Wine":
            data = datasets.load_wine()
        elif nama_dataset == "Breast Cancer":
            data = datasets.load_breast_cancer()
        elif nama_dataset == "Digits":
            data = datasets.load_digits()
        else:
            raise ValueError(f"Dataset '{nama_dataset}' tidak dikenali.")
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_names = (
            list(data.target_names) if hasattr(data, "target_names")
            else [str(i) for i in np.unique(y)]
        )
    return X, y, feature_names, target_names


# ============================================================
# FUNGSI: Ambil Model
# ============================================================
def get_model(nama_model, task):
    """Return instance model sklearn."""
    if task == "Regresi":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Support Vector Machine": SVC(probability=True, random_state=42),
        }
    return models.get(nama_model)


# ============================================================
# HEADER
# ============================================================
st.title("🤖 REGRESI / KLASIFIKASI MACHINE LEARNING")
st.caption("Aplikasi demo ML + MongoDB CRUD + Retrain dengan scikit-learn")

st.divider()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    st.caption("Atur parameter di bawah, lalu klik **Train Model**")

    task = st.radio(
        "🎯 Pilih Jenis Model",
        options=["Regresi", "Klasifikasi"],
        horizontal=True,
    )

    if task == "Regresi":
        dataset_options = ["Diabetes", "California Housing"]
    else:
        dataset_options = ["Iris", "Wine", "Breast Cancer", "Digits"]

    nama_dataset = st.selectbox("📊 Pilih Dataset", options=dataset_options)

    if task == "Regresi":
        model_options = [
            "Linear Regression", "Ridge Regression", "Lasso Regression",
            "Decision Tree", "Random Forest",
        ]
    else:
        model_options = [
            "Logistic Regression", "K-Nearest Neighbors", "Decision Tree",
            "Random Forest", "Support Vector Machine",
        ]

    nama_model = st.selectbox("🧠 Pilih Model", options=model_options)

    test_size = st.slider("📐 Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("🎲 Random State", 0, 9999, 42, 1)

    st.divider()
    train_button = st.button("🚀 Train Model", use_container_width=True, type="primary")

    st.divider()
    st.caption("💡 Tips: coba berbagai kombinasi dataset & model!")


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "is_trained": False, "model": None, "X_train": None, "X_test": None,
    "y_train": None, "y_test": None, "y_pred": None, "metrics": {},
    "cv_score": None, "task": None, "nama_dataset": None, "nama_model": None,
    "target_names": None, "feature_names": None, "train_time": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# PROSES TRAINING
# ============================================================
if train_button:
    try:
        with st.spinner("📥 Memuat dataset..."):
            X, y, feature_names, target_names = load_dataset(nama_dataset, task)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = get_model(nama_model, task)
        if model is None:
            st.error(f"❌ Model '{nama_model}' tidak ditemukan.")
            st.stop()

        with st.spinner(f"🧠 Training {nama_model}..."):
            t0 = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            train_time = time.time() - t0

        metrics = {}
        if task == "Regresi":
            metrics["MAE"] = mean_absolute_error(y_test, y_pred)
            metrics["MSE"] = mean_squared_error(y_test, y_pred)
            metrics["RMSE"] = np.sqrt(metrics["MSE"])
            metrics["R2"] = r2_score(y_test, y_pred)
            metrics["MAPE"] = mean_absolute_percentage_error(y_test, y_pred)
        else:
            metrics["Accuracy"] = accuracy_score(y_test, y_pred)
            metrics["Precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["Recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["F1-Score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        with st.spinner("🔄 Cross-validation 5-fold..."):
            cv_scores = cross_val_score(model, X, y, cv=5)
            cv_score = (cv_scores.mean(), cv_scores.std())

        # Simpan ke session_state
        st.session_state.update({
            "is_trained": True, "model": model,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test, "y_pred": y_pred,
            "metrics": metrics, "cv_score": cv_score,
            "task": task, "nama_dataset": nama_dataset, "nama_model": nama_model,
            "target_names": target_names, "feature_names": feature_names,
            "train_time": train_time,
        })

        st.success(f"✅ Model ditraining dalam {train_time:.3f} detik!")

    except Exception as e:
        st.error(f"❌ Terjadi error: {e}")
        st.stop()


# ============================================================
# BODY — Info Ringkas
# ============================================================
if not st.session_state.is_trained:
    st.info(
        "👈 **Silakan atur parameter di sidebar, lalu klik `🚀 Train Model`** "
        "untuk memulai training."
    )

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 📊 Dataset")
    st.write("Pilih dari toy datasets scikit-learn")
with col2:
    st.markdown("### 🧠 Model")
    st.write("5 model regresi & 5 model klasifikasi")
with col3:
    st.markdown("### 📈 Evaluasi")
    st.write("Metrics + visualisasi + MongoDB CRUD")

st.divider()


# ============================================================
# 3 TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Training & Visualisasi",
    "📈 Evaluation Metrics",
    "🗄️ Database & Retrain",
])


# ============================================================
# TAB 1: TRAINING & VISUALISASI
# ============================================================
with tab1:
    if not st.session_state.is_trained:
        st.warning("⚠️ Klik **Train Model** dulu di sidebar.")
        st.stop()

    st.subheader("📊 Hasil Visualisasi")

    task = st.session_state.task
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    feature_names = st.session_state.feature_names
    target_names = st.session_state.target_names
    model = st.session_state.model

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Data Training:** {X_train.shape[0]} sampel")
    with col2:
        st.info(f"**Data Testing:** {X_test.shape[0]} sampel")

    st.divider()

    # ============ REGRESI ============
    if task == "Regresi":
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📈 Actual vs Predicted (Scatter)")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, y_pred, alpha=0.5, color="#1f77b4", edgecolor="k", s=40)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
            ax.set_xlabel("Aktual")
            ax.set_ylabel("Prediksi")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.markdown("#### 📉 Perbandingan per Index (50 pertama)")
            fig, ax = plt.subplots(figsize=(6, 5))
            n = min(50, len(y_test))
            ax.plot(range(n), np.array(y_test)[:n], marker="o", label="Actual", color="#1f77b4")
            ax.plot(range(n), np.array(y_pred)[:n], marker="x", label="Predicted", color="#ff7f0e")
            ax.set_xlabel("Index")
            ax.set_ylabel("Nilai")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### 📊 Residual Plot")
        fig, ax = plt.subplots(figsize=(10, 4))
        residuals = np.array(y_test) - np.array(y_pred)
        ax.scatter(y_pred, residuals, alpha=0.5, color="#2ca02c", edgecolor="k")
        ax.axhline(y=0, color="red", linestyle="--", lw=2)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Residual")
        ax.set_title("Residual Plot")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ============ KLASIFIKASI ============
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=target_names, yticklabels=target_names, ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.markdown("#### 🌟 Feature Importance")
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                idx = np.argsort(imp)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.barh(range(len(idx)), imp[idx], color="#ff7f0e", edgecolor="k")
                ax.set_yticks(range(len(idx)))
                ax.set_yticklabels([feature_names[i] for i in idx])
                ax.set_xlabel("Importance")
                ax.grid(True, alpha=0.3, axis="x")
                st.pyplot(fig)
                plt.close(fig)
            elif hasattr(model, "coef_"):
                coef = np.abs(model.coef_)
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
                idx = np.argsort(coef)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.barh(range(len(idx)), coef[idx], color="#ff7f0e", edgecolor="k")
                ax.set_yticks(range(len(idx)))
                ax.set_yticklabels([feature_names[i] for i in idx])
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("ℹ️ Model ini tidak menyediakan feature importance.")


# ============================================================
# TAB 2: EVALUATION METRICS
# ============================================================
with tab2:
    if not st.session_state.is_trained:
        st.warning("⚠️ Klik **Train Model** dulu di sidebar.")
        st.stop()

    st.subheader("📈 Evaluation Metrics")

    metrics = st.session_state.metrics
    task = st.session_state.task
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    cv_score = st.session_state.cv_score
    target_names = st.session_state.target_names

    if task == "Regresi":
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📏 MAE", f"{metrics['MAE']:.4f}")
        c2.metric("📐 MSE", f"{metrics['MSE']:.4f}")
        c3.metric("📊 RMSE", f"{metrics['RMSE']:.4f}")
        c4.metric("🎯 R²", f"{metrics['R2']:.4f}")
        c5.metric("📉 MAPE", f"{metrics['MAPE']*100:.2f}%")

        st.divider()
        r2 = metrics["R2"]
        if r2 >= 0.9:
            st.success(f"🌟 **R² = {r2:.4f}** — Model SANGAT BAIK ({r2*100:.1f}% variansi)")
        elif r2 >= 0.7:
            st.info(f"✅ **R² = {r2:.4f}** — Model BAIK ({r2*100:.1f}% variansi)")
        elif r2 >= 0.5:
            st.warning(f"⚠️ **R² = {r2:.4f}** — Model CUKUP ({r2*100:.1f}% variansi)")
        else:
            st.error(f"❌ **R² = {r2:.4f}** — Model KURANG BAIK")

    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Accuracy", f"{metrics['Accuracy']*100:.2f}%")
        c2.metric("🎪 Precision", f"{metrics['Precision']*100:.2f}%")
        c3.metric("🔍 Recall", f"{metrics['Recall']*100:.2f}%")
        c4.metric("⚖️ F1-Score", f"{metrics['F1-Score']*100:.2f}%")

        st.divider()
        acc = metrics["Accuracy"]
        if acc >= 0.95:
            st.success(f"🌟 **Akurasi = {acc*100:.2f}%** — SANGAT BAIK")
        elif acc >= 0.85:
            st.info(f"✅ **Akurasi = {acc*100:.2f}%** — BAIK")
        elif acc >= 0.70:
            st.warning(f"⚠️ **Akurasi = {acc*100:.2f}%** — CUKUP")
        else:
            st.error(f"❌ **Akurasi = {acc*100:.2f}%** — KURANG BAIK")

        st.markdown("### 📋 Classification Report")
        report = classification_report(y_test, y_pred, target_names=target_names,
                                       output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.4f}"), use_container_width=True)

    st.divider()
    st.markdown("### 🔄 Cross-Validation (5-Fold)")
    cv_mean, cv_std = cv_score
    c1, c2 = st.columns(2)
    c1.metric("CV Score (Mean)", f"{cv_mean:.4f}")
    c2.metric("CV Score (Std)", f"±{cv_std:.4f}")


# ============================================================
# TAB 3: DATABASE & RETRAIN
# ============================================================
with tab3:
    st.subheader("🗄️ MongoDB CRUD & Retrain Model")

    # Cek koneksi
    client = get_mongo_client()

    if client is None:
        st.error(
            "❌ **MongoDB tidak terhubung!**\n\n"
            "Pastikan:\n"
            "1. `pymongo` sudah terinstall: `pip install pymongo`\n"
            "2. File `.streamlit/secrets.toml` ada dengan `MONGODB_URI`\n"
            "3. IP sudah di-whitelist di MongoDB Atlas"
        )
        st.stop()

    db = client["test_db"]
    collection = db["iris_db"]

    # --- Statistik ---
    st.markdown("### 📊 Statistik Collection")
    c1, c2, c3 = st.columns(3)
    total = collection.count_documents({})
    c1.metric("Total Dokumen", total)
    c2.metric("Database", "test_db")
    c3.metric("Collection", "iris_db")

    st.divider()

    # --- Form Insert ---
    st.markdown("### ➕ Tambah Data Iris")
    with st.form("form_iris"):
        c1, c2 = st.columns(2)
        with c1:
            sl = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1, 0.1)
            sw = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5, 0.1)
        with c2:
            pl = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4, 0.1)
            pw = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2, 0.1)
        species = st.selectbox("Species", ["setosa", "versicolor", "virginica"])

        submitted = st.form_submit_button("💾 Simpan ke MongoDB", type="primary")

        if submitted:
            collection.insert_one({
                "sepal_length": float(sl),
                "sepal_width": float(sw),
                "petal_length": float(pl),
                "petal_width": float(pw),
                "species": species,
                "timestamp": datetime.now(),
            })
            st.success("✅ Data berhasil disimpan!")
            st.rerun()

    st.divider()

    # --- Tampilkan Data ---
    st.markdown("### 📋 Data Iris di MongoDB")
    data = list(collection.find().sort("timestamp", -1))

    if not data:
        st.info("📭 Belum ada data. Silakan tambah di atas.")
    else:
        df_mongo = pd.DataFrame(data)
        df_display = df_mongo.copy()
        df_display["_id"] = df_display["_id"].astype(str)
        st.dataframe(df_display, use_container_width=True)

        # --- Hapus ---
        st.markdown("### 🗑️ Hapus Data")
        c1, c2 = st.columns([3, 1])
        with c1:
            opsi = st.selectbox(
                "Pilih ID:",
                df_mongo["_id"].astype(str).tolist(),
            )
        with c2:
            if st.button("🗑️ Hapus", type="secondary"):
                collection.delete_one({"_id": ObjectId(opsi)})
                st.success(f"✅ Dihapus: `{opsi[:12]}...`")
                st.rerun()

    st.divider()

    # --- Retrain ---
    st.markdown("### 🧠 Retrain Model dengan Data MongoDB")
    st.caption(
        "Model akan digabung: **data sklearn (150)** + **data MongoDB**. "
        "Cocok untuk melihat efek data tambahan."
    )

    if st.button("🚀 Retrain Sekarang", type="primary"):
        try:
            # Data sklearn
            iris = datasets.load_iris()
            X_orig = iris.data
            y_orig = iris.target

            # Data MongoDB
            df_new = pd.DataFrame(list(collection.find({}, {"_id": 0})))

            if df_new.empty:
                st.warning("⚠️ Tidak ada data MongoDB. Training dengan sklearn saja.")
                X_comb, y_comb = X_orig, y_orig
            else:
                species_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
                df_new["target"] = df_new["species"].map(species_map)

                if df_new["target"].isna().any():
                    st.error("❌ Ada species tidak dikenal!")
                    st.stop()

                X_new = df_new[["sepal_length", "sepal_width",
                                "petal_length", "petal_width"]].values
                y_new = df_new["target"].values

                X_comb = np.vstack([X_orig, X_new])
                y_comb = np.hstack([y_orig, y_new])

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_comb, y_comb, test_size=0.2, random_state=42
            )

            retrained = RandomForestClassifier(n_estimators=100, random_state=42)
            retrained.fit(X_tr, y_tr)
            y_pred_rt = retrained.predict(X_te)
            acc_rt = accuracy_score(y_te, y_pred_rt)

            st.success(f"✅ Model di-retrain! Akurasi: **{acc_rt*100:.2f}%**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Data sklearn", len(X_orig))
            c2.metric("Data MongoDB", len(df_new) if not df_new.empty else 0)
            c3.metric("Total Training", len(X_comb))

            st.text("Classification Report:")
            st.text(classification_report(y_te, y_pred_rt,
                                          target_names=iris.target_names,
                                          zero_division=0))

            # Feature importance
            imp_df = pd.DataFrame({
                "Fitur": iris.feature_names,
                "Importance": retrained.feature_importances_,
            }).sort_values("Importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("🤖 Dibuat dengan ❤️ menggunakan Streamlit + scikit-learn + MongoDB")