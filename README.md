# 📄 README.md — Versi Singkat (1/3 Panjang)

```markdown
# 🤖 Streamlit Pertama — Belajar Web ML + MongoDB

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aisuryafsmuksw.streamlit.app)

Aplikasi Machine Learning interaktif dengan **Streamlit** + **MongoDB Atlas** untuk CRUD data dan retraining model.

🚀 **Live Demo**: https://aisuryafsmuksw.streamlit.app

---

## ✨ Fitur

| Tab | Isi |
|-----|-----|
| **1. Training & Visualisasi** | Actual vs Predicted, Confusion Matrix, Feature Importance |
| **2. Evaluation Metrics** | MAE/MSE/RMSE/R²/MAPE atau Accuracy/Precision/Recall/F1 |
| **3. Database & Retrain** | CRUD data Iris + retrain dengan data MongoDB |

Dataset sklearn: Diabetes, California Housing, Iris, Wine, Breast Cancer, Digits.

---

## 🗄️ Setup MongoDB

1. Register di [MongoDB Atlas](https://cloud.mongodb.com)
2. **Add Organization** → **Add Project** → **Add Cluster** (M0 Free)
3. Buat **Database User** → catat username & password
4. **Network Access** → whitelist IP `0.0.0.0/0` (testing)
5. **Connect → Drivers → Python** → copy connection string

Buat `.streamlit/secrets.toml`:
```toml
MONGODB_URI = "mongodb://username:password@cluster.mongodb.net/..." (copy paste dari mongodb)
```

> ⚠️ File ini tidak di-commit ke GitHub. Isi juga di **Streamlit Cloud → Settings → Secrets**.

---

## 📦 Instalasi

```bash
git clone https://github.com/suryafsm/streamlit-pertama.git
cd streamlit-pertama
python -m venv venv
venv\Scripts\activate.bat        # Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎮 Cara Pakai

**Tab 1 & 2 (ML)**: pilih Regresi/Klasifikasi + dataset + model → klik **🚀 Train Model**.

**Tab 3 (MongoDB)**: insert data → delete → klik **🚀 Retrain** untuk latih ulang model.

### Alur Retraining

```
Data sklearn (150) + Data MongoDB (N)
        ↓
   Gabung → Train Random Forest
        ↓
   Akurasi + Classification Report
```

---

## 🔐 Keamanan

Jangan commit file ini (sudah ada di `.gitignore`):

```gitignore
.streamlit/
.vscode/
venv/
__pycache__/
```

---

## 🛠️ Teknologi

Python 3.12 · Streamlit · scikit-learn · MongoDB Atlas · matplotlib · seaborn · pandas · Git

---

## 👤 Author

**Surya FSM** [Live App](https://aisuryafsmuksw.streamlit.app)

## 📄 License

[MIT License](LICENSE)

---

🤖 **Dibuat dengan ❤️ menggunakan Streamlit + scikit-learn + MongoDB**
```

