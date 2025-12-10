# 📘 Content Monetization Modeler  
### YouTube Ad Revenue Prediction using Machine Learning + Streamlit

---

## 📌 Project Overview  
This project builds a complete ML pipeline to predict **YouTube Ad Revenue** using video metadata, engagement metrics, and viewer information.

It includes:
- Data preprocessing  
- Feature engineering  
- Linear & ensemble model training  
- SHAP explainability  
- A deployable Streamlit web application  

---

## ✨ Key Features  
✔ Full Machine Learning Pipeline  
✔ Linear Regression model with **R² = 0.95**  
✔ RandomForest, Ridge & Lasso comparison  
✔ SHAP explainability  
✔ Streamlit Web App  
✔ Clean modular folder structure  
✔ Git-friendly (large models excluded)

---

## 📂 Project Structure  
Content_Monetization_Modeler/
│
├── data/
├── models/ (excluded from GitHub)
├── outputs/
├── scripts/
└── streamlit_app/


---

## 🔧 Tech Stack  
- Python  
- Scikit-Learn  
- Pandas  
- Seaborn, Matplotlib  
- SHAP  
- Streamlit  

---

## 📊 Model Performance  
| Model            | R²     | RMSE   |
|------------------|--------|--------|
| Linear Regression | 0.9504 | 13.76 |
| Ridge             | 0.9504 | 13.76 |
| RandomForest      | 0.9470 | 14.22 |
| Lasso             | 0.9331 | 15.99 |

---

## 🧠 SHAP Explainability  
Used for:
- Global feature importance  
- Local prediction explanation  
- Waterfall & summary plots  

**Top impactful features:**  
📌 Watch Time  
📌 Views  
📌 Engagement Rate  
📌 Subscribers  

---

## 🖥️ Running the Streamlit App  

### 🚀 Setup & Installation  

#### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/dsvarsha/Content-Monetization-Modeler.git
cd Content-Monetization-Modeler

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run streamlit_app/app.py

python scripts/step3_preprocessing.py
python scripts/step4_model.py
python scripts/step5_shap.py

🙋‍♀️ About the Author

Varsha SureshKumar
ECE | IoT | Data Analytics | ML | UI/UX
Passionate about building things that think ✨
