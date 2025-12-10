📘 Content Monetization Modeler
YouTube Ad Revenue Prediction using Machine Learning + Streamlit
<p align="center"> <img src="https://img.shields.io/badge/Project-ML%20Pipeline-blueviolet?style=for-the-badge" /> <img src="https://img.shields.io/badge/Framework-ScikitLearn-orange?style=for-the-badge" /> <img src="https://img.shields.io/badge/App-Streamlit-red?style=for-the-badge" /> <img src="https://img.shields.io/badge/Explainability-SHAP-green?style=for-the-badge" /> </p>
📌 Project Overview

This project builds a complete ML pipeline to predict YouTube Ad Revenue using video metadata, engagement metrics, and viewer information.
It includes preprocessing, feature engineering, model building, explainability using SHAP, and a deployable Streamlit application.

✨ Key Features

✔ Full Machine Learning Pipeline
✔ Linear Regression model with R² = 0.95
✔ RandomForest, Ridge & Lasso comparison
✔ SHAP explainability
✔ Streamlit Web App
✔ Clean modular folder structure
✔ Git-friendly (large models excluded)

📂 Project Structure
Content_Monetization_Modeler/
│
├── data/
├── models/ (excluded from GitHub)
├── outputs/
├── scripts/
└── streamlit_app/

🔧 Tech Stack

Python

Scikit-Learn

Pandas

Seaborn, Matplotlib

SHAP

Streamlit

📊 Model Performance
Model	R²	RMSE
Linear Regression	0.9504	13.76
Ridge	0.9504	13.76
RandomForest	0.9470	14.22
Lasso	0.9331	15.99
🧠 Explainability (SHAP)

The project uses SHAP for:

Global feature importance

Local prediction explanation

Waterfall & summary plots

Top factors:

Watch Time

Views

Engagement Rate

Subscribers

🖥️ Running the Streamlit App

Install requirements:

pip install -r requirements.txt


Run:

streamlit run streamlit_app/app.py

🚀 Training the Model
python scripts/step3_preprocessing.py
python scripts/step4_model.py
python scripts/step5_shap.py

☁️ Model Artifacts

Model files are excluded due to size.
You can download the model here:

➡️ Add Google Drive link here

🙋‍♀️ About the Author

Varsha SureshKumar
ECE | IoT | Data Analytics | ML | UI/UX
Passionate about building things that think.