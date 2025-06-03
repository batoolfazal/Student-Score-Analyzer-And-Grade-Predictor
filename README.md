# Student-Score-Analyzer-And-Grade-Predictor
The Student Score Analyzer and Grade Predictor is an interactive web application designed to analyze student performance data and predict future academic outcomes. This tool combines data visualization, machine learning, and user-friendly interfaces to help educators and administrators gain insights from student performance metrics.
This project is a **Streamlit-based web app** that allows users to upload student performance datasets, clean and analyze the data, and apply machine learning models to **predict student scores**, **classify pass/fail**, and **assign letter grades**.

## 🚀 Features
- 📤 Upload any CSV file containing student performance data  
- 🧹 Automated data cleaning (NaN removal, duplicates handling)  
- 📊 Real-time visualizations of student performance by gender  
- 📚 Feature engineering (average score, study efficiency, education factor)  
- 🔤 Label encoding of categorical variables  
- 📈 Linear Regression for score prediction  
- 🧠 Logistic Regression for pass/fail classification  
- 🏅 Grade assignment based on predicted scores  
- 🧑‍🎓 Predict scores and grades for a new student using custom inputs  
- 💡 Actionable feedback based on predicted performance

## 🧪 Technologies Used
- **Python 3.12**
- **Streamlit** – for building the interactive web app  
- **Pandas & NumPy** – for data handling and preprocessing  
- **Matplotlib** – for data visualization  
- **Scikit-learn** – for regression and classification models

## 📂 File Structure
├── ai102_lab_project_140_540.py # Main Streamlit application file
└── README.md 

## ⚙️ How to Run the App

### 1. Clone the Repository
git clone https://github.com/batoolfazal/Student-Score-Analyzer-And-Grade-Predictor.git
cd student-score-analyzer

2. Install Dependencies
It's recommended to use a virtual environment:
pip install -r requirements.txt

4. Launch the App
streamlit run ai102_lab_project_140_540.py
📄 Input Data Requirements
Your CSV file should contain at least these three columns:
math score
reading score
writing score

Optional but supported columns:
gender
study time
parental level of education

📘 Example Use Case
Upload a dataset of student performance
Explore cleaned and visualized data
Choose a subject (e.g., "math score") for prediction
View model RMSE and accuracy
Enter new student data to get score, grade, and pass/fail prediction

🤖 Machine Learning Details
Regression Model: LinearRegression for predicting scores
Classification Model: LogisticRegression to predict pass/fail
Uses train_test_split with random_state=42 for reproducibility
Evaluation metrics: RMSE (regression), Accuracy (classification)

📬 Contact
Made with ❤️ using Python and Streamlit.
For questions or suggestions:

Name: Batool Binte Fazal / Rida Fakhir
GitHub: batoolfazal / ridafaakhar1
Email: batoolbintefazal2006@gmail.com 
