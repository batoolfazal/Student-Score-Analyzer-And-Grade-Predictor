import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

# === App Config ===
st.set_page_config(page_title="Student Score Analyzer & Grade Predictor", layout="wide")
st.title("📊 Student Score Analyzer & Grade Predictor")

# === Grade Function ===
def get_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

# === File Upload ===
uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # === Data Cleaning ===
    st.header("🧹 Data Cleaning")
    st.write("Initial shape:", df.shape)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    st.write("Shape after cleaning:", df.shape)

    # === Feature Engineering ===
    df['average score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

    if 'study time' in df.columns:
        df['study_efficiency'] = df['average score'] / (df['study time'] + 1)
        df['study_engagement'] = df['study time'] * (df['average score'] / 100)

    if 'parental level of education' in df.columns and pd.api.types.is_numeric_dtype(df['parental level of education']):
        df['education_factor'] = df['average score'] / (df['parental level of education'] + 1)

    # === Label Encoding ===
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # === Cleaned Data Preview ===
    st.subheader("👁 Cleaned Data Preview")
    st.dataframe(df.head())

    # === Visualizations ===
    st.header("📊 Data Visualizations")

    score_cols = ['math score', 'reading score', 'writing score']
    for score_col in score_cols:
        if 'gender' in df.columns:
            st.subheader(f"📈 Average {score_col.title()} by Gender")
            fig, ax = plt.subplots()
            df.groupby('gender')[score_col].mean().plot(kind='bar', ax=ax, color='skyblue')
            ax.set_xlabel("Gender")
            ax.set_ylabel(score_col.title())
            st.pyplot(fig)

    # === Correlation Matrix ===
    st.subheader("🔗 Correlation Matrix")
    st.dataframe(df.corr(numeric_only=True).style.background_gradient(cmap="YlGnBu"))

    # === Machine Learning Models ===
    st.header("🤖 Machine Learning Models")

    subject = st.selectbox("📘 Select subject to predict", options=["math score", "reading score", "writing score"])
    target_score = df[subject]

    # Define features
    exclude_cols = ["math score", "reading score", "writing score", "average score", "study_efficiency", "pass_fail", "grade"]
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = target_score

    # === 🔧 Regression Model ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    pred = reg_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    st.success(f"RMSE for {subject}: {round(rmse, 2)}")

    # === Classification (Pass/Fail) ===
    df['pass_fail'] = (df[subject] >= 50).astype(int)
    y_class = df['pass_fail']
    X_class = X

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    clf_model = LogisticRegression(max_iter=1000, random_state=42)
    clf_model.fit(X_train_c, y_train_c)
    pred_class = clf_model.predict(X_test_c)
    acc = accuracy_score(y_test_c, pred_class)
    st.success(f"🎯 Classification Accuracy: {round(acc * 100, 2)}%")

    # Assign grades
    df['grade'] = df[subject].apply(get_grade)

    # === Predict for a New Student ===
    st.header("🧑‍🎓 Predict for a New Student")

    user_input = {}
    for col in X.columns:
        if col in label_encoders:
            options = list(label_encoders[col].classes_)
            selected = st.selectbox(f"🔤 {col}", options=options)
            user_input[col] = label_encoders[col].transform([selected])[0]
        else:
            user_input[col] = st.number_input(f"🔢 {col}", value=0.0)

    if st.button("🚀 Predict"):
        input_df = pd.DataFrame([user_input])
        score_prediction = reg_model.predict(input_df)[0]
        pass_prediction = clf_model.predict(input_df)[0]
        grade = get_grade(score_prediction)

        st.subheader("📣 Prediction Results")
        st.write(f"📘 *Predicted {subject} Score*: {round(score_prediction, 2)}")
        st.write(f"🏅 *Predicted Grade*: {grade}")
        st.write("✅ *Pass/Fail*:", "Pass" if pass_prediction == 1 else "Fail")

        # 💡 Feedback
        if score_prediction < 50:
            st.info("⚠ Suggestion: Focus on time management and consider tutoring.")
        elif score_prediction < 75:
            st.info("👍 Good effort! Keep practicing.")
        else:
            st.success("🌟 Excellent! You're doing great!")

else:
    st.warning("⚠ Please upload a valid CSV file to begin.")