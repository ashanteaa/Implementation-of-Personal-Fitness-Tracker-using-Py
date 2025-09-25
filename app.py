import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

st.set_page_config(page_title="Personal Fitness Tracker", page_icon="💪", layout="wide")

# Title
st.title("💪 Personal Fitness Tracker")
st.markdown("Predict your **calories burned** based on personal health parameters using Machine Learning.")

# Sidebar Inputs
st.sidebar.header("⚙️ User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min)", 0, 60, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    return pd.DataFrame(data_model, index=[0])

df = user_input_features()

# Load Data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    data = exercise.merge(calories, on="User_ID")
    data.drop(columns="User_ID", inplace=True)
    data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)
    return data

exercise_df = load_data()

# Preprocess
train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

X_train, y_train = train_data.drop("Calories", axis=1), train_data["Calories"]
X_test, y_test = test_data.drop("Calories", axis=1), test_data["Calories"]

# Model Selection
st.sidebar.subheader("🔍 Select Model")
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Linear Regression"])

if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=1)
else:
    model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
df_aligned = df.reindex(columns=X_train.columns, fill_value=0)
prediction = model.predict(df_aligned)[0]

st.subheader("📊 Your Input Parameters")
st.write(df)

st.subheader("🔥 Predicted Calories Burned")
st.metric(label="Estimated Calories", value=f"{round(prediction, 2)} kcal")

# Model Performance
st.subheader("📈 Model Performance")
y_pred = model.predict(X_test)
st.write("**RMSE:**", round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
st.write("**R² Score:**", round(metrics.r2_score(y_test, y_pred), 2))

# Feature Importance (only for RF)
if model_choice == "Random Forest":
    st.subheader("⚡ Feature Importance")
    feat_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    fig, ax = plt.subplots()
    sns.barplot(x=feat_importance.values, y=feat_importance.index, ax=ax, palette="viridis")
    ax.set_title("Feature Importance in Calorie Prediction")
    st.pyplot(fig)

# Similar Data
st.subheader("🔎 Similar Results")
calorie_range = [prediction - 10, prediction + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

# Comparisons
st.subheader("📊 General Comparisons")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(exercise_df["Calories"], bins=30, kde=True, ax=axes[0], color="blue")
axes[0].axvline(prediction, color="red", linestyle="--")
axes[0].set_title("Calories Distribution")

sns.scatterplot(x=exercise_df["BMI"], y=exercise_df["Calories"], hue=exercise_df["Gender"], ax=axes[1])
axes[1].scatter(df["BMI"], prediction, color="red", s=100, label="You")
axes[1].set_title("Calories vs BMI")
axes[1].legend()

st.pyplot(fig)
