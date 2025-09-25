# 💪 Personal Fitness Tracker

A **Streamlit web app** that predicts **calories burned** during exercise based on personal health parameters such as Age, Gender, BMI, Duration, Heart Rate, and Body Temperature.  
The app uses **Machine Learning models (Random Forest & Linear Regression)** trained on fitness datasets to provide calorie predictions and insights.  

---

## 🚀 Features
- 🔢 **Input Parameters**: Age, BMI, Duration, Heart Rate, Body Temp, Gender  
- 🔮 **Prediction**: Estimated calories burned  
- 📊 **Model Performance**: RMSE & R² score displayed  
- ⚡ **Feature Importance**: Shows which features impact calorie prediction (for Random Forest)  
- 📈 **Visualizations**:  
  - Calories distribution  
  - Calories vs BMI scatterplot 
  - Similar results from dataset  
- 🧑‍💻 **Choose Model**: Random Forest or Linear Regression  

---

## 📂 Project Structure
fitness-tracker-app/

│── app.py 

│── calories.csv 

│── exercise.csv 

│── requirements.txt 

│── README.md 


---

## ⚙️ Installation & Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fitness-tracker-app.git
   cd fitness-tracker-app

2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   
3.Run the app:
```bash
streamlit run app.py


