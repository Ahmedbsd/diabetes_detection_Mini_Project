import pandas as pd
import joblib

# Load saved model and preprocessor
preprocessor, model = joblib.load('diabetes_model.pkl')

# Example new user
new_user = pd.DataFrame([{
    'age': 45,
    'alcohol_consumption_per_week': 3,
    'physical_activity_minutes_per_week': 120,
    'diet_score': 7,
    'sleep_hours_per_day': 7,
    'screen_time_hours_per_day': 5,
    'bmi': 28.5,
    'waist_to_hip_ratio': 0.95,
    'systolic_bp': 130,
    'diastolic_bp': 85,
    'heart_rate': 78,
    'cholesterol_total': 210,
    'hdl_cholesterol': 45,
    'ldl_cholesterol': 135,
    'triglycerides': 160,
    'glucose_fasting': 110,
    'glucose_postprandial': 160,
    'insulin_level': 18,
    'hba1c': 6.1,
    'gender': 'Male',
    'ethnicity': 'White',
    'employment_status': 'Employed',
    'smoking_status': 'Former',
    'family_history_diabetes': 1,
    'hypertension_history': 1,
    'cardiovascular_history': 0,
    'income_level': 'Middle',
    'education_level': 'Graduate'
}])

high_risk_user = pd.DataFrame([{
    'age': 55,
    'alcohol_consumption_per_week': 10,
    'physical_activity_minutes_per_week': 30,  # very low
    'diet_score': 3,  # poor diet
    'sleep_hours_per_day': 5,
    'screen_time_hours_per_day': 8,
    'bmi': 33,  # obese
    'waist_to_hip_ratio': 1.05,  # high
    'systolic_bp': 145,
    'diastolic_bp': 90,
    'heart_rate': 88,
    'cholesterol_total': 250,
    'hdl_cholesterol': 35,
    'ldl_cholesterol': 160,
    'triglycerides': 220,
    'glucose_fasting': 140,  # high
    'glucose_postprandial': 220,  # high
    'insulin_level': 35,
    'hba1c': 7.5,  # above normal
    'gender': 'Male',
    'ethnicity': 'Hispanic',
    'employment_status': 'Employed',
    'smoking_status': 'Current',
    'family_history_diabetes': 1,  # family history
    'hypertension_history': 1,
    'cardiovascular_history': 1,
    'income_level': 'Low',
    'education_level': 'Highschool'
}])

# Preprocess new user
new_user_processed = preprocessor.transform(new_user)
user_processed = preprocessor.transform(high_risk_user)

# Predict
prediction = model.predict(new_user_processed)
probability = model.predict_proba(new_user_processed)[:,1]

print("Predicted Diabetes (0=No, 1=Yes):", prediction[0])
print("Probability of Diabetes:", round(probability[0], 4))

prediction = model.predict(user_processed)
probability = model.predict_proba(user_processed)[:,1]

print("Predicted Diabetes (0=No, 1=Yes):", prediction[0])
print("Probability of Diabetes:", round(probability[0], 4))
