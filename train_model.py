import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('Accidents.csv')
df.columns = df.columns.str.strip()
print("Columns in CSV:", df.columns.tolist())


df['age_of_driver'] = 30
df['vehicle_type'] = 2
df['age_of_vehicle'] = 5
df['engine_cc'] = 1400
df['gender'] = 1


df.rename(columns={
    'Did_Police_Officer_Attend_Scene_of_Accident': 'Did_Police_Officer_Attend',
    'Day_of_Week': 'day',
    'Weather_Conditions': 'weather',
    'Road_Surface_Conditions': 'roadsc',
    'Light_Conditions': 'light',
    'Speed_limit': 'speedl',
    'Accident_Severity': 'severity'
}, inplace=True)

feature_cols = ['Did_Police_Officer_Attend', 'age_of_driver', 'vehicle_type',
                'age_of_vehicle', 'engine_cc', 'day', 'weather',
                'roadsc', 'light', 'gender', 'speedl']
target_col = 'severity'

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'litemodel.sav')
print("✅ Model saved to litemodel.sav")
