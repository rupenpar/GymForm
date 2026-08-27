import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join("data","processed_data","basic_gym_data.csv")
MODEL_DIR = "models"
df = pd.read_csv(DATA_PATH)

TARGET_COL = df.columns[-1]

x = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

label_encode = None
if y.dtype == 'object' or isinstance(y.iloc[0], str):
    label_encode = LabelEncoder()
    y = label_encode.fit_transform(y)

    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    joblib.dump(label_encode, encoder_path)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2, random_state=40, stratify= y
)

rf_model = RandomForestClassifier(
    n_estimators= 200,
    random_state=30
)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy*100:.2f}")

xgb = XGBClassifier(
    n_estimators = 300,
    random_state=30,
    eval_metric='logloss'
)
xgb.fit(x_train,y_train)
xgb_pred = xgb.predict(x_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy*100:.2f}")

if xgb_accuracy >= rf_accuracy:
    best_model = xgb
    best_name = "XGBoost"
    best_acc = xgb_accuracy
    best_preds = xgb_pred
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_acc = rf_accuracy
    best_preds = rf_pred

model_path = os.path.join(MODEL_DIR, "exercise_classifier.pkl")
joblib.dump(best_model, model_path)





