from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os
import joblib

MODEL_PATH = 'model.pkl'
TRAIN_CSV = 'train.csv'

app = Flask(__name__)

def train_and_save_model():
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv(TRAIN_CSV)

    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
    df['Credit_History'] = df['Credit_History'].fillna(0)
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna('No')
    df['Dependents'] = df['Dependents'].fillna('0')
    df['Self_Employed'] = df['Self_Employed'].fillna('No')

    df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
    df['Married'] = df['Married'].map({'Yes':1,'No':0})
    df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0})
    df['Property_Area'] = df['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2})
    df['Dependents'] = df['Dependents'].replace('3+','3').astype(int)
    df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})

    X = df[['Gender','Married','Dependents','Education','Self_Employed',
            'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
            'Credit_History','Property_Area']]
    y = df['Loan_Status']

    X['TotalIncome'] = X['ApplicantIncome'] + X['CoapplicantIncome']
    X['LoanAmountLog'] = np.log1p(X['LoanAmount'])
    X = X.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount'], axis=1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    num_cols = ['TotalIncome','Loan_Amount_Term','LoanAmountLog']
    X[num_cols] = scaler.fit_transform(X[num_cols])

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return joblib.load(MODEL_PATH)


model_bundle = None

# 🔥 FIX: use before_request instead of before_first_request
@app.before_request
def startup():
    global model_bundle
    if model_bundle is None:
        model_bundle = load_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    Gender = int(data['Gender'])
    Married = int(data['Married'])
    Dependents = int(data['Dependents'])
    Education = int(data['Education'])
    Self_Employed = int(data['Self_Employed'])
    ApplicantIncome = float(data['ApplicantIncome'])
    CoapplicantIncome = float(data['CoapplicantIncome'])
    LoanAmount = float(data['LoanAmount'])
    Loan_Amount_Term = float(data['Loan_Amount_Term'])
    Credit_History = int(data['Credit_History'])
    Property_Area = int(data['Property_Area'])

    TotalIncome = ApplicantIncome + CoapplicantIncome
    LoanAmountLog = np.log1p(LoanAmount)

    X = pd.DataFrame([{
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area,
        'TotalIncome': TotalIncome,
        'LoanAmountLog': LoanAmountLog
    }])

    num_cols = ['TotalIncome','Loan_Amount_Term','LoanAmountLog']
    X[num_cols] = model_bundle['scaler'].transform(X[num_cols])

    prob = model_bundle['model'].predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    return jsonify({
        'probability': float(prob),
        'prediction': "Approved" if pred == 1 else "Not Approved"
    })


if __name__ == '__main__':
    app.run(debug=True)
