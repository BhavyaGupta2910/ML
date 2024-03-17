from flask import Flask, render_template,request
import pickle
import numpy as np

model=pickle.load(open('modelHK2.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])

def predict():
    home_owner=(request.form.get('home_owner'))
    income=(request.form.get('income'))
    current_address_year=int(request.form.get('current_address_year'))
    has_debt=(request.form.get('has_debt'))
    amount_requested=(request.form.get('amount_requested'))
    risk_score=(request.form.get('risk_score'))
    inquiries_last_month=(request.form.get('inquiries_last_month'))
    EMPLOYED=(request.form.get('EMPLOYED'))
    PA=(request.form.get('PA'))
    RISK=(request.form.get('RISK'))
    QUALITY=(request.form.get('QUALITY'))
    ADULT=(request.form.get('ADULT'))
    SENIOR=(request.form.get('SENIOR'))
    monthly=(request.form.get('monthly'))
    weekly=(request.form.get('weekly'))

    result =  model.predict(np.array([home_owner,income,current_address_year,has_debt,amount_requested,risk_score,inquiries_last_month,EMPLOYED,PA,RISK,QUALITY,ADULT,SENIOR,monthly,weekly
                ]).reshape(1, 15))
    
    output='{0:.{1}f}'.format(result[0], 2)
    print(output)
    if output[0]==1:
        output = "Loan Can't be Approved"
    else :
        output = "Congrats Your Loan Approved"
    
    return str(output)
    
    #return render_template('index.html', prediction_text='Prediction: {}'.format(output))


if __name__ =='__main__':
    app.run(debug=True)
