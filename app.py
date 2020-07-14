# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


# Load the Random Forest reg model
filename = 'area_price_lr_model.pkl'
reg = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        area = int(request.form['areas'])
    
        data = np.array([[area]])
        my_prediction = reg.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
















































































