from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('rf_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index3.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Area=float(request.form['Area'])
    ED=float(request.form['ED'])
    Energy=float(request.form['Energy'])
    SS=int(request.form['SS'])
    Residue_Length=int(request.form['ResidueLength'])
    Pair_Number=int(request.form['PairNumber'])
    prediction=model.predict([[Area,ED,Energy,SS,Residue_Length,Pair_Number]])
    output=round(prediction[0],3)
    return render_template('index3.html',prediction_text="Final predicted RMSD value is {}".format(output))
@app.route("/again")
def again():
    return render_template('index3.html',prediction_text="Final predicted RMSD value is....")
    

if __name__=="__main__":
    app.run(debug=True)

