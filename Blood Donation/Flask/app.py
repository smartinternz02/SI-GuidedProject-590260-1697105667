import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    int_features= [int(x) for x in request.form.values()]
    features=[np.array(int_features)]
    prediction=model.predict(features)
    
    output = prediction[0]
    return render_template('findTheDonor.html', prediction_text='Chance of donor to donate blood is {}'.format(output))
   
    

if __name__ == "__main__":
    app.run(debug=True)
