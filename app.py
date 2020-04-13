import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, request, render_template

#Build a model for the salary estimate

#Load training data
#data is loaded from files for simplicity
data = pd.DataFrame({
    'Experience': [1,2,3,4,5,6,7,8],
    'Test_Score': [86,89,75,46,80,39,100,69],
    'Interview_Score': [97,75,56,78,96,99,53,78],
    'Salary': [40000, 50000, 60000, 70000, 80000,90000, 100000, 150000,]
})

#Train a model
model = LinearRegression()
model.fit(data[['Experience', 'Test_Score', 'Interview_Score']], data['Salary'])
prediction = model.predict([[5, 60, 60]])

print(prediction)

#add instructions on how to respond to internet queries

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Load information from user, apply the model, return the result
    """
    inputs = np.array([[int(x) for x in request.form.values()]])
    prediction = model.predict(inputs)
    return render_template('index.html', prediction_text=int((prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)