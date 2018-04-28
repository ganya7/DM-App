import flask
import numpy as np
from flask import Flask, render_template, request
from random_forest_classification import randforest

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        X_test_input = []
        age = request.form['age']
        X_test_input.append(age)
        salary = request.form['salary']
        X_test_input.append(salary)
        y_prediction = randforest(X_test_input)
        label_pred = str(y_prediction)
        return render_template('index.html', label=label_pred)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
