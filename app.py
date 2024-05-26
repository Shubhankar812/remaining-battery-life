import pickle
from flask import Flask, request, jsonify
import numpy as np
import sklearn.metrics._dist_metrics as dist_metrics

app = Flask(__name__)


dt = pickle.load(open('knn_model.pkl', 'rb'))

@app.route("/")
def Home():
    return 'Hello World!!'


@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    f1 = float(data['f1'])
    f2 = float(data['f2'])
    f3 = float(data['f3'])


    final_input = np.array([f1, f2, f3]).reshape(1, -1)

    output = dt.predict(final_input)

    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)
