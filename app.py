from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from prediction import Predict
import time

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/predict', methods=['POST'])
def uploadCsv():
    '''
    This fucntion does what it's name suggests, recieves the csv from frontend or 
    in future it will take csv/json from pipeline and then read it, process it and 
    make prediction over it. 
    '''
    uploadedfile = request.files['file']
    print('Recieved Dataeset for prediction: ', uploadedfile)

    if uploadedfile.filename != '':
        uploadedfile.save(uploadedfile.filename)

    predictor = Predict()
    payload = predictor.predictLogistic(str(uploadedfile.filename))
    return jsonify(payload)


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/genCSV', methods=['POST'])
def generateCsv():
    recv_timestamp = time.time()
    csv_file = request.files['file']
    print('Recieved CSV for Normalizing: ', csv_file)
    newCSV = Predict()
    zip_path = newCSV.generateCSV(csv_file, recv_timestamp)

    return send_file(zip_path, mimetype='zip', attachment_filename='out.zip', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
