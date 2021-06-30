from types import new_class
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from prediction import Predict
from flask import send_file

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
    csvFile = request.files['file']
    print('Recieved Dataeset for prediction: ', csvFile)
    newCSV = Predict()
    payload = newCSV.generateCSV(csvFile)
    
    return send_file('out.zip', mimetype='zip', attachment_filename='out.zip', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
