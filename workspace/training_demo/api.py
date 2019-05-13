import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/obj', methods=['GET'])
def api_id():
    import json
    import os
    imgstring = request.data 
    import base64
    imgdata = base64.b64decode(imgstring)
    filename = 'test_image.jpg' 
    with open(filename, 'wb') as f:
        f.write(imgdata)
        

    os.system('python detection.py')
    s = open('json_file.json','r')
    return s.read()
app.run()
