from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import json
app = Flask(__name__)

CORS(app)

@app.route('/voice', methods=['POST'])
def voice_based():
    
    voice_input = json.loads(request.data)
    voice_input=voice_input['data']
    lst =[[float(voice_input["input1"]),float(voice_input["input2"]),float(voice_input["input3"]),float(voice_input["input4"]),float(voice_input["input5"]),float(voice_input["input6"]),float(voice_input["input7"]),float(voice_input["input8"]),float(voice_input["input9"]),float(voice_input["input10"]),float(voice_input["input11"]),float(voice_input["input12"]),float(voice_input["input13"]),float(voice_input["input14"]),float(voice_input["input15"]),float(voice_input["input16"]),float(voice_input["input17"]),float(voice_input["input18"]),float(voice_input["input19"]),float(voice_input["input20"]),float(voice_input["input21"]),float(voice_input["input22"])]]
    # voice_input = voice_input[0]
    # print(voice_input)
    # lst = []
    
    with open('voice_model.pkl', 'rb') as file:
        saved_model = pickle.load(file)
    op = saved_model.predict(lst)
    if op[0] == 1:
        return jsonify({'message': 'You show sign of parkinsons, Please seek medical guidance as early'})
    else:
        return jsonify({'message': 'You are currently not at a risk of Parkinsons'})

@app.route('/symptoms', methods=['POST'])
def symptoms_based():
    symptoms_input = json.loads(request.data)
    symptoms_input=symptoms_input['data']
    symptoms_input=[[int(symptoms_input["input1"]),int(symptoms_input["input2"]),int(symptoms_input["input3"]),int(symptoms_input["input4"]),int(symptoms_input["input5"]),int(symptoms_input["input6"]),int(symptoms_input["input7"]),int(symptoms_input["input8"]),int(symptoms_input["input9"]),int(symptoms_input["input10"]),int(symptoms_input["input11"]),int(symptoms_input["input12"]),int(symptoms_input["input13"])]]
    with open('symptoms_model.pkl', 'rb') as file:
        saved_model = pickle.load(file)
    op = saved_model.predict(symptoms_input)
    if op[0] == 1:
        return jsonify({'message': 'You show sign of parkinsons, Please seek medical guidance as early'})
    else:
        return jsonify({'message': 'You are currently not at a risk of Parkinsons'})


if __name__ == '__main__':
    app.run()
