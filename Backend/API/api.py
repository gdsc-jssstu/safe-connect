from flask import Flask, jsonify, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/disease_predictor', methods = ['POST'])
def predict_disease():
    mlp = pickle.load(open('/home/brahmakeerthi/mysite/disease_predictor.pkl', 'rb'))
    disease = request.form.get('disease')

    disease = list(disease)
    input = list([])

    for i in disease:
        if i.isnumeric() :
            input.append(int(i))

    input = np.array([input])
    probs = mlp.predict_proba([input][0])

    top_3 = np.argsort(probs,axis=1)[:, -3:]
    class_probs = np.take_along_axis(probs, top_3, axis=1)

    disease_mapping = { 0: 'AIDS', 1: 'Chlamydia', 2: 'Genital Herpes', 3: 'Gonorrhea', 4: 'HPV', 5: 'Hepatitis A', 6: 'Hepatitis B', 7: 'Hepatitis C', 8: 'Hepatitis D', 9: 'Syphilis', 10: 'Trichomoniasis'}

    disease = list([])

    for j in range(3):
        class_idx = top_3[0][2-j]
        class_prob = class_probs[0][2-j]
        print(f"\t{disease_mapping[class_idx]}: {class_prob:.5f}")
        disease.append({ 'name' : disease_mapping[class_idx], 'percent' : round(class_prob*100, 2)})

    return jsonify( {'disease' : disease} )


@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.debug=True
    app.run(port=8080)


