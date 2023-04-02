import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


dataset = pd.read_csv('sti_final.csv')


X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam')


mlp.fit(X_train, y_train)


probs = mlp.predict_proba([[0,2,3,2,3,0,7,7,9,3,4,3,5,5,6,3,4,0,9,3,3,5,8]])
# print(probs)

prob_sum = np.sum(probs)
# print(prob_sum)

indices=np.argsort(probs)
# print(indices)
# print(probs[0][indices])
# if probs[0][indices[0][0]]> probs[0][indices[0][1]]:
#     print("a is larger than b")
# else:
#     print("b is larger than a")
    
top_3 = np.argsort(probs,axis=1)[:, -3:]
# print(top_3)
class_probs = np.take_along_axis(probs, top_3, axis=1)
# print(class_probs)

# print(f"The test sample {X_test[4]} ")
# print("Top three Predicted classes and their corresponding probabilties\n")

disease_mapping = { 0: 'AIDS', 1: 'Chlamydia', 2: 'Genital Herpes', 3: 'Gonorrhea', 4: 'HPV', 5: 'Hepatitis A', 6: 'Hepatitis B', 7: 'Hepatitis C', 8: 'Hepatitis D', 9: 'Syphllis', 10: 'Trichomoniasis'}

disease = list([])

for j in range(3):
        class_idx = top_3[0][2-j]
        class_prob = class_probs[0][2-j]
        print(f"\t{disease_mapping[class_idx]}: {class_prob:.5f}")

        disease.append({ 'name' : disease_mapping[class_idx], 'percent' : round(class_prob*100, 2)})
# print(f"Actual class to which the test sample belongs to {y_test[4]}  ")

print(disease)

import pickle
pickle.dump(mlp, open('disease_predictor.pkl', 'wb'))