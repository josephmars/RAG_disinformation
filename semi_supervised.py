import json
import pickle

with open('./models/rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('translated_data.json', 'r') as file:
    data = json.load(file)

texts = [item['Translated Text'] for item in data['data']]

predicted_labels = model.predict(texts)

for i, item in enumerate(data['data']):
    item['label'] = predicted_labels[i]

with open('labeled_data.json', 'w') as file:
    json.dump(data, file)