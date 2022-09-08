import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'vehicleType':14, 'Fuel_Consumption_Overall_in_per_100km':12})

print(r.json())