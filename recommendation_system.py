import json
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

"""**Pre-Processed Data**"""

#open data

with open('parking_data.json') as json_data:
  parking_data = json.load(json_data)

parking_data

column_names = ['parkingId','name','latitude','longitude', 'rating_count', 'rating']

#instantiate the dataframe
callData = pd.DataFrame(columns=column_names)

#placed into a data frame

for data in parking_data:
  parkingId = data['parkingId']
  name = data['name']
  latitude = data['latitude']
  longitude = data['longitude']
  rating_count = data['rating_count']
  rating = data['rating']

  callData = callData.append({
      'parkingId' : parkingId,
      'name' : name,
      'latitude' : latitude,
      'longitude' : longitude,
      'rating_count' : rating_count,
      'rating' : rating}, ignore_index=True)

callData

"""**K-Means Clustering**"""

#variable that need in system but can be change

K = range(1,11)
kmax = 50
n_clusters = 5

coordinate = callData[['longitude','latitude']]

#clustering

kmeans = KMeans(n_clusters, init='k-means++')
kmeans.fit(coordinate)
callData['cluster'] = kmeans.predict(coordinate)
callData

top_parking = callData.sort_values(by=['rating_count', 'rating'], ascending=False)

def recommend_parking(callData, latitude, longitude):
    # Predict the cluster
    cluster = kmeans.predict(np.array([latitude, longitude]).reshape(1,-1))[0]
    print(cluster)
   
    # Get recommended parking
    return  callData[callData['cluster']==cluster].iloc[0:5][['name', 'rating']]
  

# FLASK

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/predict", methods=['POST'])


def predict():
    data = request.get_json()
    
    lat = data['lat']
    lng = data['lng']

    prediction = recommend_parking(top_parking, lng, lat)
    output = prediction.to_json()

    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)

app.run(debug=True)
