from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
import pandas as pd
import pickle as pkl


class Crop(APIView):
    def post(self, request):
        Nitrogen = request.data["Nitrogen"]
        phosphorus  = request.data["phosphorus"]
        potassium = request.data["potassium"]
        temperature = request.data["temperature"]
        humidity = request.data["humidity"]
        ph  = request.data["ph"]
        rainfall = request.data["rainfall"]
 
        df = pd.DataFrame({
            "Nitrogen": Nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }, index = [0])

        with open ("./Models/RandomForest.pkl", "rb") as f:
               rf_model = pkl.load(f)

        with open ("./Encoder/LabelEncoder.pkl", "rb") as f:
               encoder = pkl.load(f)
               
        pred = rf_model.predict(df)

        return Response ({
           "Random Forest" : encoder.inverse_transform(pred)
        })
 
