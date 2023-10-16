from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import List

app = FastAPI()

# Load the DataFrame from the CSV file
df = pd.read_csv('MLRESULTS.csv')

# Independent variables and dependent variable
X = df[['Estado_California', 'Estado_Colorado', 'Estado_Georgia', 'Estado_NY',
       'Estado_Texas', 'Negocio_kfc', 'Sentimiento_Negativo', 'Sentimiento_Neutral',
       'Sentimiento_Positivo', 'Numero_reviews']]
y = df['Calificacion_promedio']

# Train the model
model = LinearRegression()
model.fit(X, y)

class RestaurantData(BaseModel):
    Estado_California: int
    Estado_Colorado: int
    Estado_Georgia: int
    Estado_NY: int
    Estado_Texas: int
    Negocio_kfc: int
    Sentimiento_Negativo: int
    Sentimiento_Neutral: int
    Sentimiento_Positivo: int
    Numero_reviews: int

@app.post('/predict/')
def predict(data: List[RestaurantData]):
    predictions = []
    for entry in data:
        input_data = [entry.Estado_California, entry.Estado_Colorado, entry.Estado_Georgia, entry.Estado_NY,
                      entry.Estado_Texas, entry.Negocio_kfc, entry.Sentimiento_Negativo, entry.Sentimiento_Neutral,
                      entry.Sentimiento_Positivo, entry.Numero_reviews]

        prediction = model.predict([input_data])
        predictions.append({'prediction': prediction[0]})
    
    return predictions
