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

class RestaurantScenario(BaseModel):
    Estado: str  # Example: 'California'
    Negocio: str  # Example: 'kfc'

@app.post('/predict/')
def predict(data: List[RestaurantScenario]):
    predictions = []
    for scenario in data:
        # Get the index corresponding to the state and restaurant chain
        state_column = f'Estado_{scenario.Estado}'
        chain_column = f'Negocio_{scenario.Negocio}'

        input_data = [0] * len(X.columns)  # Initialize with zeros for all columns
        input_data[X.columns.get_loc(state_column)] = 1  # Set the state column to 1
        input_data[X.columns.get_loc(chain_column)] = 1  # Set the restaurant chain column to 1

        prediction = model.predict([input_data])
        predictions.append({
            'Estado': scenario.Estado,
            'Negocio': scenario.Negocio,
            'prediction': prediction[0]
        })
    
    return predictions
