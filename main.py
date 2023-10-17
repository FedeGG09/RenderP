from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Cargar el DataFrame desde el archivo CSV
df = pd.read_csv('MLRESULTS.csv')

# Variables independientes y variable dependiente
X = df[['Estado_California', 'Estado_Colorado', 'Estado_Georgia', 'Estado_NY',
       'Estado_Texas', 'Negocio_kfc', 'Sentimiento_Negativo', 'Sentimiento_Neutral',
       'Sentimiento_Positivo', 'Numero_reviews']]
y = df['Calificacion_promedio']

# Entrenar el modelo
model = LinearRegression()
model.fit(X, y)

class EstadoInput(BaseModel):
    Estado: str  # Ejemplo: 'California'

@app.post('/predict/')
def predict(data: EstadoInput):
    estado = data.Estado

    if f'Estado_{estado}' not in X.columns:
        return {"error": f"El estado {estado} no está en el conjunto de datos"}

    input_data = [0] * len(X.columns)  # Inicializar con ceros para todas las columnas
    input_data[X.columns.get_loc(f'Estado_{estado}')] = 1  # Establecer la columna del estado en 1
    input_data[X.columns.get_loc('Negocio_kfc')] = 1  # Establecer la columna de Negocio_kfc en 1 (siempre se asume kfc en este ejemplo)

    prediction = model.predict([input_data])

    return {"Estado": estado, "prediction": prediction[0]}
