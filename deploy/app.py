from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo, scaler y encoders
best_model = joblib.load('mejor_modelo.joblib')
scaler = joblib.load('scaler.joblib')
encoders = joblib.load('label_encoders.joblib')
column_order = joblib.load('column_order.joblib')

# Definir las columnas categóricas
categorical_columns = ['ClaseObrera', 'Educacion', 'EstadoMarital', 'Ocupacion', 'EstadoCivil', 'Raza', 'Sexo', 'Pais']

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del JSON
    data = request.json
    df = pd.DataFrame([data])
    
    # Crear una copia de los datos para no modificar el original
    data_encoded = df.copy()
    
    # Aplicar Label Encoding a las columnas categóricas
    for col in categorical_columns:
        if col in data_encoded.columns:
            if col not in encoders:
                return jsonify({'error': f"No encoder found for column {col}. Skipping encoding for this column."}), 400
            encoder = encoders[col]
            # Manejar valores desconocidos
            data_encoded[col] = data_encoded[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
            if 'Unknown' not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, 'Unknown')
            data_encoded[col] = encoder.transform(data_encoded[col])
    
    # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
    missing_cols = set(column_order) - set(data_encoded.columns)
    for col in missing_cols:
        data_encoded[col] = 0  # o cualquier otro valor por defecto
    data_encoded = data_encoded.reindex(columns=column_order)
    
    # Convertir las columnas a float64 y llenar NaNs con 0
    for col in data_encoded.columns:
        data_encoded[col] = pd.to_numeric(data_encoded[col], errors='coerce')
        data_encoded[col] = data_encoded[col].fillna(0)  # Reemplazar NaNs con 0
    
    # Escalar los datos
    scaled_features = scaler.transform(data_encoded)
    
    # Hacer la predicción
    prediction = best_model.predict(scaled_features)
    
    # Verificar si el modelo tiene el método predict_proba
    if hasattr(best_model, 'predict_proba'):
        probability = best_model.predict_proba(scaled_features)[:, 1]
    else:
        probability = None
    
    # Devolver la respuesta en formato JSON
    response = {'prediction': int(prediction[0])}
    if probability is not None:
        response['probability'] = float(probability[0])

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
