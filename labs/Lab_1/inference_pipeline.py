from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialisation de l'application Flask
app = Flask(__name__)

def add_combined_feature(X):
    """Fonction de transformation de caractéristiques créée lors de l'entraînement."""
    X = X.copy()
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    return X

# Chargement du modèle entraîné (artefact du pipeline)
model_pipeline = joblib.load('best_cancer_model_pipeline.joblib')

# Liste ordonnée des noms de variables correspondant aux données d'entraînement
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius',
    'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
    'worst fractal dimension'
]

# Route d'accueil
@app.route('/')
def home():
    return "Bienvenue sur l'API de prédiction du cancer du sein !"

# Route de prédiction (POST)
@app.route('/predict', methods=['POST'])
def predict():
    # Récupération des données JSON envoyées par la requête
    data = request.json
    
    # Extraction du vecteur de variables
    features = data['features']
    
    # Conversion du tableau de variables en DataFrame Pandas
    input_data = pd.DataFrame([features], columns=feature_names)
    
    # Prédiction avec le pipeline de modèle chargé
    prediction = model_pipeline.predict(input_data)
    
    # Renvoi du résultat au format JSON
    return jsonify({'prediction': int(prediction[0])})

# Lancement de l'application
if __name__ == '__main__':
    app.run(debug=True, port=5001)