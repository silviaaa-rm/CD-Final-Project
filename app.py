# -- Backend --

# Flask para crear servidor web
from flask import Flask, request, jsonify, send_from_directory

# Conectar frontendcon backend
from flask_cors import CORS

# joblib
import joblib

# pandas para DataFrame
import pandas as pd


# Crear la app Flask
app = Flask(__name__)
CORS(app)

# Cargar modelo y escalador
modelo = joblib.load("ModeloMLP.pkl")
sc = joblib.load("SC.pkl")

# Columnas del modelo
columnas_modelo = [
    "Education",
    "JoiningYear",
    "PaymentTier",
    "Age",
    "Gender",
    "EverBenched",
    "ExperienceInCurrentDomain",
    "City_New Delhi",
    "City_Pune"
]

# Variables a escalar
columnas_escalar = [
    "JoiningYear",
    "Age",
    "ExperienceInCurrentDomain"
]

# Estos rangos sirven para mostrar advertencias.
# No bloquean la predicción.
rango_dataset = {
    "JoiningYear": (2012, 2018),
    "Age": (22, 58),
    "ExperienceInCurrentDomain": (0, 7)
}

# Ruta principal
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# Ruta para hacer predicciones
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recibir datos enviados desde JavaScript
        data = request.get_json()

        # Variables codificadas desde selects de HTML
        education = int(data["Education"]) # Bachelors = 0; Masters = 1; PHD = 2
        payment_tier = int(data["PaymentTier"]) # 1, 2 o 3
        gender = int(data["Gender"]) # Female = 0; Male = 1
        ever_benched = int(data["EeverBenched"]) # No = 0; Yes = 1

        # Variables escritas
        joining_year = int(data["JoiningYear"])
        age = int(data["Age"])
        experience = int(data["ExperienceInCurrentDomain"])

        # Variable de texto
        city = data["City"]

        # Validaciones básicas para cuidar que todo venga en orden
        if education not in [0, 1, 2]:
            return jsonify({"error": "Education debe ser 0, 1 o 2."}), 400
        
        if payment_tier not in [1, 2, 3]:
            return jsonify({"error": "PaymentTier debe ser 1, 2 o 3."}), 400
        
        if gender not in [0, 1]:
            return jsonify({"error": "Gender debe ser 0 o 1."}), 400
        
        if ever_benched not in [0, 1]:
            return jsonify({"error": "EverBenched debe ser 0 o 1."}), 400
        
        if joining_year <= 1990:
            return jsonify({"error": "Joining Year debe ser mayor a 1990."}), 400

        if age <= 0:
            return jsonify({"error": "Age debe ser mayor a 0."}), 400

        if experience < 0:
            return jsonify({"error": "Experience no puede ser negativa."}), 400
        
        # Validaciones para números fuera del rango
        warnings = []
        min_year, max_year = rango_dataset["JoiningYear"]
        if joining_year < min_year or joining_year > max_year:
            warnings.append(
                f"Joining Year está fuera del rango observado en el dataset ({min_year}-{max_year}). "
                "La predicción puede ser menos confiable."
            )

        min_age, max_age = rango_dataset["Age"]
        if age < min_age or age > max_age:
            warnings.append(
                f"Age está fuera del rango observado en el dataset ({min_age}-{max_age}). "
                "La predicción puede ser menos confiable."
            )

        min_exp, max_exp = rango_dataset["ExperienceInCurrentDomain"]
        if experience < min_exp or experience > max_exp:
            warnings.append(
                f"Experience está fuera del rango observado en el dataset ({min_exp}-{max_exp}). "
                "La predicción puede ser menos confiable."
            )

        # Codificación de City
        # Bangalore [0][0]
        # New Delhi [1][0]
        # Pune [0][1]
        if city == "Bangalore":
            city_new_delhi = 0
            city_pune = 0
        elif city == "New Delhi":
            city_new_delhi = 1
            city_pune = 0
        elif city == "Pune":
            city_new_delhi = 0
            city_pune = 1
        else:
            return jsonify({"error": "City no es válida."}), 400
        
        # Crear DataFrame con el empleado
        nuevo_empleado = pd.DataFrame([{
            "Education": education,
            "JoiningYear": joining_year,
            "PaymentTier": payment_tier,
            "Age": age,
            "Gender": gender,
            "EverBenched": ever_benched,
            "ExperienceInCurrentDomain": experience,
            "City_New Delhi": city_new_delhi,
            "City_Pune": city_pune
        }])

        # Asegurarse que se siga el mismo orden de las columnas
        nuevo_empleado = nuevo_empleado[columnas_modelo]

        # Escalar variables numéricas
        nuevo_empleado[columnas_escalar] = sc.transform(
            nuevo_empleado[columnas_escalar]
        )

        # Llevar a cabo predicción
        # Predict regresa 0 o 1
        prediccion = modelo.predict(nuevo_empleado)[0]

        # Da la probabilidad de cada clase
        probabilidades = modelo.predict_proba(nuevo_empleado)[0]

        # Revisar orden de las clases del modelo
        clases = list(modelo.classes_)

        indice_stay = clases.index(0)
        indice_leave = clases.index(1)

        prob_stay = float(probabilidades[indice_stay])
        prob_leave = float(probabilidades[indice_leave])

        # Recomendaciones
        if int(prediccion) == 1:
            recomendaciones = [
                "Realizar una entrevista de permanencia para identificar inconformidades antes de que tome una decisión definitiva.",
                "Revisar el nivel salarial y compararlo con el mercado actual, especialmente si está en Payment Tier 1 o 2.",
                "Ofrecer oportunidades de crecimiento, cambio de proyecto o capacitación para renovar su motivación."
            ]
        else:
            recomendaciones = [
                "Mantener seguimiento periódico de satisfacción para detectar cambios a tiempo.",
                "Conservar y reforzar las condiciones laborales que han contribuido a su estabilidad.",
                "Reconocer su compromiso con la empresa y apoyar su desarrollo profesional a largo plazo."
            ]
        
        # Regresar respuesta al  HTML
        return jsonify({
            "prediction": int(prediccion),
            "prob_leave": prob_leave,
            "prob_stay": prob_stay,
            "main_factors": recomendaciones,
            "warnings": warnings
        })
    
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    
    # Ejecutar servidor
    if __name__ == "__main__":
        app.run(debug=True)
    