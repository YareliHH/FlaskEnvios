from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import pickle
import joblib  # para cargar el encoder
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Cargar escalador, modelo y encoder
try:
    scaler = joblib.load('scalerGis.pkl')
    logging.info("scalerGis.pkl cargado correctamente.")
except Exception as e:
    scaler = StandardScaler()
    logging.error(f"❌ Error al cargar scalerGis.pkl: {e}")

try:
    model = joblib.load('randomforest_model.pkl')
    logging.info("randomforest_model.pkl cargado correctamente.")
except Exception as e:
    model = None
    logging.error(f"❌ Error al cargar randomforest_model.pkl: {e}")

try:
    encoder = joblib.load('ordinalencoder_estado.pkl')
    logging.info("ordinalencoder_estado.pkl cargado correctamente.")
except Exception as e:
    encoder = None
    logging.error(f"❌ Error al cargar ordinalencoder_estado.pkl: {e}")

@app.route('/calcular_envio', methods=['POST'])
def calcular_envio():
    try:
        data = request.get_json()

        # Campos esperados
        num_items = data.get('num_items')
        subtotal = data.get('subtotal', 0.0)
        total_quantity = data.get('total_quantity', 0)
        total = data.get('total', 0.0)
        estado = data.get('estado')

        if num_items is None:
            return jsonify({"error": "Debes proporcionar el campo 'num_items'."}), 400
        if estado is None:
            return jsonify({"error": "Debes proporcionar el campo 'estado'."}), 400

        num_items = int(num_items)
        total_quantity = int(total_quantity)
        subtotal = float(subtotal)
        total = float(total)

        # Codificar estado si el encoder está disponible
        if encoder is not None:
            try:
                estado_encoded = encoder.transform([[estado]])[0][0]
            except Exception as e:
                logging.error(f"Error al codificar estado: {e}")
                return jsonify({"error": f"Estado '{estado}' no válido."}), 400
        else:
            estado_encoded = 0  # fallback

        # Preparar input para modelo (ordenar las features igual que en entrenamiento)
        input_data = [[total_quantity, total, subtotal, num_items, estado_encoded]]

        # Escalar
        input_scaled = scaler.transform(input_data)

        # Predecir si modelo disponible
        if model is not None:
            costo_envio_pred = model.predict(input_scaled)[0]
        else:
            # lógica fija si no hay modelo
            if num_items <= 3:
                costo_envio_pred = 150.0
            elif num_items <= 6:
                costo_envio_pred = 200.0
            else:
                costo_envio_pred = 300.0

        return jsonify({
            "num_items": num_items,
            "subtotal": subtotal,
            "estado": estado,
            "costo_envio": float(costo_envio_pred)
        }), 200

    except Exception as e:
        logging.error(f"❌ Error en /calcular_envio: {e}")
        return jsonify({"error": "Error al procesar la solicitud."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
