from flask import Flask, request, jsonify
import pickle
import os
import gdown

app = Flask(__name__)

# Google Drive file mapping
model_files = {
    "models-api/trained_model.pkl": "1HIPk5W5Ia8PRBDvmJWlDsaicxkxoPq06",
    "models-api/timeseries_pipeline_model.pkl": "1phGvaIjJiDwFok4ush9kRfdXL4hF84oS",
    "models-api/label_encoders.pkl": "10H__IvnotRcDvdcmbSjK7lZggl6F5_ro",
    "models-api/time_series_encoders.pkl": "11k172tkSHJ2TuR2ZXQud2vJW8J63h0Gj"
}

def download_if_missing():
    os.makedirs("models-api", exist_ok=True)
    for path, file_id in model_files.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)

download_if_missing()

# Load model & encoder
model_regresi = pickle.load(open('./models-api/trained_model.pkl', 'rb'))
model_time_series = pickle.load(open('./models-api/timeseries_pipeline_model.pkl', 'rb'))
encoder_regresi = pickle.load(open('./models-api/label_encoders.pkl', 'rb'))
encoder_timeseries = pickle.load(open('./models-api/time_series_encoders.pkl', 'rb'))

features_regresi = ["User_ID", "Bidang_Usaha", "Tahun", "Bulan", "Pendapatan", "Beban_Operasional", "Pajak"]
features_timeseries = ["Bidang_Usaha", "Tahun", "Bulan", "Pendapatan", "Beban_Operasional", "Pajak", "Laba_Rugi_Lag"]

def interpret_output(value):
    return "Laba" if value >= 0 else "Rugi"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data.get("model")
    fitur_input = data.get("fitur")

    if not model_type or not fitur_input:
        return jsonify({"error": "Model dan fitur harus disediakan"}), 400

    try:
        if model_type == "regresi":
            input_data = [fitur_input[feat] for feat in features_regresi]
            input_data[1] = encoder_regresi["Bidang_Usaha"].transform([input_data[1]])[0]
            pred = model_regresi.predict([input_data])[0]
        elif model_type == "time_series":
            input_data = [fitur_input[feat] for feat in features_timeseries]
            input_data[0] = encoder_timeseries["Bidang_Usaha"].transform([input_data[0]])[0]
            pred = model_time_series.predict([input_data])[0]
        else:
            return jsonify({"error": "Model harus 'regresi' atau 'time_series'"}), 400

        hasil = interpret_output(pred)
        return jsonify({"hasil_prediksi": hasil, "nilai_prediksi": float(pred)})

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
