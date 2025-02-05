
import os
import io
import base64
import numpy as np
import pandas as pd
import joblib
import neurokit2 as nk
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

NUMERIC_COLS = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD",
    "HRV_pNN50", "HRV_pNN20", "HRV_IQRNN", "HRV_HTI",
    "HRV_TINN", "HRV_HFn", "HRV_LnHF", "ECG_R_Peaks_Count"
]

IMPUTER_PATH = r"C:\Users\Jm\Desktop\MASTER MII\SEGUNDO\E-HEALTH\PROYECTO-eHEALTH\Resultados_HRV\Dataset_4\Artifacts\imputer_v4.pkl"
SCALER_PATH = r"C:\Users\Jm\Desktop\MASTER MII\SEGUNDO\E-HEALTH\PROYECTO-eHEALTH\Resultados_HRV\Dataset_4\Artifacts\scaler_v4.pkl"
PCA_PATH = r"C:\Users\Jm\Desktop\MASTER MII\SEGUNDO\E-HEALTH\PROYECTO-eHEALTH\Resultados_HRV\Dataset_4\Artifacts\pca_v4.pkl"
LABELENCODER_PATH = r"C:\Users\Jm\Desktop\MASTER MII\SEGUNDO\E-HEALTH\PROYECTO-eHEALTH\Resultados_HRV\Dataset_4\Artifacts\labelencoder_v4.pkl"
MODEL_PATH = r"C:\Users\Jm\Desktop\MASTER MII\SEGUNDO\E-HEALTH\PROYECTO-eHEALTH\Resultados_HRV\Dataset_4\Models\k-NN_Dataset_4.pkl"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def generate_graphs(ecg_cleaned, info_ecg, sampling_rate):
    import io
    import base64
    import matplotlib.pyplot as plt

    graphs = {}

    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Gráfico 1: Señal ECG limpia
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(ecg_cleaned, color="green", label="ECG limpio")
    ax1.set_title("Señal ECG limpia")
    ax1.set_xlabel("Muestras")
    ax1.set_ylabel("ECG (mV)")
    ax1.legend()
    graphs["ecg_plot"] = fig_to_base64(fig1)

    rr_intervals = np.diff(info_ecg["ECG_R_Peaks"]) / sampling_rate * 1000

    # Gráfico 2: Serie temporal de intervalos RR
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(rr_intervals, marker='o', linestyle='-', color="green", label="Intervalos RR (ms)")
    ax2.set_title("Serie temporal de intervalos RR")
    ax2.set_xlabel("Índice de latido")
    ax2.set_ylabel("Intervalo RR (ms)")
    ax2.legend()
    graphs["rr_plot"] = fig_to_base64(fig2)

    # Gráfico 3: Poincaré Plot (RR(n) vs. RR(n+1))
    if len(rr_intervals) > 1:
        rr1 = rr_intervals[:-1]
        rr2 = rr_intervals[1:]
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.scatter(rr1, rr2, color="darkgreen", alpha=0.7)
        ax3.set_title("Poincaré Plot")
        ax3.set_xlabel("RR(n) (ms)")
        ax3.set_ylabel("RR(n+1) (ms)")
        ax3.grid(True)
        graphs["poincare_plot"] = fig_to_base64(fig3)
    else:
        graphs["poincare_plot"] = None

    # Gráfico 4: Histograma de intervalos RR
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.hist(rr_intervals, bins=20, color="green", edgecolor="black", alpha=0.75)
    ax4.set_title("Histograma de intervalos RR")
    ax4.set_xlabel("Intervalo RR (ms)")
    ax4.set_ylabel("Frecuencia")
    graphs["rr_histogram"] = fig_to_base64(fig4)

    # Gráfico 5: Espectro de potencia usando FFT de los intervalos RR
    if len(rr_intervals) > 0:
        fft_vals = np.fft.rfft(rr_intervals - np.mean(rr_intervals))
        fft_freq = np.fft.rfftfreq(len(rr_intervals), d=1/sampling_rate)
        power = np.abs(fft_vals)**2

        fig5, ax5 = plt.subplots(figsize=(12, 4))
        ax5.plot(fft_freq, power, color="darkgreen")
        ax5.set_title("Espectro de potencia (FFT) de RR")
        ax5.set_xlabel("Frecuencia (Hz)")
        ax5.set_ylabel("Potencia")
        graphs["rr_fft"] = fig_to_base64(fig5)
    else:
        graphs["rr_fft"] = None

    # Gráfico 6: Segmento ECG ampliado con pico destacado
    if len(info_ecg["ECG_R_Peaks"]) > 0:
        first_peak = info_ecg["ECG_R_Peaks"][0]
        start = max(0, first_peak - 50)
        end = min(len(ecg_cleaned), first_peak + 150)
        zoom_segment = ecg_cleaned[start:end]
        fig6, ax6 = plt.subplots(figsize=(12, 4))
        ax6.plot(range(start, end), zoom_segment, color="darkgreen",label="Segmento ECG")
        ax6.axvline(x=first_peak, color="red", linestyle="--", label="R Peak")
        ax6.set_title("Segmento ECG ampliado con pico destacado")
        ax6.set_xlabel("Muestras")
        ax6.set_ylabel("ECG (mV)")
        ax6.legend()
        graphs["ecg_zoom_plot"] = fig_to_base64(fig6)
    else:
        graphs["ecg_zoom_plot"] = None

    return graphs

def process_bitalino_txt(txt_path, window_seconds=10, sampling_rate=1000):

    #Procesa el archivo TXT de Bitalino, segmentando la señal ECG en ventanas de 'window_seconds' y calculando las métricas HRV para cada ventana.
    
    datos_raw = pd.read_csv(txt_path, sep="\t", comment="#", header=0, usecols=range(6))
    datos_raw.columns = ['nSeq', 'I1', 'I2', 'O1', 'O2', 'A2']

    vcc = 3.0
    max_adc = 1023
    datos_raw['ECG_mV'] = ((datos_raw['A2'] / max_adc) - 0.5) * vcc

    ecg_signal = datos_raw['ECG_mV'].values
    total_samples = len(ecg_signal)

    window_samples = window_seconds * sampling_rate


    rows = []
    ecg_cleaned_all = []
    info_ecg_all = []

    for start in range(0, total_samples - window_samples + 1, window_samples):
        end = start + window_samples
        segment = ecg_signal[start:end]

        ecg_cleaned = nk.ecg_clean(segment, sampling_rate=sampling_rate, method="neurokit")
        signals_ecg, info_ecg = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        print(f"Ventana {start}-{end}: {len(info_ecg['ECG_R_Peaks'])} picos R detectados.")

        hrv_time = nk.hrv_time(info_ecg["ECG_R_Peaks"], sampling_rate=sampling_rate, show=False)
        hrv_freq = nk.hrv_frequency(info_ecg["ECG_R_Peaks"], sampling_rate=sampling_rate, show=False)
        hrv_features = {**hrv_time, **hrv_freq}

        if "HRV_HF" in hrv_features and "HRV_LF" in hrv_features:
            HRV_HF = hrv_features["HRV_HF"]
            HRV_LF = hrv_features["HRV_LF"]
            hrv_features["HRV_HFn"] = HRV_HF / (HRV_HF + HRV_LF)
            hrv_features["HRV_LnHF"] = np.log(HRV_HF)
        else:
            hrv_features["HRV_HFn"] = np.nan
            hrv_features["HRV_LnHF"] = np.nan

        hrv_features["ECG_R_Peaks_Count"] = len(info_ecg["ECG_R_Peaks"])

        data_filtrada = {}
        for key in NUMERIC_COLS:
            if key in hrv_features:
                value = hrv_features[key]
                if isinstance(value, pd.Series):
                    data_filtrada[key] = value.iloc[0]
                elif isinstance(value, np.ndarray):
                    data_filtrada[key] = value[0] if value.size > 0 else np.nan
                else:
                    data_filtrada[key] = value
            else:
                data_filtrada[key] = np.nan

        data_filtrada["window_start"] = start
        data_filtrada["window_end"] = end

        rows.append(data_filtrada)
        ecg_cleaned_all.append(ecg_cleaned)
        info_ecg_all.append(info_ecg)

    df_hrv = pd.DataFrame(rows)
    return df_hrv, ecg_cleaned_all, info_ecg_all, sampling_rate

def run_model(df_features):
    missing_cols = [c for c in NUMERIC_COLS if c not in df_features.columns]
    if missing_cols:
        return {"error": f"Faltan columnas: {missing_cols}"}

    X_new = df_features[NUMERIC_COLS].copy()

    if not (os.path.exists(IMPUTER_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(PCA_PATH)):
        return {"error": "No se encontraron los objetos de preprocesado (imputer, scaler, PCA)."}
    imputer_v4 = joblib.load(IMPUTER_PATH)
    scaler_v4 = joblib.load(SCALER_PATH)
    pca_v4 = joblib.load(PCA_PATH)

    le_v4 = None
    if os.path.exists(LABELENCODER_PATH):
        le_v4 = joblib.load(LABELENCODER_PATH)

    X_new_imputed = imputer_v4.transform(X_new)
    X_new_scaled = scaler_v4.transform(X_new_imputed)
    X_new_pca = pca_v4.transform(X_new_scaled)
    print(f"Tras PCA, shape = {X_new_pca.shape}")

    if not os.path.exists(MODEL_PATH):
        return {"error": f"No se encontró el modelo en {MODEL_PATH}"}
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado: {MODEL_PATH}")

    y_pred = model.predict(X_new_pca)
    if le_v4 is not None:
        y_pred_clases = le_v4.inverse_transform(y_pred)
    else:
        y_pred_clases = y_pred

    total = len(y_pred_clases)
    conteo = {}
    for val, cnt in zip(*np.unique(y_pred_clases, return_counts=True)):
        pct = (cnt / total) * 100.0
        conteo[val] = {"count": int(cnt), "percentage": round(pct, 2)}

    result = {
        "predictions": y_pred_clases.tolist(),
        "summary": conteo,
        "total_windows": total
    }
    return result

@app.route('/upload', methods=["POST"])
def upload():
    if "txtFile" not in request.files:
        return jsonify({"error": "No se recibió el archivo TXT"}), 400

    txt_file = request.files["txtFile"]
    if txt_file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    filename = secure_filename(txt_file.filename)
    txt_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    txt_file.save(txt_path)

    try:
        df_features, ecg_cleaned_all, info_ecg_all, sampling_rate = process_bitalino_txt(txt_path)
        print("Características HRV extraídas:")
        print(df_features)

        result_model = run_model(df_features)

        # Generar gráficos para la primera ventana 
        graphs = generate_graphs(ecg_cleaned_all[0], info_ecg_all[0], sampling_rate)

        result = {
            "model_result": result_model,
            "graphs": graphs,
            "num_windows": len(df_features)
        }

        os.remove(txt_path)
        return jsonify(result)
    except Exception as e:
        if os.path.exists(txt_path):
            os.remove(txt_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))  # Replit asigna el puerto vía la variable de entorno PORT
    app.run(host="0.0.0.0", port=port, debug=True)
