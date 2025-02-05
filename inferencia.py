

import os
import numpy as np
import pandas as pd
import joblib



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

CSV_DATOS_NUEVOS = r"C:\Users\Jm\Documents\resultados_hrv2.csv"


def main():
    input_csv = CSV_DATOS_NUEVOS
    if not os.path.exists(input_csv):
        print(f"ERROR: No existe el archivo {input_csv}")
        return

    df_new = pd.read_csv(input_csv)
    print(f"\nCargado archivo {input_csv} con forma {df_new.shape}")

    missing_cols = [c for c in NUMERIC_COLS if c not in df_new.columns]
    if missing_cols:
        print(f"ERROR: Faltan columnas en el CSV de inferencia: {missing_cols}")
        return

    X_new = df_new[NUMERIC_COLS].copy()

    if not (os.path.exists(IMPUTER_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(PCA_PATH)):
        print("ERROR: No se encontraron todos los objetos de preprocesado (imputer, scaler, pca).")
        return

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
        print(f"ERROR: No se encontró el modelo en {MODEL_PATH}")
        return
    svm_model = joblib.load(MODEL_PATH)
    print(f"Modelo SVM cargado con éxito: {MODEL_PATH}")

    y_pred = svm_model.predict(X_new_pca)

    if le_v4 is not None:
        print("LabelEncoder (versión 4) clases =>", le_v4.classes_)
        y_pred_clases = le_v4.inverse_transform(y_pred)
    else:
        y_pred_clases = y_pred

    print("\n=== PREDICCIONES ===\n")
    unique_vals, counts = np.unique(y_pred_clases, return_counts=True)
    total = len(y_pred_clases)
    for val, cnt in zip(unique_vals, counts):
        pct = (cnt / total) * 100.0
        print(f"Clase '{val}': {cnt} filas -> {pct:.2f}%")
    print(f"Total filas: {total}")

    print("\n=== Detalle (primeras 5 filas) ===")
    for i in range(min(15, total)):
        print(f"Fila {i} => Predicción: {y_pred_clases[i]}")

if __name__ == "__main__":
    main()
