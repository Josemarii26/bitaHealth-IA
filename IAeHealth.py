

"""
Script completo 
  1) Crea cuatro versiones del dataset a partir de un CSV original de HRV
  2) Entrena múltiples modelos (k-NN, Decision Tree, RandomForest, etc.)
  3) Organiza resultados en carpetas (Models, Plots, Reports) por cada dataset
  4) Genera un resumen (Accuracy, Precision, Recall, F1, AUC) y gráficas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, roc_auc_score)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

import joblib

###############################################################################
# 1. CONFIGURACIÓN DEL DATASET Y COLUMNAS
###############################################################################

ORIGINAL_DATASET_PATH = r"C:\Users\Jm\Desktop\MASTER MII\SEGUNDO\E-HEALTH\PROYECTO-eHEALTH\dataset1-10\dataset1-10.csv"

TARGET_COLUMN = "Label"

NUMERIC_COLS = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD",
    "HRV_pNN50", "HRV_pNN20", "HRV_IQRNN", "HRV_HTI",
    "HRV_TINN", "HRV_HFn", "HRV_LnHF", "ECG_R_Peaks_Count"
]

# Directorio base donde se guardarán todos los resultados
BASE_OUTPUT_DIR = "Resultados_HRV"

###############################################################################
# 2. CREACIÓN DE FUNCIONES DE APOYO
###############################################################################

def create_directory(path):

    os.makedirs(path, exist_ok=True)

def save_model(model, filepath):

    joblib.dump(model, filepath)
    print(f"Modelo guardado en: {filepath}")

def save_report(report_text, filepath):

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Reporte guardado en: {filepath}")

def load_dataset(file_path, target_column=TARGET_COLUMN):

    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def train_evaluate_model(model, X_train, X_test, y_train, y_test, 
                         model_name, dataset_name, output_dirs):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Intentar obtener probabilidades
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        
        pass

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    
    try:
        if y_proba is not None and len(np.unique(y_test)) > 1:
            # one-vs-rest, average=macro
            auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        else:
            auc_score = np.nan
    except:
        auc_score = np.nan

    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {model_name} en {dataset_name} ---")
    print(f"Accuracy: {accuracy:.4f} | Precision(macro): {precision:.4f} | Recall(macro): {recall:.4f} | F1(macro): {f1:.4f} | AUC: {auc_score:.4f}")
    print(report_text)

    report_path = os.path.join(output_dirs['Reports'], f"{model_name}_{dataset_name}_Report.txt")
    save_report(report_text, report_path)

    plt.figure(figsize=(6,5))
    sns.set(style="whitegrid")
    unique_classes = np.unique(y_test)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                     xticklabels=unique_classes,
                     yticklabels=unique_classes)
    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Realidad', fontsize=12)
    ax.set_title(f'Matriz de Confusión - {model_name} en {dataset_name}', fontsize=14, pad=20)
    plt.yticks(rotation=0)
    cm_filepath = os.path.join(output_dirs['Plots'], f"{model_name}_{dataset_name}_Confusion_Matrix.png")
    plt.savefig(cm_filepath, bbox_inches='tight')
    plt.close()

    cm_df = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
    cm_csv_path = os.path.join(output_dirs['Plots'], f"{model_name}_{dataset_name}_Confusion_Matrix.csv")
    cm_df.to_csv(cm_csv_path)


    if y_proba is not None and len(unique_classes) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:,1], pos_label=unique_classes[1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} en {dataset_name}')
        plt.legend(loc="lower right")
        roc_filepath = os.path.join(output_dirs['Plots'], f"{model_name}_{dataset_name}_ROC_Curve.png")
        plt.savefig(roc_filepath, bbox_inches='tight')
        plt.close()

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba[:,1], pos_label=unique_classes[1])
        pr_auc = auc(recall_vals, precision_vals)
        plt.figure()
        plt.plot(recall_vals, precision_vals, color='blue', lw=2, 
                 label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name} en {dataset_name}')
        plt.legend(loc="upper right")
        pr_filepath = os.path.join(output_dirs['Plots'], f"{model_name}_{dataset_name}_Precision_Recall_Curve.png")
        plt.savefig(pr_filepath, bbox_inches='tight')
        plt.close()

    return y_pred, y_proba, {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1, 'auc': auc_score
    }, cm

def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring='accuracy', search_type='grid'):

    if search_type == 'grid':
        from sklearn.model_selection import GridSearchCV
        search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring=scoring, verbose=0)
    else:
        from sklearn.model_selection import RandomizedSearchCV
        search = RandomizedSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring=scoring, random_state=42, verbose=0)

    search.fit(X_train, y_train)
    print(f"Mejores Parámetros: {search.best_params_}")
    print(f"Mejor {search.scoring} (CV): {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_

def visualize_results(results_dict):

    results_df = pd.DataFrame(results_dict).T
    print("\n=== Resultados de Modelos ===")
    print(results_df)

    summary_dir = os.path.join(BASE_OUTPUT_DIR, "Summary")
    create_directory(summary_dir)

    summary_csv_path = os.path.join(summary_dir, "models_results_summary.csv")
    results_df.to_csv(summary_csv_path)
    print(f"Resumen de resultados guardado en: {summary_csv_path}")

    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    for metric in metrics:
        if metric not in results_df.columns:
            continue
        plt.figure(figsize=(12,6))
        sns.barplot(x=results_df.index, y=results_df[metric])
        plt.xticks(rotation=90)
        plt.title(f'{metric} por Modelo y Dataset')
        plt.ylabel(metric)
        plt.xlabel('Modelo - Dataset')
        plt.tight_layout()
        metric_plot_path = os.path.join(summary_dir, f"{metric}_Barplot.png")
        plt.savefig(metric_plot_path)
        plt.close()
        print(f"Gráfico de {metric} guardado en: {metric_plot_path}")

###############################################################################
# 3. CREACIÓN DE LAS 4 VERSIONES DEL DATASET
###############################################################################

def create_datasets_versions(df):
    """
    Crea las 4 versiones del dataset según tu lógica previa.
    Retorna rutas a CSV o DataFrames. Aquí devolveremos DataFrames para el pipeline.
    """

    # -- Versión 1: Eliminar filas con NA, LabelEncoder, StandardScaler --
    df_v1 = df.copy()
    df_v1.dropna(axis=0, inplace=True)

    le_v1 = LabelEncoder()
    df_v1[TARGET_COLUMN] = le_v1.fit_transform(df_v1[TARGET_COLUMN])

    scaler_v1 = StandardScaler()
    df_v1[NUMERIC_COLS] = scaler_v1.fit_transform(df_v1[NUMERIC_COLS])

    # -- Versión 2: Imputar (mediana / moda), LabelEncoder, StandardScaler --
    df_v2 = df.copy()

    imputer_num_v2 = SimpleImputer(strategy='median')
    df_v2[NUMERIC_COLS] = imputer_num_v2.fit_transform(df_v2[NUMERIC_COLS])

    imputer_cat_v2 = SimpleImputer(strategy='most_frequent')
    if df_v2[TARGET_COLUMN].isnull().sum() > 0:
        df_v2[[TARGET_COLUMN]] = imputer_cat_v2.fit_transform(df_v2[[TARGET_COLUMN]])

    le_v2 = LabelEncoder()
    df_v2[TARGET_COLUMN] = le_v2.fit_transform(df_v2[TARGET_COLUMN])

    scaler_v2 = StandardScaler()
    df_v2[NUMERIC_COLS] = scaler_v2.fit_transform(df_v2[NUMERIC_COLS])

    # -- Versión 3: Eliminar col. con >30% nulos, imputar, LabelEncoder, StandardScaler --
    df_v3 = df.copy()
    missing_percent_v3 = df_v3.isnull().mean() * 100
    cols_to_drop_v3 = missing_percent_v3[missing_percent_v3 > 30].index
    df_v3.drop(columns=cols_to_drop_v3, inplace=True)

    numeric_cols_v3 = [c for c in NUMERIC_COLS if c in df_v3.columns]

    imputer_num_v3 = SimpleImputer(strategy='median')
    df_v3[numeric_cols_v3] = imputer_num_v3.fit_transform(df_v3[numeric_cols_v3])

    if df_v3[TARGET_COLUMN].isnull().sum() > 0:
        imputer_cat_v3 = SimpleImputer(strategy='most_frequent')
        df_v3[[TARGET_COLUMN]] = imputer_cat_v3.fit_transform(df_v3[[TARGET_COLUMN]])

    le_v3 = LabelEncoder()
    df_v3[TARGET_COLUMN] = le_v3.fit_transform(df_v3[TARGET_COLUMN])

    scaler_v3 = StandardScaler()
    df_v3[numeric_cols_v3] = scaler_v3.fit_transform(df_v3[numeric_cols_v3])

    # -- Versión 4: Imputar, LabelEncoder, StandardScaler, PCA, SMOTE --
    
    df_v4 = df.copy()
    imputer_num_v4 = SimpleImputer(strategy='median')
    df_v4[NUMERIC_COLS] = imputer_num_v4.fit_transform(df_v4[NUMERIC_COLS])

    if df_v4[TARGET_COLUMN].isnull().sum() > 0:
        imputer_cat_v4 = SimpleImputer(strategy='most_frequent')
        df_v4[[TARGET_COLUMN]] = imputer_cat_v4.fit_transform(df_v4[[TARGET_COLUMN]])

    le_v4 = LabelEncoder()
    df_v4[TARGET_COLUMN] = le_v4.fit_transform(df_v4[TARGET_COLUMN])

    scaler_v4 = StandardScaler()
    df_v4[NUMERIC_COLS] = scaler_v4.fit_transform(df_v4[NUMERIC_COLS])

    X_v4 = df_v4.drop(TARGET_COLUMN, axis=1)
    y_v4 = df_v4[TARGET_COLUMN]

    pca_v4 = PCA(n_components=0.95, random_state=42)
    X_pca_v4 = pca_v4.fit_transform(X_v4)

    smote = SMOTE(random_state=42)
    X_bal_v4, y_bal_v4 = smote.fit_resample(X_pca_v4, y_v4)

    df_v4_final = pd.DataFrame(X_bal_v4, columns=[f"PC{i+1}" for i in range(X_pca_v4.shape[1])])
    df_v4_final[TARGET_COLUMN] = y_bal_v4

    # === GUARDAR OBJETOS DE PREPROCESADO DE VERSIÓN 4 ===
    artifact_dir_4 = os.path.join(BASE_OUTPUT_DIR, "Dataset_4", "Artifacts")
    os.makedirs(artifact_dir_4, exist_ok=True)

    joblib.dump(imputer_num_v4, os.path.join(artifact_dir_4, "imputer_v4.pkl"))
    joblib.dump(le_v4, os.path.join(artifact_dir_4, "labelencoder_v4.pkl"))
    joblib.dump(scaler_v4, os.path.join(artifact_dir_4, "scaler_v4.pkl"))
    joblib.dump(pca_v4, os.path.join(artifact_dir_4, "pca_v4.pkl"))
    print(f"[INFO] Objetos de preprocesado para Dataset_4 guardados en: {artifact_dir_4}")

    return df_v1, df_v2, df_v3, df_v4_final

###############################################################################
# 4. DEFINICIÓN DE MODELOS Y PARÁMETROS
###############################################################################

models_params = {
    'k-NN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [1, 3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 1, 10],
            'penalty': ['l2']  # ajusta 'solver' si cambias penalty
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'MLP Classifier': {
        'model': MLPClassifier(random_state=42, max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001]
        }
    }
}

###############################################################################
# 5. EJECUCIÓN PRINCIPAL
###############################################################################

def main():
    df_orig = pd.read_csv(ORIGINAL_DATASET_PATH)
    print("Forma del dataset original:", df_orig.shape)



    create_directory(BASE_OUTPUT_DIR)

    df_v1, df_v2, df_v3, df_v4 = create_datasets_versions(df_orig)

    df_v1.to_csv(os.path.join(BASE_OUTPUT_DIR, "hrv_dataset_v1.csv"), index=False)
    df_v2.to_csv(os.path.join(BASE_OUTPUT_DIR, "hrv_dataset_v2.csv"), index=False)
    df_v3.to_csv(os.path.join(BASE_OUTPUT_DIR, "hrv_dataset_v3.csv"), index=False)
    df_v4.to_csv(os.path.join(BASE_OUTPUT_DIR, "hrv_dataset_v4.csv"), index=False)

    datasets = {
        "Dataset_1": df_v1,
        "Dataset_2": df_v2,
        "Dataset_3": df_v3,
        "Dataset_4": df_v4
    }

    results = {}

    for dataset_name, df_data in datasets.items():
        print(f"\n=== Procesando {dataset_name} ===")

        dataset_output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
        models_output_dir = os.path.join(dataset_output_dir, 'Models')
        plots_output_dir = os.path.join(dataset_output_dir, 'Plots')
        reports_output_dir = os.path.join(dataset_output_dir, 'Reports')
        create_directory(dataset_output_dir)
        create_directory(models_output_dir)
        create_directory(plots_output_dir)
        create_directory(reports_output_dir)

        output_dirs = {
            'Models': models_output_dir,
            'Plots': plots_output_dir,
            'Reports': reports_output_dir
        }

        X = df_data.drop(TARGET_COLUMN, axis=1)
        y = df_data[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        for model_key, model_info in models_params.items():
            model = model_info['model']
            param_grid = model_info['params']

            print(f"\n--- Entrenando {model_key} en {dataset_name} ---")

            best_model, best_params, best_score = hyperparameter_tuning(
                model, param_grid, X_train, y_train, cv=3,
                scoring='accuracy', search_type='grid'
            )

            y_pred, y_proba, metrics_dict, cm = train_evaluate_model(
                best_model, X_train, X_test, y_train, y_test,
                model_key, dataset_name, output_dirs
            )

            key = f"{model_key} - {dataset_name}"
            results[key] = {
                'Accuracy': metrics_dict['accuracy'],
                'Precision': metrics_dict['precision'],
                'Recall': metrics_dict['recall'],
                'F1-Score': metrics_dict['f1'],
                'AUC': metrics_dict['auc']
            }

            model_filename = f"{model_key.replace(' ', '_')}_{dataset_name}.pkl"
            model_path = os.path.join(models_output_dir, model_filename)
            save_model(best_model, model_path)

    visualize_results(results)

    print("\nFinalizado el procesamiento completo.")

###############################################################################
# 6. EJECUTAR
###############################################################################
if __name__ == "__main__":
    main()
