import os
import sys
import time
import logging
import warnings
import pandas as pd
import numpy as np
import joblib

# Imports para Scikit-Learn (Comunes, LR, SVM, y RF)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Imports para TensorFlow/Keras (NN)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Importación para el modelo XGBoost
import xgboost as xgb

# --- CONFIGURACIÓN GLOBAL ---

CSV_PATH = "C:/Users/saave/Desktop/Master_Thesis/Credit_card_data/creditcard.csv"
# CAMBIO: Nueva carpeta de salida para los resultados de FEUS
BASE_OUTPUT = "C:/Users/saave/Desktop/data_balance/Recall_scenario2/FEUS"

warnings.filterwarnings("ignore")
N_SIMULATIONS = 30  # Número de simulaciones a realizar
# --- Configuración de logging --- 
LOG_FMT = "%(asctime)s %(levelname)-8s %(message)s"
TARGET_COLUMN_NAME = 'Class'

# --- PARÁMETROS DE FEUS ---
N_SAMPLES_FEUS = 5000 

# Parámetros óptimos para cada modelo (mantenidos del script original para consistencia)
OPTIMAL_NN_PARAMS = {'learning_rate': 0.001, 'dropout_rate': 0.05, 'batch_size': 16, 'epochs': 150}
OPTIMAL_LR_PARAMS = {'penalty': 'l2', 'C': 50, 'solver': 'liblinear', 'max_iter': 2000, 'fit_intercept': True, 'intercept_scaling': 1.0}
OPTIMAL_SVM_PARAMS = {'C': 150, 'kernel': 'rbf', 'gamma': 'scale', 'shrinking': True, 'max_iter':2000} 

OPTIMAL_XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 250,
                       'learning_rate': 0.1, 'max_depth': 1, 'subsample': 0.9, 'colsample_bytree': 0.1, 'gamma': 1, 
                       'min_child_weight': 0.6, 'reg_alpha': 0.9, 'use_label_encoder': False}

OPTIMAL_RF_PARAMS = {'n_estimators': 1000, 'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 2,
                      'min_samples_leaf': 10, 'max_leaf_nodes': None, 'bootstrap': True, 'max_samples': None, 
                      'oob_score': True, 'ccp_alpha': 0.001, 'warm_start': False, 'verbose': 0}

# --- FUNCIONES DE LOGGING Y CARGA DE DATOS ---
def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler); handler.close()
    logging.basicConfig(level=logging.INFO, format=LOG_FMT,
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler(sys.stdout)])

def load_data(csv_path):
    logging.info(f"Cargando datos desde: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado en la ruta: {csv_path}")
        sys.exit("Error: El archivo de datos no fue encontrado. Verifica la variable CSV_PATH.")

# <<< INICIO DE LA MODIFICACIÓN: Implementación de FEUS >>>
def feus_apply_logic(data_features_input: pd.DataFrame, n_samples_to_keep: int, operation_context: str) -> pd.Index:
    """
    Lógica central de FEUS: calcula la distancia de Mahalanobis y selecciona las muestras más lejanas.
    """
    if data_features_input.empty:
        logging.warning(f"FEUS {operation_context}: DataFrame de entrada vacío.")
        return pd.Index([])
    
    # Se usa un scaler interno para el cálculo de Mahalanobis, sin afectar los datos originales.
    scaler_internal = MinMaxScaler()
    X_scaled_for_mahalanobis = pd.DataFrame(scaler_internal.fit_transform(data_features_input),
                                            columns=data_features_input.columns,
                                            index=data_features_input.index)
    
    if X_scaled_for_mahalanobis.shape[0] < 2:
        logging.warning(f"FEUS {operation_context}: No hay suficientes muestras para calcular la covarianza.")
        return data_features_input.head(min(n_samples_to_keep, len(data_features_input))).index

    mean_vector = X_scaled_for_mahalanobis.mean().values
    try:
        inv_cov_matrix = np.linalg.pinv(np.cov(X_scaled_for_mahalanobis.values, rowvar=False))
    except Exception as e:
        logging.error(f"Error al calcular la matriz de covarianza inversa en FEUS: {e}")
        # Fallback: devolver las primeras N muestras si el cálculo falla
        return data_features_input.head(min(n_samples_to_keep, len(data_features_input))).index

    diff_values = X_scaled_for_mahalanobis.values - mean_vector
    # Usar np.einsum para un cálculo eficiente del producto punto diagonal
    distances_sq = np.einsum('ij,ij->i', diff_values.dot(inv_cov_matrix), diff_values)
    distances = np.sqrt(np.maximum(distances_sq, 0)) # Asegurar que no haya valores negativos
    
    data_features_with_dist = data_features_input.copy()
    data_features_with_dist['Mahalanobis_Distance_Internal'] = distances
    
    selected_indices = data_features_with_dist.nlargest(n_samples_to_keep, 'Mahalanobis_Distance_Internal').index
    return selected_indices

def apply_feus_on_train_data(X_train_scaled: pd.DataFrame, y_train: pd.Series, n_samples_to_keep: int) -> (pd.DataFrame, pd.Series):
    """
    Aplica la técnica de submuestreo FEUS (Furthest Euclidean Under-Sampling)
    al conjunto de entrenamiento completo.
    """
    logging.info(f"Aplicando técnica FEUS para seleccionar las {n_samples_to_keep} muestras más lejanas...")
    
    # Asegurarse de que no intentamos seleccionar más muestras de las que hay
    if n_samples_to_keep > len(X_train_scaled):
        logging.warning(f"N_SAMPLES_FEUS ({n_samples_to_keep}) es mayor que el tamaño del set de entrenamiento ({len(X_train_scaled)}). "
                        "Se devolverá el set de entrenamiento completo.")
        return X_train_scaled.reset_index(drop=True), y_train.reset_index(drop=True)

    selected_indices = feus_apply_logic(X_train_scaled, n_samples_to_keep, "TrainSet")
    
    X_res_scaled = X_train_scaled.loc[selected_indices].reset_index(drop=True)
    y_res = y_train.loc[selected_indices].reset_index(drop=True)
    
    logging.info(f"FEUS: Tamaño del conjunto de entrenamiento original: {len(X_train_scaled)}, nuevo tamaño: {len(X_res_scaled)}")
    logging.info(f"Distribución de clases después de FEUS: {dict(y_res.value_counts())}")
    
    return X_res_scaled, y_res

# --- PREPARACIÓN DE DATOS ---
def prepare_data(df_original, target_col_name, random_seed):
    logging.info(">> Iniciando preparación de datos (Split -> Scale -> Balance con FEUS)")
    X_original, y_original = df_original.drop(target_col_name, axis=1), df_original[target_col_name]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, stratify=y_original, random_state=random_seed)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
    
    # CAMBIO: Se llama a la función de FEUS en lugar de MEUS
    X_train_final, y_train_final = apply_feus_on_train_data(X_train_scaled, y_train, N_SAMPLES_FEUS)
    
    return X_train_final, X_test_scaled, y_train_final, y_test, scaler

# --- FUNCIONES ESPECÍFICAS DE CADA MODELO  ---

def train_nn_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo de Red Neuronal (NN)...")
    if X_train.empty: logging.error("X_train está vacío. No se puede entrenar el modelo NN."); return None
    
    # Usar una división para validación si hay suficientes datos y clases
    stratify_val = y_train if len(y_train.unique()) > 1 else None
    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=stratify_val, random_state=random_seed)

    model = Sequential([Dense(32, input_shape=(X_train_new.shape[1],), activation='softplus'), Dropout(OPTIMAL_NN_PARAMS['dropout_rate']),
                        Dense(16, activation='softplus'), Dropout(OPTIMAL_NN_PARAMS['dropout_rate']), Dense(2, activation='softmax')])
    
    model.compile(optimizer=Adam(learning_rate=OPTIMAL_NN_PARAMS['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=0)]
    
    start_time = time.time()
    model.fit(X_train_new, y_train_new, epochs=OPTIMAL_NN_PARAMS['epochs'], batch_size=OPTIMAL_NN_PARAMS['batch_size'],
              validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)
    logging.info(f"Entrenamiento NN completado en {time.time() - start_time:.2f} segundos.")
    model_path = os.path.join(output_dir, f"{exp_name}_nn_model.h5")
    model.save(model_path)
    logging.info(f"✔ Modelo NN (Keras) guardado en: {model_path}")
    return model_path

def evaluate_nn_model(model_path, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo NN...")
    model = load_model(model_path, compile=False)
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación NN:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    return recall, precision

def train_lr_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo de Regresión Logística (LR)...")
    if X_train.empty: logging.error("X_train está vacío. No se puede entrenar el modelo LR."); return None
    model = LogisticRegression(**OPTIMAL_LR_PARAMS, random_state=random_seed)
    start_time = time.time()
    model.fit(X_train, y_train)
    logging.info(f"Entrenamiento LR completado en {time.time() - start_time:.2f} segundos.")
    model_path = os.path.join(output_dir, f"{exp_name}_lr_model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"✔ Modelo LR (joblib) guardado en: {model_path}")
    return model_path

def evaluate_lr_model(model_path, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo LR...")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación LR:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    return recall, precision

def train_svm_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo Support Vector Machine (SVM)...")
    if X_train.empty: logging.error("X_train está vacío. No se puede entrenar el modelo SVM."); return None
    model = SVC(**OPTIMAL_SVM_PARAMS, random_state=random_seed, probability=True)
    start_time = time.time()
    model.fit(X_train, y_train)
    logging.info(f"Entrenamiento SVM completado en {time.time() - start_time:.2f} segundos.")
    model_path = os.path.join(output_dir, f"{exp_name}_svm_model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"✔ Modelo SVM (joblib) guardado en: {model_path}")
    return model_path

def evaluate_svm_model(model_path, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo SVM...")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación SVM:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    return recall, precision

def train_xgb_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo XGBoost...")
    if X_train.empty or y_train.empty:
        logging.error("X_train o y_train están vacíos. No se puede entrenar el modelo XGBoost.")
        return None
        
    stratify_val = y_train if len(y_train.unique()) > 1 else None
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=stratify_val, random_state=random_seed
    )
    current_xgb_params = OPTIMAL_XGB_PARAMS.copy()
    current_xgb_params['seed'] = random_seed
    model = xgb.XGBClassifier(**current_xgb_params)
    start_time = time.time()
    model.fit(
        X_train_new, y_train_new,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    logging.info(f"Entrenamiento XGBoost completado en {time.time() - start_time:.2f} segundos.")
    logging.info(f"Mejor iteración de XGBoost (Early Stopping): {model.best_iteration}")
    model_path = os.path.join(output_dir, f"{exp_name}_xgb_model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"✔ Modelo XGBoost (joblib) guardado en: {model_path}")
    return model_path

def evaluate_xgb_model(model_path, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo XGBoost...")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación XGBoost:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    return recall, precision

def train_rf_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo Random Forest (RF)...")
    if X_train.empty:
        logging.error("X_train está vacío. No se puede entrenar el modelo RF.")
        return None
    
    current_rf_params = OPTIMAL_RF_PARAMS.copy()
    current_rf_params['random_state'] = random_seed
    
    model = RandomForestClassifier(**current_rf_params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    logging.info(f"Entrenamiento RF completado en {time.time() - start_time:.2f} segundos.")
    
    model_path = os.path.join(output_dir, f"{exp_name}_rf_model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"✔ Modelo RF (joblib) guardado en: {model_path}")
    return model_path

def evaluate_rf_model(model_path, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo RF...")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación RF:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    return recall, precision

# --- BLOQUE PRINCIPAL DE EJECUCIÓN (MODIFICADO PARA FEUS) ---
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT, exist_ok=True)
    all_results = []
    all_precision_results = []
    df_original = load_data(CSV_PATH)

    for i in range(1, N_SIMULATIONS + 1):

        exp_name_sim = f"FEUS_Compare_Sim_{i}"
        output_dir_sim = os.path.join(BASE_OUTPUT, exp_name_sim)
        log_file_sim = os.path.join(output_dir_sim, f"run_{exp_name_sim}.log")
        setup_logging(log_file_sim)
        
        print(f"\n{'='*25} INICIANDO SIMULACIÓN CON FEUS {i}/{N_SIMULATIONS} {'='*25}")
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando Simulación Comparativa (FEUS): {i}/{N_SIMULATIONS} ===")
        
        current_seed = i 
        np.random.seed(current_seed)
        tf.random.set_seed(current_seed)
        logging.info(f"Semillas (Numpy, TF, Split, Sklearn, XGB, RF) establecidas en: {current_seed}")
        
        X_train, X_test, y_train, y_test, _ = prepare_data(df_original, TARGET_COLUMN_NAME, random_seed=current_seed)
        
        model_runners = {
            "NN": (train_nn_model, evaluate_nn_model),
            "LR": (train_lr_model, evaluate_lr_model),
            "SVM": (train_svm_model, evaluate_svm_model),
            "XGB": (train_xgb_model, evaluate_xgb_model),
            "RF": (train_rf_model, evaluate_rf_model)
        }
        
        sim_results = {'Simulacion': i}
        sim_precision_results = {'Simulacion': i}

        for model_name, (train_func, eval_func) in model_runners.items():
            logging.info(f"\n--- Procesando Modelo: {model_name} con datos de FEUS ---")
            # El conjunto de entrenamiento ya fue submuestreado con FEUS
            model_path = train_func(X_train.copy(), y_train.copy(), output_dir_sim, exp_name_sim, random_seed=current_seed)
            if model_path:
                recall, precision = eval_func(model_path, X_test, y_test)
                logging.info(f"Macro Avg Recall {model_name}: {recall:.4f} | Macro Avg Precision {model_name}: {precision:.4f}")
                # CAMBIO: Nombres de columnas de resultados para FEUS
                sim_results[f'recall_{model_name}_FEUS'] = recall
                sim_precision_results[f'precision_{model_name}_FEUS'] = precision
            else:
                sim_results[f'recall_{model_name}_FEUS'] = None
                sim_precision_results[f'precision_{model_name}_FEUS'] = None

        all_results.append(sim_results)
        all_precision_results.append(sim_precision_results)
        
        print(f"--- Simulación {i} con FEUS completada. ---")
        logging.info(f"=== Fin Simulación {i} ===\n")

    print(f"\n{'='*25} TODAS LAS SIMULACIONES CON FEUS COMPLETADAS {'='*25}")
    
    # Reporte de Recall
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\nResumen de Resultados de Macro Avg Recall (FEUS):")
        print(results_df.to_string(index=False))
        
        stats_summary = "Estadísticas de Robustez del Macro Avg Recall (FEUS)\n" + "="*60 + "\n"
        stats_summary += f"Total de Simulaciones: {len(results_df)}\n\n"

        for col in results_df.columns:
            if 'recall' in col:
                stats_summary += f"--- Estadísticas para {col} ---\n"
                stats_summary += f"  - Media:          {results_df[col].mean():.4f}\n"
                stats_summary += f"  - Desv. Estándar: {results_df[col].std():.4f}\n"
                stats_summary += f"  - Mínimo:         {results_df[col].min():.4f}\n"
                stats_summary += f"  - Máximo:         {results_df[col].max():.4f}\n\n"

        print("\n" + stats_summary)
        
        # CAMBIO: Nombres de archivos de salida para FEUS
        summary_csv_path = os.path.join(BASE_OUTPUT, 'ALL_MODELS_recall_FEUS_summary.csv')
        results_df.to_csv(summary_csv_path, index=False)
        print(f"Resultados de recall guardados en: {summary_csv_path}")
        
        stats_summary_path = os.path.join(BASE_OUTPUT, 'Statistics_summary_recall_FEUS.txt')
        with open(stats_summary_path, 'w') as f:
            f.write(stats_summary)
        print(f"Estadísticas de robustez de recall guardadas en: {stats_summary_path}")

    # Reporte de Precisión
    if all_precision_results:
        precision_results_df = pd.DataFrame(all_precision_results)
        print("\nResumen de Resultados de Macro Avg Precision (FEUS):")
        print(precision_results_df.to_string(index=False))
        
        # CAMBIO: Nombres de archivos de salida para FEUS
        precision_summary_csv_path = os.path.join(BASE_OUTPUT, 'ALL_MODELS_precision_FEUS_summary.csv')
        precision_results_df.to_csv(precision_summary_csv_path, index=False)
        print(f"\nResultados de precisión guardados en: {precision_summary_csv_path}")
