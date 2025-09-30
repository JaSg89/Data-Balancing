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

# Import para imbalanced-learn (RUS)
from imblearn.under_sampling import RandomUnderSampler

# Imports para TensorFlow/Keras (NN)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb


# ¡IMPORTANTE! Asegúrate de que estas rutas sean correctas para tu sistema.
CSV_PATH = "C:/Users/saave/Desktop/Master_Thesis/Credit_card_data/creditcard.csv"
# *** CAMBIO IMPORTANTE: Nueva carpeta de salida para resultados de RUS ***
BASE_OUTPUT = "C:/Users/saave/Desktop/data_balance/Recall_scenario2/RUS"

warnings.filterwarnings("ignore")
N_SIMULATIONS = 30 
LOG_FMT = "%(asctime)s %(levelname)-8s %(message)s"
TARGET_COLUMN_NAME = 'Class'

# Parámetros óptimos para cada modelo (los mismos que en el script anterior para una comparación justa)
OPTIMAL_NN_PARAMS = {'learning_rate': 0.0001, 'dropout_rate': 0.2, 'batch_size': 16, 'epochs': 150}
OPTIMAL_LR_PARAMS = {'penalty': 'l2', 'C': 0.05, 'solver': 'saga', 'max_iter': 2000}
OPTIMAL_SVM_PARAMS = {'C': 0.05, 'kernel': 'rbf', 'gamma': 'scale', 'shrinking': True, 'max_iter':1000} 


OPTIMAL_XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 2000, 'learning_rate': 0.00001, 
                      'max_depth': 2, 'subsample': 0.05, 'colsample_bytree': 0.1, 'gamma': 0, 'min_child_weight': 0.6, 
                      'reg_alpha': 0.9}

OPTIMAL_RF_PARAMS = {'n_estimators': 500, 'criterion': 'entropy', 'max_depth': 1,  
                     'min_samples_split': 2, 'min_samples_leaf': 5, 'max_leaf_nodes': None, 'bootstrap': True, 
                     'max_samples': None, 'oob_score': True, 'ccp_alpha': 0.01, 'warm_start': True, 'verbose': 0}

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

#  Implementación de Random Under-Sampling (RUS) >>>
def apply_rus_on_train_data(X_train_scaled: pd.DataFrame, y_train: pd.Series, random_seed: int) -> (pd.DataFrame, pd.Series):
    """
    Aplica la técnica de submuestreo aleatorio (Random Under-Sampling)
    sobre el conjunto de datos de entrenamiento.
    """
    logging.info("Aplicando técnica de Submuestreo Aleatorio (RUS) sobre el conjunto de entrenamiento...")
    
    # Usar la semilla de la simulación para que el submuestreo sea reproducible en cada ciclo
    rus = RandomUnderSampler(random_state=random_seed)
    
    # fit_resample devuelve arrays de numpy, es buena práctica convertirlos de nuevo a pandas
    X_res, y_res = rus.fit_resample(X_train_scaled, y_train)
    
    X_res_df = pd.DataFrame(X_res, columns=X_train_scaled.columns)
    y_res_series = pd.Series(y_res, name=y_train.name)
    
    logging.info(f"RUS: Tamaño del conjunto de entrenamiento original: {len(X_train_scaled)}, nuevo tamaño: {len(X_res_df)}")
    logging.info(f"Distribución de clases después de RUS: {dict(y_res_series.value_counts())}")
    
    return X_res_df, y_res_series


# --- PREPARACIÓN DE DATOS ---
def prepare_data(df_original, target_col_name, random_seed):
    """
    Prepara los datos: 1. Split, 2. Scale, 3. Balance (con RUS).
    Este orden previene la fuga de datos (data leakage).
    """
    logging.info(">> Iniciando preparación de datos (Split -> Scale -> Balance)")
    X_original, y_original = df_original.drop(target_col_name, axis=1), df_original[target_col_name]
    
    # 1. Split del dataset original
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, stratify=y_original, random_state=random_seed)
    
    # 2. Escalado de datos
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
    
    # 3. Balanceo del conjunto de entrenamiento (AQUÍ SE USA LA NUEVA FUNCIÓN)
    X_train_final, y_train_final = apply_rus_on_train_data(X_train_scaled, y_train, random_seed)
    
    return X_train_final, X_test_scaled, y_train_final, y_test, scaler

# --- FUNCIONES ESPECÍFICAS DE CADA MODELO  ---

def train_nn_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo de Red Neuronal (NN)...")
    if X_train.empty: logging.error("X_train está vacío. No se puede entrenar el modelo NN."); return None

    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=random_seed)
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
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train, y_train, test_size=0.5, stratify=y_train, random_state=random_seed
    )
    current_xgb_params = OPTIMAL_XGB_PARAMS.copy()
    current_xgb_params['seed'] = random_seed
    model = xgb.XGBClassifier(**current_xgb_params)
    start_time = time.time()
    model.fit(
        X_train_new, y_train_new,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=25,
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

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT, exist_ok=True)
    all_results = []
    all_precision_results = []
    df_original = load_data(CSV_PATH)

    for i in range(1, N_SIMULATIONS + 1):
        # *** CAMBIO IMPORTANTE: Nomenclatura para RUS ***
        exp_name_sim = f"RUS_Compare_Sim_{i}"
        output_dir_sim = os.path.join(BASE_OUTPUT, exp_name_sim)
        log_file_sim = os.path.join(output_dir_sim, f"run_{exp_name_sim}.log")
        setup_logging(log_file_sim)
        
        print(f"\n{'='*25} INICIANDO SIMULACIÓN {i}/{N_SIMULATIONS} (TÉCNICA: RUS) {'='*25}")
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando Simulación Comparativa con RUS: {i}/{N_SIMULATIONS} ===")
        
        current_seed = i 
        np.random.seed(current_seed)
        tf.random.set_seed(current_seed)
        logging.info(f"Semillas (Numpy, TF, Split, Sklearn, XGB, RF) establecidas en: {current_seed}")
        
        X_train, X_test, y_train, y_test, _ = prepare_data(df_original, TARGET_COLUMN_NAME, random_seed=current_seed)
        
        # Procesamiento de todos los modelos
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
            logging.info(f"\n--- Procesando Modelo: {model_name} con datos de RUS ---")
            model_path = train_func(X_train.copy(), y_train.copy(), output_dir_sim, exp_name_sim, random_seed=current_seed)
            if model_path:
                recall, precision = eval_func(model_path, X_test, y_test)
                logging.info(f"Macro Avg Recall {model_name}: {recall:.4f} | Macro Avg Precision {model_name}: {precision:.4f}")
                # *** CAMBIO IMPORTANTE: Nomenclatura de columnas para RUS ***
                sim_results[f'recall_{model_name}_RUS'] = recall
                sim_precision_results[f'precision_{model_name}_RUS'] = precision
            else:
                sim_results[f'recall_{model_name}_RUS'] = None
                sim_precision_results[f'precision_{model_name}_RUS'] = None

        all_results.append(sim_results)
        all_precision_results.append(sim_precision_results)
        
        print(f"--- Simulación {i} completada. ---")
        logging.info(f"=== Fin Simulación {i} ===\n")

    print(f"\n{'='*25} TODAS LAS SIMULACIONES (RUS) COMPLETADAS {'='*25}")
    
    # Reporte de Recall
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\nResumen de Resultados de Macro Avg Recall (RUS):")
        print(results_df.to_string(index=False))
        
        stats_summary = "Estadísticas de Robustez del Macro Avg Recall (RUS)\n" + "="*60 + "\n"
        stats_summary += f"Total de Simulaciones: {len(results_df)}\n\n"

        for col in results_df.columns:
            if 'recall' in col:
                stats_summary += f"--- Estadísticas para {col} ---\n"
                stats_summary += f"  - Media:          {results_df[col].mean():.4f}\n"
                stats_summary += f"  - Desv. Estándar: {results_df[col].std():.4f}\n"
                stats_summary += f"  - Mínimo:         {results_df[col].min():.4f}\n"
                stats_summary += f"  - Máximo:         {results_df[col].max():.4f}\n\n"

        print("\n" + stats_summary)
        
        # *** CAMBIO IMPORTANTE: Nombres de archivo de salida para RUS ***
        summary_csv_path = os.path.join(BASE_OUTPUT, 'ALL_MODELS_recall_RUS_summary.csv')
        results_df.to_csv(summary_csv_path, index=False)
        print(f"Resultados de recall guardados en: {summary_csv_path}")
        
        stats_summary_path = os.path.join(BASE_OUTPUT, 'Statistics_summary.txt')
        with open(stats_summary_path, 'w') as f:
            f.write(stats_summary)
        print(f"Estadísticas de robustez guardadas en: {stats_summary_path}")

    # Reporte de Precisión
    if all_precision_results:
        precision_results_df = pd.DataFrame(all_precision_results)
        print("\nResumen de Resultados de Macro Avg Precision (RUS):")
        print(precision_results_df.to_string(index=False))
        
        # *** CAMBIO IMPORTANTE: Nombres de archivo de salida para RUS ***
        precision_summary_csv_path = os.path.join(BASE_OUTPUT, 'ALL_MODELS_precision_RUS_summary.csv')
        precision_results_df.to_csv(precision_summary_csv_path, index=False)
        print(f"\nResultados de precisión guardados en: {precision_summary_csv_path}")