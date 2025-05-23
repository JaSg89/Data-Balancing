import os
import sys
import time
import logging
import warnings
import joblib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import tensorflow as tf # Importado para tf.random.set_seed

from sklearn.model_selection import (
    train_test_split,
    # StratifiedKFold, # No se usa directamente, GridSearchCV lo maneja
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# --- CONFIGURACIÓN DE HILOS DE TENSORFLOW ---

try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    logging.info("TensorFlow inter-op parallelism threads set to 1.")
    logging.info("TensorFlow intra-op parallelism threads set to 1.")
except RuntimeError as e:
    # Esto puede ocurrir si los hilos ya fueron configurados (ej. en un entorno interactivo como Jupyter)
    logging.warning(f"Could not set TensorFlow threading: {e}. This might be okay if already configured.")

# -------------------------------------------------------------------
# Configuración 
# -------------------------------------------------------------------
warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)
LOG_FMT = "%(asctime)s %(levelname)-8s %(message)s"
N_SAMPLES_FEUS = 55000 # Número de muestras a conservar por FEUS

# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------
def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FMT,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_and_prepare_unscaled(csv_path): # Renombrado para claridad
    df = pd.read_csv(csv_path)
    if 'Clase' in df.columns and 'Class' not in df.columns:
        df = df.rename(columns={'Time':'Tiempo','Amount':'Cantidad','Clase':'Class'})
    elif 'Tiempo' in df.columns and 'Time' not in df.columns:
        df = df.rename(columns={'Tiempo':'Time','Cantidad':'Amount'})
    # NO ESCALA NADA AQUÍ
    return df

def build_nn_model(n_inputs, learning_rate=0.001, dropout_rate=0.5):
    model = Sequential([
        Dense(32, input_shape=(n_inputs,), activation='relu'), 
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------------------------------------------
# Funciones de resampling FEUS con escalado controlado para Mahalanobis
# Estas funciones devuelven datos en la escala que tenían al ENTRAR
# (o escalados si es Scen2 y entraron escalados)
# -------------------------------------------------------------------
def feus_apply_logic(data_features_input: pd.DataFrame, # Características del conjunto sobre el cual operar
                     n_samples_to_keep: int,
                     operation_context: str # "Scen1_Full" o "Scen2_Train"
                     ) -> pd.Index: # Devuelve los índices de data_features_input a conservar
    """
    Lógica central de FEUS.
    Escala INTERNAMENTE todas las características de 'data_features_input' para el cálculo de Mahalanobis.
    Selecciona las n_samples_to_keep más distantes globalmente DENTRO de data_features_input.
    Devuelve los índices de 'data_features_input' a conservar.
    """
    if data_features_input.empty or data_features_input.shape[1] == 0:
        logging.warning(f"FEUS {operation_context}: No hay características para calcular Mahalanobis. "
                        f"Devolviendo hasta {n_samples_to_keep} índices disponibles.")
        return data_features_input.head(min(n_samples_to_keep, len(data_features_input))).index

    # Escalado INTERNO de TODAS las características para el cálculo de Mahalanobis
    scaler_internal = MinMaxScaler()
    # data_features_input ya es SOLO las features
    X_scaled_for_mahalanobis = pd.DataFrame(scaler_internal.fit_transform(data_features_input),
                                            columns=data_features_input.columns,
                                            index=data_features_input.index)

    if X_scaled_for_mahalanobis.shape[0] < 2:
        logging.warning(f"FEUS {operation_context}: Menos de 2 muestras para calcular covarianza. "
                        f"Devolviendo hasta {n_samples_to_keep} índices disponibles.")
        return data_features_input.head(min(n_samples_to_keep, len(data_features_input))).index

    mean_vector = X_scaled_for_mahalanobis.mean().values
    
    if X_scaled_for_mahalanobis.shape[0] <= X_scaled_for_mahalanobis.shape[1]:
        logging.warning(f"FEUS {operation_context}: Covarianza podría ser singular (muestras={X_scaled_for_mahalanobis.shape[0]}, feats={X_scaled_for_mahalanobis.shape[1]}). Usando ddof=0.")
        cov_matrix = np.cov(X_scaled_for_mahalanobis.values, rowvar=False, ddof=0)
    else:
        cov_matrix = np.cov(X_scaled_for_mahalanobis.values, rowvar=False)

    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        logging.warning(f"FEUS {operation_context}: Matriz de covarianza singular, usando pseudo-inversa.")
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    diff_values = X_scaled_for_mahalanobis.values - mean_vector
    left = diff_values.dot(inv_cov_matrix)
    distances_sq = np.einsum('ij,ij->i', left, diff_values)
    distances = np.sqrt(np.maximum(distances_sq, 0))
    
    # Añadir distancias al DataFrame de características de ENTRADA (para usar sus índices)
    data_features_with_dist = data_features_input.copy() # Evitar SettingWithCopyWarning
    data_features_with_dist['Mahalanobis_Distance_Internal'] = distances
    
    selected_indices = data_features_with_dist.sort_values(
        by='Mahalanobis_Distance_Internal', ascending=False
    ).head(n_samples_to_keep).index
    
    return selected_indices


def feus_scenario1_apply(df_orig_unscaled: pd.DataFrame, target_col_name='Class', n_samples_to_keep=N_SAMPLES_FEUS) -> (pd.DataFrame, pd.Series):
    """
    Aplica FEUS al dataset completo (df_orig_unscaled, que NO está escalado para modelado).
    Escala internamente para Mahalanobis. Devuelve X_res, y_res en su escala original (unscaled).
    """
    df = df_orig_unscaled.copy()
    X_for_feus_unscaled = df.drop(columns=[target_col_name])
    # y_original = df[target_col_name] # No se usa y_original directamente aquí, solo para reconstruir

    if X_for_feus_unscaled.empty:
        logging.warning("FEUS Scen1: df_orig_unscaled sin características. Devolviendo truncado si es necesario.")
        df_resampled = df.head(min(n_samples_to_keep, len(df)))
        return df_resampled.drop(columns=[target_col_name]), df_resampled[target_col_name]

    selected_indices = feus_apply_logic(X_for_feus_unscaled, n_samples_to_keep, "Scen1_Full")
    
    # Seleccionar del DataFrame ORIGINAL (df_orig_unscaled)
    df_resampled = df_orig_unscaled.loc[selected_indices].reset_index(drop=True) # No es necesario sample(frac=1) aquí
    
    X_res_unscaled = df_resampled.drop(columns=[target_col_name])
    y_res = df_resampled[target_col_name]
    
    logging.info(f"FEUS Scen1: Resampled. Original: {df_orig_unscaled.shape[0]}, Seleccionado: {len(df_resampled)}, Objetivo: {n_samples_to_keep}")
    logging.info(f"   Distribución de clases en FEUS Scen1 (después de seleccionar {n_samples_to_keep} más lejanos): {dict(y_res.value_counts())}")
    return X_res_unscaled, y_res


def feus_scenario2_apply(X_train_input_scaled: pd.DataFrame, # X_train YA ESCALADO para modelado
                         y_train_orig: pd.Series,
                         n_samples_to_keep=N_SAMPLES_FEUS) -> (pd.DataFrame, pd.Series):

    X_train_current_scaled = X_train_input_scaled.copy() # Datos ya escalados para modelado
    y_train = y_train_orig.copy()

    if X_train_current_scaled.empty:
        logging.warning("FEUS Scen2: X_train_input_scaled vacío. Devolviendo truncado si es necesario.")
        num_to_keep_actual = min(n_samples_to_keep, len(X_train_current_scaled)) # len puede ser 0
        return X_train_current_scaled.head(num_to_keep_actual), y_train.head(num_to_keep_actual)

    selected_indices = feus_apply_logic(X_train_current_scaled, n_samples_to_keep, "Scen2_Train")
    
    # Seleccionar de los datos de entrada (que ya están escalados)
    X_res_scaled = X_train_current_scaled.loc[selected_indices].reset_index(drop=True)
    y_res = y_train.loc[selected_indices].reset_index(drop=True)

    logging.info(f"FEUS Scen2: Resampled train set. Original: {X_train_input_scaled.shape[0]}, Seleccionado: {len(X_res_scaled)}, Objetivo: {n_samples_to_keep}")
    logging.info(f"   Distribución de clases en y_train después de FEUS Scen2: {dict(y_res.value_counts())}")
    return X_res_scaled, y_res

# -------------------------------------------------------------------
# Función principal de escenario 
# -------------------------------------------------------------------
def run_scenario_feus_controlled_scaling(df_original_unscaled, scenario, target_col_name='Class'):
    X_original_unscaled = df_original_unscaled.drop(target_col_name, axis=1)
    y_original = df_original_unscaled[target_col_name]
    
    X_train_final, X_test_final, y_train_final, y_test_final = [None]*4 

    logging.info(f"Usando FEUS (escalado interno para Mahalanobis, escalado de modelado controlado por escenario).")

    if scenario == 'scenario1':
        logging.info(f">> ESCENARIO 1: FEUS en TODO el dataset (sin escalar) -> Split -> Escalado Separado")
        X_res_unscaled, y_res = feus_scenario1_apply(df_original_unscaled.copy(), 
                                                     target_col_name=target_col_name, 
                                                     n_samples_to_keep=N_SAMPLES_FEUS)
        logging.info(f"   FEUS Scen1: Después de resample (antes de split y escalar): {X_res_unscaled.shape}, Distribución y_res: {dict(y_res.value_counts())}")

        if X_res_unscaled.empty or y_res.empty:
            logging.error("   FEUS Scen1: X_res_unscaled o y_res vacíos después del balanceo. Abortando escenario.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

        stratify_param_s1 = y_res if not y_res.empty and len(y_res.unique()) > 1 else None
        X_train_raw, X_test_raw, y_train_final, y_test_final = train_test_split(
            X_res_unscaled, y_res, test_size=0.2, stratify=stratify_param_s1, random_state=42
        )
        logging.info(f"   FEUS Scen1: Después de Split. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

        if not X_train_raw.empty:
            scaler_s1 = MinMaxScaler()
            X_train_final = pd.DataFrame(scaler_s1.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            if not X_test_raw.empty:
                X_test_final = pd.DataFrame(scaler_s1.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
            else: X_test_final = pd.DataFrame()
            logging.info(f"   FEUS Scen1: Después de Escalado. X_train_final: {X_train_final.shape}, X_test_final: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
        else:
            X_train_final = pd.DataFrame()
            X_test_final = pd.DataFrame()

    else: # scenario2
        logging.info(f">> ESCENARIO 2: Split del dataset (sin escalar) -> Escalado Separado -> FEUS (solo en train escalado)")
        stratify_param_s2_initial = y_original if not y_original.empty and len(y_original.unique()) > 1 else None
        X_train_raw, X_test_raw, y_train_orig, y_test_final = train_test_split(
            X_original_unscaled, y_original, test_size=0.2, stratify=stratify_param_s2_initial, random_state=42
        )
        logging.info(f"   FEUS Scen2: Después de Split inicial. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

        X_train_scaled_for_feus = pd.DataFrame()
        X_test_final = pd.DataFrame() 

        if not X_train_raw.empty:
            scaler_s2 = MinMaxScaler()
            X_train_scaled_for_feus = pd.DataFrame(scaler_s2.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            if not X_test_raw.empty:
                X_test_final = pd.DataFrame(scaler_s2.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
            logging.info(f"   FEUS Scen2: Después de Escalado. X_train_scaled_for_feus: {X_train_scaled_for_feus.shape}, X_test_final: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
        
        if not X_train_scaled_for_feus.empty and not y_train_orig.empty:
            X_train_final, y_train_final = feus_scenario2_apply(X_train_scaled_for_feus, y_train_orig,
                                                                n_samples_to_keep=N_SAMPLES_FEUS)
            logging.info(f"   FEUS Scen2: Después de FEUS en train. X_train_final: {X_train_final.shape}, y_train_final dist: {dict(y_train_final.value_counts() if not y_train_final.empty else {})}")
        else:
            X_train_final = X_train_scaled_for_feus
            y_train_final = y_train_orig

    logging.info(f"   → Tamaños finales para modelado. Train: {X_train_final.shape if not X_train_final.empty else '(empty)'}, Test: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
    if not (y_train_final is None or y_train_final.empty):
        logging.info(f"   → Distribución y_train_final: {dict(y_train_final.value_counts())}")
    else:
        logging.warning("   → y_train_final está vacío o es None.")
    if not (y_test_final is None or y_test_final.empty):
        logging.info(f"   → Distribución y_test_final: {dict(y_test_final.value_counts())}")
    else:
        logging.warning("   → y_test_final está vacío o es None.")
        
    return X_train_final, X_test_final, y_train_final, y_test_final

# -------------------------------------------------------------------
# Funciones de entrenamiento y evaluación 
# -------------------------------------------------------------------

def train_and_save_models(X_train, y_train, exp_name, output_dir):
    hyper_file = os.path.join(output_dir, 'hyperparameters.txt')
    with open(hyper_file, 'w') as hf:
        hf.write(f"Hyperparameters for experiment {exp_name}\n")
        hf.write("="*60 + "\n\n")
    print(f"> Hyperparameters will be saved to: {hyper_file}")

    if X_train.empty or X_train.shape[1] == 0:
        logging.error(f"X_train está vacío o no tiene características ANTES de entrenar modelos para {exp_name}. Saltando entrenamiento.")
        return {}

    n_inputs = X_train.shape[1]

    nn_wrapper = KerasClassifier(
        build_fn=build_nn_model,
        n_inputs=n_inputs,
        verbose=0,
    )

    specs = { 

        'nn': (
            nn_wrapper,
            {
                'clf__learning_rate': [0.0001],
                'clf__dropout_rate':  [0.05, 0.1, 0.3],
                'clf__batch_size':    [16, 32, 64],
                'clf__epochs':        [100], 
                # No 'clf__validation_split' aquí si se maneja en fit_params o se omite
            }
        ),

        'logreg': (
            LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            {'clf__penalty':['l1','l2'], 'clf__C':[0.01, 0.1, 1, 10, 100], 'clf__solver':['liblinear']}
        ),
        'svm': (
            SVC(probability=True, random_state=42, class_weight='balanced'),
            {'clf__C': [0.5, 1, 5, 10], 'clf__kernel': ['rbf', 'linear', 'poly']}
        ),
        'rf': (
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            {'clf__n_estimators':[100, 150, 200], 'clf__max_depth':[10, 20, 30, None], 
             'clf__min_samples_split': [2, 5, 10], 'clf__min_samples_leaf': [1, 2, 4]}
        ),
        'xgb': (
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
             {
                'clf__n_estimators': [100, 150, 200], 'clf__max_depth': [3, 5, 7, 10],
                'clf__learning_rate': [0.01, 0.05, 0.1, 0.2], 'clf__subsample': [0.7, 0.8, 0.9]
            }
        )
    }

    best_models = {}
    for name, (clf, param_grid) in specs.items():
        print(f"\n> Entrenando {name.upper()}...")
        logging.info(f"Entrenando {name.upper()}...")
        
        pipe = ImbPipeline([ 
            ('scaler', MinMaxScaler()), 
            ('clf', clf)
        ])
        
        y_train_processed = y_train.astype(int) 

        fit_params = {}
        if name == 'nn':
            # Opción recomendada: Sin validation_split interno para Keras cuando se usa GridSearchCV
            # Los callbacks monitorearán 'loss'
            fit_params['clf__callbacks'] = [
                ReduceLROnPlateau(monitor='loss', patience=5, factor=0.2, min_lr=1e-6, verbose=0), 
                EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0) 
            ]
            # NO fit_params['clf__validation_split'] = 0.1

        # --- INICIO: Lógica para determinar n_cv_splits y crear cv_obj ---
        current_n_cv_splits = 4 # Valor base que tenías, o el que desees (e.g., 5)
        
        if len(y_train_processed.unique()) < 2:
            logging.error(f"Solo una clase en y_train para {name}. No se puede hacer CV. Saltando GridSearchCV.")
            # ... (código para entrenar modelo por defecto o simplemente saltar) ...
            best_params_str = "CV SKIPPED (una clase)"
            best_score_str = "N/A"
            best_estimator_for_model = None # Marcar para no guardar
            # (continúa al final del bucle para guardar estado y pasar al siguiente modelo)
        else:
            min_class_count = min(y_train_processed.value_counts())
            # n_splits no puede ser mayor que el número de miembros en cualquier clase.
            actual_n_splits = min(current_n_cv_splits, min_class_count)

            if actual_n_splits < 2:
                logging.error(f"No se puede realizar CV para {name} con n_splits={actual_n_splits} (basado en min_class_count). Saltando GridSearchCV.")
                best_params_str = f"CV SKIPPED (n_splits={actual_n_splits})"
                best_score_str = "N/A"
                best_estimator_for_model = None
            elif X_train.shape[0] < actual_n_splits:
                logging.error(f"No hay suficientes muestras en X_train ({X_train.shape[0]}) para CV con {actual_n_splits} splits en {name}. Saltando GridSearchCV.")
                best_params_str = f"CV SKIPPED (X_train pequeño, n_splits={actual_n_splits})"
                best_score_str = "N/A"
                best_estimator_for_model = None
            else:
                logging.info(f"Usando StratifiedKFold con n_splits={actual_n_splits}, shuffle=True para {name}.")
                cv_obj = StratifiedKFold(n_splits=actual_n_splits, shuffle=True, random_state=42)
                # --- FIN: Lógica para determinar n_cv_splits y crear cv_obj ---

                grid = GridSearchCV(
                    pipe, param_grid, 
                    cv=cv_obj, 
                    scoring='roc_auc', 
                    n_jobs=5, verbose=1, refit=True, error_score='raise'
                )
                
                best_estimator_for_model = None # Inicializar
                try:
                    if name == 'nn':
                        grid.fit(X_train, y_train_processed, **fit_params)
                    else:
                        grid.fit(X_train, y_train_processed)
                    
                    best_estimator_for_model = grid.best_estimator_
                    best_params_str = str(grid.best_params_)
                    best_score_str = f"{grid.best_score_:.4f}"

                except ValueError as ve:
                    logging.error(f"Error de ValueError (posiblemente CV o scorer) durante GridSearchCV para {name} en {exp_name}: {ve}", exc_info=True)
                    best_params_str = "Error en CV/Scorer"
                    best_score_str = "Error"
                except Exception as e:
                    logging.error(f"Error general durante GridSearchCV para {name} en {exp_name}: {e}", exc_info=True)
                    best_params_str = "Error general"
                    best_score_str = "Error"
        
        # Guardado de modelos y logs (como lo tienes)
        if best_estimator_for_model:
            # ... (tu lógica de guardado)
            if name == 'nn':
                if hasattr(best_estimator_for_model.named_steps['clf'], 'model'):
                    scaler_nn = best_estimator_for_model.named_steps['scaler']
                    keras_model_nn = best_estimator_for_model.named_steps['clf'].model
                    scaler_path = os.path.join(output_dir, f"{exp_name}_{name}_scaler.joblib")
                    keras_model_path = os.path.join(output_dir, f"{exp_name}_{name}_keras_model.h5")
                    joblib.dump(scaler_nn, scaler_path)
                    keras_model_nn.save(keras_model_path)
                    logging.info(f"  ✔ Scaler NN guardado: {scaler_path}")
                    logging.info(f"  ✔ Modelo Keras NN guardado: {keras_model_path}")
                    best_models[name] = (scaler_path, keras_model_path)
                else:
                    logging.error(f"  ✘ No se pudo extraer el modelo Keras para {name}.")
                    best_models[name] = None 
            else:
                model_path = os.path.join(output_dir, f"{exp_name}_{name}_best_pipeline.joblib") # Cambiado de _best.joblib
                joblib.dump(best_estimator_for_model, model_path)
                logging.info(f"  ✔ Modelo (pipeline) guardado: {model_path}")
                best_models[name] = best_estimator_for_model
        else: # Si best_estimator_for_model es None (por error o salto de CV)
            best_models[name] = None
            if not (best_params_str.startswith("CV SKIPPED") or best_params_str.startswith("Error")): # Evitar doble log si ya se registró error
                 logging.warning(f"No se pudo obtener best_estimator para {name} debido a un error previo no capturado por salto de CV.")


        with open(hyper_file, 'a') as hf:
            hf.write(f"{name.upper()} best params: {best_params_str}\n")
            hf.write(f"{name.upper()} best CV ROC-AUC: {best_score_str}\n\n") # O la métrica que uses
        print(f"  ✔ Hiperparámetros (o estado de error) guardados para {name.upper()}")
        
    return {k: v for k, v in best_models.items() if v is not None}

def evaluate_and_save_reports(models, X_test, y_test, output_dir):


    report_file = os.path.join(output_dir, "classification_reports.txt")
    with open(report_file, "w") as rf:
        rf.write(f"Classification reports for {os.path.basename(output_dir)}\n")
        rf.write("=" * 60 + "\n\n")
    print(f"\n> Reports will be saved to: {report_file}")

    if X_test.empty or y_test.empty:
        logging.error("X_test o y_test están vacíos. No se pueden generar reportes.")
        print("X_test o y_test están vacíos. Saltando evaluación.")
        return

    y_test_processed = y_test.astype(int)

    for name, model_or_paths in models.items():
        print(f"\n> Evaluando {name.upper()}...")
        logging.info(f"Evaluando {name.upper()}...")
        y_pred, y_prob = None, None 

        try:
            if name == "nn":
                if isinstance(model_or_paths, tuple) and len(model_or_paths) == 2: # NN models
                    scaler_path, keras_model_path = model_or_paths
                    scaler_loaded = joblib.load(scaler_path) # Este es el scaler del pipeline del NN
                    keras_model = load_model(keras_model_path, compile=False)
                    
                    # X_test ya está escalado por run_scenario_...
                    # Aplicar el scaler_loaded es para consistencia con cómo se entrenó
                    # (especialmente si hubo validation_split interno en Keras)
                    X_test_for_eval = scaler_loaded.transform(X_test) 
                    
                    y_prob_all_classes = keras_model.predict(X_test_for_eval)
                    if y_prob_all_classes.shape[1] == 2:
                        y_prob = y_prob_all_classes[:, 1]
                        y_pred = np.argmax(y_prob_all_classes, axis=1)
                    elif y_prob_all_classes.shape[1] == 1:
                         logging.warning(f"Modelo NN predijo una sola columna de probabilidad: {y_prob_all_classes.shape}")
                         y_prob = y_prob_all_classes[:,0]
                         y_pred = (y_prob > 0.5).astype(int)
                    else:
                        logging.error(f"Forma inesperada de y_prob_all_classes para NN: {y_prob_all_classes.shape}")
                        continue
                else: # Fallback si model_or_paths no es la tupla esperada
                    logging.error(f"  ✘ Rutas de modelo NN no válidas para {name}. Tipo: {type(model_or_paths)}")
                    continue
            else: # Otros modelos (pipeline completo)
                model_pipeline = model_or_paths
                # El pipeline (scaler + clf) se encarga del escalado de X_test
                y_prob_all_classes = model_pipeline.predict_proba(X_test) 
                y_pred = model_pipeline.predict(X_test)
                
                if y_prob_all_classes.shape[1] == 2:
                    y_prob = y_prob_all_classes[:, 1]
                elif y_prob_all_classes.shape[1] == 1:
                    logging.warning(f"Modelo {name} predijo una sola columna de probabilidad. Clases del clasificador: {model_pipeline.named_steps['clf'].classes_}")
                    if model_pipeline.named_steps['clf'].classes_[0] == 1:
                        y_prob = y_prob_all_classes[:, 0] 
                    else:
                        y_prob = 1.0 - y_prob_all_classes[:, 0]
                else: 
                    logging.error(f"Forma inesperada de y_prob_all_classes para {name}: {y_prob_all_classes.shape}")
                    continue

            auc = float('nan')
            report_str = "N/A"

            if y_pred is not None and y_prob is not None:
                if len(np.unique(y_test_processed)) < 2 :
                    logging.warning(f"Solo una clase presente en y_test para {name}. ROC-AUC no es calculable.")
                elif len(y_prob) == 0 :
                    logging.warning(f"y_prob está vacío para {name}. ROC-AUC no es calculable.")
                else:
                    try:
                        auc = roc_auc_score(y_test_processed, y_prob)
                        if len(np.unique(y_pred)) < 2 and len(np.unique(y_test_processed)) >=2 :
                            logging.warning(f"Todas las predicciones son de una sola clase para {name}, pero y_test tiene variabilidad. ROC-AUC: {auc:.4f}")
                    except ValueError as e_auc:
                        logging.error(f"Error al calcular ROC-AUC para {name}: {e_auc}. y_test unique: {np.unique(y_test_processed)}, y_prob (first 5 unique): {np.unique(y_prob[:5]) if len(y_prob)>0 else 'empty'}")
                
                report_str = classification_report(
                    y_test_processed, y_pred,
                    target_names=["No Fraude (0)", "Fraude (1)"],
                    digits=4, zero_division=0
                )
            else:
                logging.error(f"Predicciones (y_pred o y_prob) no generadas para {name}.")

            print(f"{name.upper()} ROC-AUC: {auc:.4f}\n{report_str}")
            with open(report_file, "a") as rf:
                rf.write(f"{name.upper()} ROC-AUC: {auc:.4f}\n{report_str}\n{'-'*60}\n")
            print(f"  ✔ Reporte guardado para {name.upper()}")
        
        except Exception as e_eval:
            logging.error(f"Error durante la evaluación de {name} en {output_dir}: {e_eval}", exc_info=True)
            print(f"Error durante la evaluación de {name}: {e_eval}")
            with open(report_file, "a") as rf:
                rf.write(f"{name.upper()} - ERROR EN EVALUACIÓN: {e_eval}\n{'-'*60}\n")

# -------------------------------------------------------------------
# Bloque principal
# -------------------------------------------------------------------

if __name__ == "__main__":
    CSV_PATH = "/scratch/sivar/jarevalo/jsaavedra/creditcard.csv"
    BASE_OUTPUT = "/scratch/sivar/jarevalo/jsaavedra/resultados_FEUS" 
    TARGET_COLUMN_NAME = 'Class'

    for scenario in ["scenario1", "scenario2"]:
        exp_name   = f"FEUS_GlobalSelection_Scaling{N_SAMPLES_FEUS}_{scenario}"
        output_dir = os.path.join(BASE_OUTPUT, exp_name)
        log_file   = os.path.join(output_dir, f"run_{exp_name}.log")

        os.makedirs(output_dir, exist_ok=True)
        setup_logging(log_file)
        
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando experimento: {exp_name} ===")
        logging.info(f"Técnica: FEUS (Selección Global con Escalado Controlado), N samples: {N_SAMPLES_FEUS}")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"=================================================")
        print(f"\n=== Iniciando experimento: {exp_name} ===")

        df_original = load_and_prepare_unscaled(CSV_PATH) # Carga datos SIN escalar
        logging.info(f"Dataset cargado (sin escalar). Forma: {df_original.shape}. Clases: {dict(df_original[TARGET_COLUMN_NAME].value_counts())}")

        if df_original.shape[0] < 10: 
            logging.error(f"Dataset con muy pocas filas ({df_original.shape[0]}). Abortando {exp_name}.")
            continue
        class_counts = df_original[TARGET_COLUMN_NAME].value_counts()
        if len(class_counts) < 1 : # Permitir si solo hay una clase, FEUS podría manejarlo
            logging.warning(f"Dataset no tiene clases o está vacío: {class_counts.to_dict()}.")
        elif class_counts.get(0,0) == 0 and class_counts.get(1,0) == 0:
             logging.warning(f"Dataset no tiene muestras en ninguna clase.")
        elif len(class_counts) < 2: # Si solo hay una clase, pero tiene muestras
             logging.warning(f"Dataset tiene solo una clase: {class_counts.to_dict()}. El modelado podría fallar.")


        X_train, X_test, y_train, y_test = run_scenario_feus_controlled_scaling(
            df_original.copy(), scenario, target_col_name=TARGET_COLUMN_NAME
        )

        if X_train.empty or y_train.empty:
            logging.error(f"X_train o y_train vacíos para {exp_name} DESPUÉS de run_scenario. Saltando entrenamiento.")
            continue
        
        if len(y_train.value_counts()) < 1:
            logging.warning(f"y_train no tiene muestras para {exp_name}. El entrenamiento podría fallar.")
        elif len(y_train.value_counts()) < 2:
             logging.warning(f"y_train tiene solo una clase para {exp_name}. GridSearchCV se adaptará o podría fallar.")

        start_time = time.time()
        trained_models = train_and_save_models(X_train, y_train, exp_name, output_dir)
        logging.info(f"Entrenamiento para {exp_name} completado en {time.time() - start_time:.2f}s")

        if not trained_models:
            logging.warning(f"No se entrenaron modelos exitosamente para {exp_name}. Saltando evaluación.")
        else:
            if X_test is not None and not X_test.empty and y_test is not None and not y_test.empty:
                 evaluate_and_save_reports(trained_models, X_test, y_test, output_dir)
            else:
                logging.warning(f"X_test o y_test están vacíos para {exp_name} después de run_scenario. Saltando evaluación.")
        
        logging.info(f"=== Fin experimento {exp_name} ===\n")
        print(f"=== Fin experimento: {exp_name} ===\n")

    print("Todos los experimentos FEUS con escalado controlado han finalizado.")