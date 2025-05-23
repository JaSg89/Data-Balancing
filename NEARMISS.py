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
    

# ... (Configuración global, create_nn_model, setup_logging, load_and_prepare sin cambios) ...
# -------------------------------------------------------------------
# Configuración global
# -------------------------------------------------------------------
warnings.filterwarnings("ignore")
np.random.seed(42)
# tf.random.set_seed(42) 
LOG_FMT = "%(asctime)s %(levelname)-8s %(message)s"

# -------------------------------------------------------------------
# Constructor de la red neuronal (picklable)
# -------------------------------------------------------------------
def create_nn_model(n_inputs, learning_rate=0.001, dropout_rate=0.5):
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

def load_and_prepare(csv_path): # YA NO ESCALA INTERNAMENTE
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'Time':'Tiempo','Amount':'Cantidad','Class':'Clase'})
    return df
# -------------------------------------------------------------------
# Función principal de escenario (PARA NEARMISS con escalado controlado y tu idea para Scen1)
# -------------------------------------------------------------------
def run_scenario_nearmiss_refined_scaling(df_original_unscaled, scenario):
    target_col_name = 'Clase'
    # Asegurarse de que X_original_unscaled es un DataFrame y y_original es una Serie
    if not isinstance(df_original_unscaled, pd.DataFrame):
        raise TypeError("df_original_unscaled debe ser un DataFrame de Pandas.")
    
    X_original_unscaled = df_original_unscaled.drop(target_col_name, axis=1, errors='ignore')
    if target_col_name not in df_original_unscaled.columns:
        raise ValueError(f"La columna objetivo '{target_col_name}' no se encuentra en df_original_unscaled.")
    y_original = df_original_unscaled[target_col_name]


    # Configuración de NearMiss
    nearmiss_sampler = NearMiss(sampling_strategy='majority', version=1, n_neighbors=3)
    logging.info(f"Usando NearMiss(version={nearmiss_sampler.version}, n_neighbors={nearmiss_sampler.n_neighbors}) con escalado refinado.")

    X_train_final, X_test_final, y_train_final, y_test_final = [pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')] # Inicializar como vacíos

    if scenario == 'scenario1':
        # Escenario 1: Fuga de Datos SOLO en la Selección de Muestras por NearMiss
        # 1. Cargar Datos (hecho) -> X_original_unscaled, y_original
        logging.info(f">> ESCENARIO 1: Escalado Global TEMPORAL -> NearMiss -> Selección de ÍNDICES del original -> Split -> Escalado Separado LIMPIO para modelado")

        # 2. Crear Copia para NearMiss y Escalarla Globalmente (Temporal)
        if X_original_unscaled.empty:
            logging.error("   NearMiss Scen1: X_original_unscaled está vacío. Abortando escenario.")
            return X_train_final, X_test_final, y_train_final, y_test_final
            
        scaler_temporal_s1 = MinMaxScaler()
        X_temp_scaled_for_nearmiss = pd.DataFrame(
            scaler_temporal_s1.fit_transform(X_original_unscaled), # .copy() implícito si es necesario
            columns=X_original_unscaled.columns,
            index=X_original_unscaled.index # Mantener índice original es CLAVE
        )
        logging.info(f"   NearMiss Scen1: Datos escalados temporalmente para NearMiss. Forma: {X_temp_scaled_for_nearmiss.shape}")

        # 3. Balanceo con NearMiss (Sobre Datos Globalmente Escalados Temporalmente)
        #    para obtener los ÍNDICES de las muestras seleccionadas.
        X_res_unscaled = pd.DataFrame() # Inicializar como DataFrames vacíos
        y_res = pd.Series(dtype='float64')

        try:
            # Comprobación de suficientes muestras para NearMiss
            # NearMiss puede necesitar al menos n_neighbors muestras en la clase minoritaria (implícitamente) y en la mayoría.
            # y_original.value_counts() nos da las cuentas de cada clase.
            class_counts = y_original.value_counts()
            minority_class_count = class_counts.min() if not class_counts.empty else 0
  
            can_run_nearmiss = True
            if X_temp_scaled_for_nearmiss.empty or y_original.empty:
                can_run_nearmiss = False
            elif nearmiss_sampler.version in [1, 2]: # Selecciona de la clase mayoritaria
                 if class_counts.get(0, 0) < nearmiss_sampler.n_neighbors : # Asumiendo 0 es mayoría
                     can_run_nearmiss = False
                     logging.warning(f"   NearMiss Scen1: No hay suficientes muestras en la clase MAYORITARIA ({class_counts.get(0,0)}) para NearMiss v{nearmiss_sampler.version} con n_neighbors={nearmiss_sampler.n_neighbors}.")
            elif nearmiss_sampler.version == 3: # Selecciona de la clase minoritaria
                 if minority_class_count < nearmiss_sampler.n_neighbors:
                     can_run_nearmiss = False
                     logging.warning(f"   NearMiss Scen1: No hay suficientes muestras en la clase MINORITARIA ({minority_class_count}) para NearMiss v3 con n_neighbors={nearmiss_sampler.n_neighbors}.")


            if not can_run_nearmiss:
                logging.warning("   NearMiss Scen1: No hay suficientes datos/clases para NearMiss. Usando datos originales sin balanceo.")
                X_res_unscaled = X_original_unscaled.copy()
                y_res = y_original.copy()
            else:
                # Fitear NearMiss
                nearmiss_sampler.fit_resample(X_temp_scaled_for_nearmiss, y_original)
                # Obtener los índices de las muestras seleccionadas
                selected_indices = nearmiss_sampler.sample_indices_
                
                # 4. Seleccionar Muestras del DataFrame Original NO ESCALADO usando los Índices
                X_res_unscaled = X_original_unscaled.iloc[selected_indices].reset_index(drop=True)
                y_res = y_original.iloc[selected_indices].reset_index(drop=True)
                logging.info(f"   NearMiss Scen1: Datos seleccionados del original (sin escalar) usando índices de NearMiss. "
                             f"Forma X_res_unscaled: {X_res_unscaled.shape}, Distribución y_res: {dict(y_res.value_counts())}")
        except ValueError as e:
            logging.error(f"   NearMiss Scen1: Error durante NearMiss: {e}. Usando datos originales sin balanceo.")
            X_res_unscaled = X_original_unscaled.copy() # Fallback
            y_res = y_original.copy()

        if X_res_unscaled.empty or y_res.empty:
            logging.error("   NearMiss Scen1: X_res_unscaled o y_res vacíos después del intento de balanceo. Abortando escenario.")
            return X_train_final, X_test_final, y_train_final, y_test_final # Devuelve vacíos

        # 5. Split (División del Conjunto Balanceado NO ESCALADO)
        stratify_param_s1 = y_res if not y_res.empty and len(y_res.unique()) > 1 else None
        X_train_raw, X_test_raw, y_train_final, y_test_final = train_test_split(
            X_res_unscaled, y_res, test_size=0.2, stratify=stratify_param_s1, random_state=42, shuffle=True
        )
        logging.info(f"   NearMiss Scen1: Después de Split. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

        # 6. Escalado Correcto para Modelado (SIN FUGA entre train/test de modelado)
        if not X_train_raw.empty:
            scaler_s1_modelado = MinMaxScaler()
            X_train_final = pd.DataFrame(scaler_s1_modelado.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            if not X_test_raw.empty:
                X_test_final = pd.DataFrame(scaler_s1_modelado.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
            logging.info(f"   NearMiss Scen1: Después de Escalado LIMPIO para modelado. X_train_final: {X_train_final.shape}, X_test_final: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
        # Si X_train_raw es vacío, X_train_final y X_test_final permanecen vacíos

    else: # scenario2 (Sin Fuga de Datos en Escalado ni Balanceo) - Lógica como antes
        # 1. Split del dataset ORIGINAL (df_original_unscaled)
        logging.info(f">> ESCENARIO 2: Split del dataset (sin escalar) -> Escalado Separado LIMPIO -> NearMiss (solo en train escalado)")
        stratify_param_s2_initial = y_original if not y_original.empty and len(y_original.unique()) > 1 else None
        X_train_raw, X_test_raw, y_train_orig, y_test_final_orig = train_test_split(
            X_original_unscaled, y_original, test_size=0.2, stratify=stratify_param_s2_initial, random_state=42, shuffle=True
        )
        logging.info(f"   NearMiss Scen2: Después de Split inicial. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

        # 2. Escalado SEPARADO y LIMPIO para modelado (y para NearMiss en train)
        X_train_scaled_for_nearmiss = pd.DataFrame() # Inicializar
        # X_test_final ya está inicializado como vacío arriba
        y_test_final = y_test_final_orig # y_test no cambia por el escalado de X

        if not X_train_raw.empty:
            scaler_s2_modelado = MinMaxScaler()
            X_train_scaled_for_nearmiss = pd.DataFrame(scaler_s2_modelado.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            if not X_test_raw.empty:
                X_test_final = pd.DataFrame(scaler_s2_modelado.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
            logging.info(f"   NearMiss Scen2: Después de Escalado LIMPIO. X_train_scaled_for_nearmiss: {X_train_scaled_for_nearmiss.shape}, X_test_final: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
        
        # 3. NearMiss solo en train (que ya está escalado para modelado)
        y_train_final = y_train_orig # Inicializar y_train_final
        X_train_final = X_train_scaled_for_nearmiss.copy() # Inicializar X_train_final

        if not X_train_scaled_for_nearmiss.empty and not y_train_orig.empty:
            try:
                # Comprobación de suficientes muestras para NearMiss en train set
                can_run_nearmiss_train = True
                class_counts_train = y_train_orig.value_counts()
                minority_class_count_train = class_counts_train.min() if not class_counts_train.empty else 0

                if nearmiss_sampler.version in [1, 2]:
                     if class_counts_train.get(0,0) < nearmiss_sampler.n_neighbors: # Asume 0 es mayoría
                         can_run_nearmiss_train = False
                         logging.warning(f"   NearMiss Scen2: No hay suficientes muestras en la clase MAYORITARIA del train ({class_counts_train.get(0,0)}) para NearMiss v{nearmiss_sampler.version} con n_neighbors={nearmiss_sampler.n_neighbors}.")
                elif nearmiss_sampler.version == 3:
                     if minority_class_count_train < nearmiss_sampler.n_neighbors:
                         can_run_nearmiss_train = False
                         logging.warning(f"   NearMiss Scen2: No hay suficientes muestras en la clase MINORITARIA del train ({minority_class_count_train}) para NearMiss v3 con n_neighbors={nearmiss_sampler.n_neighbors}.")

                if not can_run_nearmiss_train:
                     logging.warning(f"   NearMiss Scen2: No hay suficientes datos/clases en train para NearMiss. Usando train escalado sin balanceo.")
                     # X_train_final y y_train_final ya están seteados a los datos pre-balanceo
                else:
                    # NearMiss opera sobre X_train_scaled_for_nearmiss
                    # Devuelve arrays numpy, hay que reconvertir
                    X_res_np_train, y_res_np_train = nearmiss_sampler.fit_resample(X_train_scaled_for_nearmiss.to_numpy(), y_train_orig.to_numpy())
                    X_train_final = pd.DataFrame(X_res_np_train, columns=X_train_scaled_for_nearmiss.columns)
                    y_train_final = pd.Series(y_res_np_train, name=y_train_orig.name)
                logging.info(f"   NearMiss Scen2: Después de NearMiss en train. X_train_final (escalado): {X_train_final.shape}, y_train_final dist: {dict(y_train_final.value_counts() if not y_train_final.empty else {})}")
            except ValueError as e:
                logging.error(f"   NearMiss Scen2: Error durante NearMiss.fit_resample en train: {e}. Usando train escalado sin balanceo.")
                # X_train_final y y_train_final ya están seteados a los datos pre-balanceo
        # Si X_train_scaled_for_nearmiss o y_train_orig son vacíos, X_train_final y y_train_final permanecen con sus valores iniciales.

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
# Funciones de entrenamiento y evaluación (Prácticamente sin cambios)
# -------------------------------------------------------------------
def train_and_save_models(X_train, y_train, exp_name, output_dir):
    hyper_file = os.path.join(output_dir, 'hyperparameters.txt')
    with open(hyper_file, 'w') as hf:
        hf.write(f"Hyperparameters for {exp_name}\n")
        hf.write("="*60 + "\n\n")
    print(f"> Hyperparameters file: {hyper_file}")

    if X_train.empty or X_train.shape[1] == 0:
        logging.error(f"X_train está vacío o no tiene características ANTES de entrenar modelos para {exp_name}. Saltando entrenamiento.")
        return {}

    n_inputs = X_train.shape[1]
    nn_wrapper = KerasClassifier(
        build_fn=create_nn_model, 
        n_inputs=n_inputs,
        verbose=1, 
    )

    specs = {
        'nn': (
            nn_wrapper,
            {
                'clf__learning_rate':    [0.001], 
                'clf__dropout_rate':     [0.3, 0.5],
                'clf__batch_size':       [32, 64],
                'clf__epochs':           [50],
                'clf__validation_split': [0.1] 
            }
        ),

        'logreg': (
            LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'), 
            {'clf__penalty': ['l1','l2'], 'clf__C': [0.001, 0.01, 0.7, 0.1, 0.2, 1, 10, 100, 1000], 'clf__solver': ['liblinear']}
        ),
        'svm': (
            SVC(probability=True, random_state=42, class_weight='balanced'),
            {'clf__C': [0.5, 0.7, 0.9, 1, 1.5], 'clf__kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
        ),
        'rf': (
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 10, 20, 30], 'clf__min_samples_split': [2, 5, 10], 'clf__min_samples_leaf': [1, 2, 4]}
        ),
        'xgb': (
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [3, 5, 7, 10], 'clf__learning_rate': [0.01, 0.1, 0.2, 0.3], 'clf__subsample': [0.7, 0.8, 0.9], 'clf__colsample_bytree': [0.7, 0.8, 1]}
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

        fit_params_grid = {}
        if name == 'nn':
            fit_params_grid['clf__callbacks'] = [
                ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, min_lr=1e-6, verbose=0),
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0) 
            ]

        n_cv_splits = 5 
        min_samples_for_cv = n_cv_splits 
        
        valid_cv = True
        if len(y_train_processed.unique()) > 1:
            min_class_count = min(y_train_processed.value_counts())
            if min_class_count < n_cv_splits:
                logging.warning(f"Clase minoritaria en y_train para {name} tiene {min_class_count} muestras, menos que n_splits={n_cv_splits}. Intentando con n_splits={min_class_count if min_class_count >=2 else 2}.")
                n_cv_splits = max(2, min_class_count) # Necesita al menos 2, o el conteo de la clase si es >=2
                min_samples_for_cv = n_cv_splits
        else: # Solo una clase
            logging.warning(f"Solo una clase en y_train para {name}. CV no es posible.")
            valid_cv = False
        
        if X_train.shape[0] < min_samples_for_cv:
            logging.warning(f"No hay suficientes muestras en X_train ({X_train.shape[0]}) para CV con {n_cv_splits} splits en {name}.")
            valid_cv = False

        if not valid_cv:
            best_params_str = "CV skipped (pocas muestras/clases)"
            best_score_str = "N/A (CV skipped)"
            best_models[name] = None
            with open(hyper_file, 'a') as hf:
                hf.write(f"{name.upper()} best params: {best_params_str}\n")
                hf.write(f"{name.upper()} best CV ROC-AUC: {best_score_str}\n\n")
            print(f"  ! Hiperparámetros (o estado de error) guardados para {name.upper()}")
            continue

        grid = GridSearchCV(
            pipe, param_grid, cv=n_cv_splits, scoring='roc_auc', 
            n_jobs=1, verbose=1, refit=True, error_score='raise' 
        )
        
        best_estimator_for_model = None
        try:
            grid.fit(X_train, y_train_processed, **fit_params_grid) 
            best_estimator_for_model = grid.best_estimator_
            best_params_str = str(grid.best_params_)
            best_score_str = f"{grid.best_score_:.4f}"
        except ValueError as ve:
            logging.error(f"Error de ValueError (posiblemente CV) durante GridSearchCV para {name} en {exp_name}: {ve}", exc_info=True)
            print(f"Error de ValueError durante GridSearchCV para {name}, saltando este modelo: {ve}")
            best_params_str = "Error en CV"
            best_score_str = "Error en CV"
        except Exception as e:
            logging.error(f"Error general durante GridSearchCV para {name} en {exp_name}: {e}", exc_info=True)
            print(f"Error general durante GridSearchCV para {name}, saltando este modelo: {e}")
            best_params_str = "Error general"
            best_score_str = "Error general"

        if best_estimator_for_model:
            if name == 'nn':
                scaler_nn = best_estimator_for_model.named_steps['scaler']
                keras_model = best_estimator_for_model.named_steps['clf'].model
                scaler_path = os.path.join(output_dir, f"{exp_name}_nn_scaler.joblib")
                keras_path  = os.path.join(output_dir, f"{exp_name}_nn_model.h5")
                joblib.dump(scaler_nn, scaler_path)
                keras_model.save(keras_path)
                best_models[name] = (scaler_path, keras_path)
                print(f"  ✔ NN scaler guardado: {scaler_path}")
                print(f"  ✔ NN model guardado: {keras_path}")
            else:
                pipe_skl = best_estimator_for_model
                path = os.path.join(output_dir, f"{exp_name}_{name}_pipeline.joblib")
                joblib.dump(pipe_skl, path)
                best_models[name] = pipe_skl
                print(f"  ✔ {name} pipeline guardado: {path}")
        else:
            best_models[name] = None
            logging.warning(f"No se pudo obtener best_estimator para {name} debido a un error previo.")

        with open(hyper_file, 'a') as hf:
            hf.write(f"{name.upper()} best params: {best_params_str}\n")
            hf.write(f"{name.upper()} best CV ROC-AUC: {best_score_str}\n\n")
        print(f"  ✔ Params guardados para {name.upper()}")
        
    return {k: v for k, v in best_models.items() if v is not None}

def evaluate_and_save_reports(models, X_test, y_test, output_dir):
    report_file = os.path.join(output_dir, 'classification_reports.txt')
    with open(report_file, 'w') as rf:
        rf.write(f"Classification reports for {os.path.basename(output_dir)}\n")
        rf.write("="*60 + "\n\n")
    print(f"\n> Reports file: {report_file}")

    if X_test.empty or y_test.empty:
        logging.error("X_test o y_test están vacíos. No se pueden generar reportes.")
        print("X_test o y_test están vacíos. Saltando evaluación.")
        return

    y_test_processed = y_test.astype(int)

    for name, mdl_or_paths in models.items():
        print(f"\n> Evaluando {name.upper()}...")
        logging.info(f"Evaluando {name.upper()}...")
        y_pred, y_prob = None, None 

        try:
            if name == 'nn':
                scaler_path, keras_path = mdl_or_paths
                scaler_loaded = joblib.load(scaler_path)
                model_loaded  = load_model(keras_path, compile=False)
                Xs_eval = scaler_loaded.transform(X_test)
                
                probs_eval = model_loaded.predict(Xs_eval)
                if probs_eval.ndim == 1 or probs_eval.shape[1] == 1: # Output sigmoide
                    y_prob = probs_eval.flatten()
                    y_pred = (y_prob > 0.5).astype(int)
                elif probs_eval.shape[1] == 2: # Output softmax
                    y_prob = probs_eval[:,1]
                    y_pred = np.argmax(probs_eval, axis=1)
                else:
                    raise ValueError(f"Forma de salida inesperada del modelo NN: {probs_eval.shape}")

            else: 
                model_pipeline_loaded = mdl_or_paths
                probs_pipeline = model_pipeline_loaded.predict_proba(X_test)
                y_pred = model_pipeline_loaded.predict(X_test)
                
                if probs_pipeline.shape[1] == 2:
                    y_prob = probs_pipeline[:, 1]
                elif probs_pipeline.shape[1] == 1: 
                    # Esto puede pasar si el clasificador solo ve una clase durante el fit (muy raro con class_weight)
                    # o si el modelo es inherentemente para una sola salida de probabilidad (no común para predict_proba)
                    logging.warning(f"predict_proba para {name} devolvió una sola columna. Asumiendo probabilidad de clase positiva.")
                    y_prob = probs_pipeline[:,0] # Asumir que es la probabilidad de la clase positiva
                else:
                    raise ValueError(f"Forma de salida inesperada de predict_proba para {name}: {probs_pipeline.shape}")


            auc = roc_auc_score(y_test_processed, y_prob)
            rpt_str = classification_report(
                y_test_processed, y_pred,
                target_names=['No Fraude','Fraude'],
                digits=4,
                zero_division=0
            )
            print(f"{name.upper()} ROC-AUC: {auc:.4f}\n{rpt_str}")
            with open(report_file, 'a') as rf:
                rf.write(f"{name.upper()} ROC-AUC: {auc:.4f}\n")
                rf.write(rpt_str + "\n")
                rf.write("="*60 + "\n")
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
    CSV_PATH    = "C:/Users/saave/Desktop/Master_Thesis/Credit_card_data/creditcard.csv"
    BASE_OUTPUT = "C:/Users/saave/Desktop/data_balance/Resultados_NearMiss2" # Nombre de carpeta actualizado
    TECHNIQUE_NAME = "NearMiss"

    for scenario_type in ["scenario1", "scenario2"]:
        experiment_name   = f"{TECHNIQUE_NAME}_Scaling_{scenario_type}" 
        output_directory = os.path.join(BASE_OUTPUT, experiment_name)
        log_file_path   = os.path.join(output_directory, f"run_{experiment_name}.log")

        os.makedirs(output_directory, exist_ok=True)
        setup_logging(log_file_path)
        
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando experimento: {experiment_name} ===")
        logging.info(f"Técnica: {TECHNIQUE_NAME} con Escalado Refinado (Idea para Scen1)")
        logging.info(f"Output directory: {output_directory}")
        logging.info(f"=================================================")
        print(f"\n=== Iniciando: {experiment_name} ===")


        df_original_unscaled = load_and_prepare(CSV_PATH)
        logging.info(f"Dataset cargado (sin escalar). Forma: {df_original_unscaled.shape}. Clases: {dict(df_original_unscaled['Clase'].value_counts())}")

        # Validaciones básicas del DataFrame
        if not isinstance(df_original_unscaled, pd.DataFrame) or df_original_unscaled.empty:
            logging.error(f"El DataFrame cargado está vacío o no es un DataFrame. Abortando {experiment_name}.")
            continue
        if 'Clase' not in df_original_unscaled.columns:
            logging.error(f"La columna 'Clase' no se encuentra en el DataFrame. Abortando {experiment_name}.")
            continue
        if df_original_unscaled.shape[0] < 10: 
            logging.error(f"Dataset con muy pocas filas ({df_original_unscaled.shape[0]}). Abortando {experiment_name}.")
            continue
        class_counts_original = df_original_unscaled['Clase'].value_counts()
        if len(class_counts_original) < 2 or class_counts_original.get(0,0) == 0 or class_counts_original.get(1,0) == 0:
            logging.error(f"Dataset no tiene ambas clases o una clase está vacía: {class_counts_original.to_dict()}. Abortando {experiment_name}.")
            continue

        X_train_final_data, X_test_final_data, y_train_final_data, y_test_final_data = run_scenario_nearmiss_refined_scaling(df_original_unscaled.copy(), scenario_type)

        # Comprobaciones después de run_scenario
        if X_train_final_data.empty or (y_train_final_data is not None and y_train_final_data.empty):
            logging.error(f"X_train o y_train vacíos para {experiment_name} DESPUÉS de run_scenario. Saltando entrenamiento.")
            continue
        
        # Si y_train_final_data es None (puede pasar si X_train_final_data se vuelve vacío), no podemos hacer value_counts
        if y_train_final_data is not None and not y_train_final_data.empty:
            y_train_counts = y_train_final_data.value_counts()
            if len(y_train_counts) < 1: 
                logging.warning(f"y_train no tiene muestras para {experiment_name}. El entrenamiento podría fallar.")
            elif len(y_train_counts) < 2:
                logging.warning(f"y_train tiene solo una clase para {experiment_name}. GridSearchCV se adaptará o podría fallar.")
        elif y_train_final_data is None or y_train_final_data.empty : # Chequeo explícito de None o vacío
             logging.error(f"y_train_final_data es None o está vacía para {experiment_name}. Saltando entrenamiento.")
             continue


        start_training_time = time.time()
        trained_models_dict = train_and_save_models(X_train_final_data, y_train_final_data, experiment_name, output_directory)
        logging.info(f"Entrenamiento para {experiment_name} completado en {time.time() - start_training_time:.2f}s")

        if not trained_models_dict:
            logging.warning(f"No se entrenaron modelos exitosamente para {experiment_name}. Saltando evaluación.")
        else:
            if X_test_final_data is not None and not X_test_final_data.empty and \
               y_test_final_data is not None and not y_test_final_data.empty:
                 evaluate_and_save_reports(trained_models_dict, X_test_final_data, y_test_final_data, output_directory)
            else:
                logging.warning(f"X_test o y_test están vacíos o son None para {experiment_name}. Saltando evaluación.")
        
        logging.info(f"=== Fin experimento {experiment_name} ===\n")
        print(f"=== Fin: {experiment_name} ===\n")

    print(f"Todos los experimentos {TECHNIQUE_NAME} con escalado refinado han finalizado.")