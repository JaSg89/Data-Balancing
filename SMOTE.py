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
# Configuración global
# -------------------------------------------------------------------
warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)
LOG_FMT = "%(asctime)s %(levelname)-8s %(message)s"

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

def build_nn_model(n_inputs, learning_rate=0.001, dropout_rate=0.5): # Nombre consistente
    model = Sequential([
        Dense(32, input_shape=(n_inputs,), activation='relu'), # Como en tu script original
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
# Función principal de escenario (PARA SMOTE con escalado controlado)
# -------------------------------------------------------------------
def run_scenario_smote_controlled_scaling(df_original_unscaled, scenario):
    target_col_name = 'Clase'
    # Asegurarse de que X_original_unscaled es un DataFrame y y_original es una Serie
    if not isinstance(df_original_unscaled, pd.DataFrame):
        raise TypeError("df_original_unscaled debe ser un DataFrame de Pandas.")
    
    X_original_unscaled = df_original_unscaled.drop(target_col_name, axis=1, errors='ignore')
    if target_col_name not in df_original_unscaled.columns:
        raise ValueError(f"La columna objetivo '{target_col_name}' no se encuentra en df_original_unscaled.")
    y_original = df_original_unscaled[target_col_name]

    smote_sampler = SMOTE(random_state=42)
    logging.info(f"Usando SMOTE con escalado controlado.")

    X_train_final, X_test_final, y_train_final, y_test_final = [pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')]

    if scenario == 'scenario1':
        # Escenario 1 SMOTE: Fuga de Datos INTENCIONAL
        logging.info(f">> ESCENARIO 1 (SMOTE): Escalado Global PRE-SMOTE -> SMOTE en TODO el dataset escalado -> Split")
        
        # 1. Cargar datos (hecho) -> X_original_unscaled, y_original

        # 2. Escalado Global PRE-SMOTE
        if X_original_unscaled.empty:
            logging.error("   SMOTE Scen1: X_original_unscaled está vacío. Abortando escenario.")
            return X_train_final, X_test_final, y_train_final, y_test_final # Devuelve vacíos

        scaler_s1_global_temp = MinMaxScaler()
        # Usar .copy() para evitar modificar X_original_unscaled si se reutiliza
        X_global_scaled_for_smote = pd.DataFrame(
            scaler_s1_global_temp.fit_transform(X_original_unscaled.copy()), 
            columns=X_original_unscaled.columns,
            index=X_original_unscaled.index
        )
        logging.info(f"   SMOTE Scen1: Datos originales escalados globalmente para SMOTE. Forma: {X_global_scaled_for_smote.shape}")

        # 3. Balanceo con SMOTE (Sobre Datos Globalmente Escalados)
        X_res_smote_global_scaled = pd.DataFrame() # Inicializar
        y_res_smote = pd.Series(dtype='float64')   # Inicializar

        try:
            if X_global_scaled_for_smote.empty or y_original.empty:
                logging.warning("   SMOTE Scen1: No hay datos para SMOTE. Usando datos escalados globalmente sin balanceo.")
                X_res_smote_global_scaled = X_global_scaled_for_smote
                y_res_smote = y_original
            else:
                # SMOTE devuelve arrays NumPy
                X_res_np, y_res_np = smote_sampler.fit_resample(X_global_scaled_for_smote.to_numpy(), y_original.to_numpy())
                X_res_smote_global_scaled = pd.DataFrame(X_res_np, columns=X_global_scaled_for_smote.columns)
                y_res_smote = pd.Series(y_res_np, name=y_original.name)
            logging.info(f"   SMOTE Scen1: Tamaño después de SMOTE (en datos escalados globalmente): {X_res_smote_global_scaled.shape}, Distribución y_res: {dict(y_res_smote.value_counts())}")
        except ValueError as e:
            logging.error(f"   SMOTE Scen1: Error durante SMOTE.fit_resample: {e}. Usando datos escalados globalmente sin balanceo.")
            X_res_smote_global_scaled = X_global_scaled_for_smote # Fallback
            y_res_smote = y_original

        if X_res_smote_global_scaled.empty or y_res_smote.empty:
            logging.error("   SMOTE Scen1: X_res o y_res vacíos después del balanceo/fallback. Abortando escenario.")
            return X_train_final, X_test_final, y_train_final, y_test_final # Devuelve vacíos
        
        # 4. Split (División del Conjunto Balanceado y Globalmente Escalado)
        # Los datos (X_train_final, X_test_final) ya están escalados por el scaler_s1_global_temp
        stratify_param_s1 = y_res_smote if not y_res_smote.empty and len(y_res_smote.unique()) > 1 else None
        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
            X_res_smote_global_scaled, y_res_smote, test_size=0.2, stratify=stratify_param_s1, random_state=42, shuffle=True
        )
        logging.info(f"   SMOTE Scen1: Después de Split. X_train_final (escalado): {X_train_final.shape}, X_test_final (escalado): {X_test_final.shape}")

    else: # scenario2 (Sin Fuga de Datos)
        # 1. Split del dataset ORIGINAL (df_original_unscaled)
        logging.info(f">> ESCENARIO 2 (SMOTE): Split del dataset (sin escalar) -> Escalado Separado LIMPIO -> SMOTE (solo en train escalado)")
        stratify_param_s2_initial = y_original if not y_original.empty and len(y_original.unique()) > 1 else None
        X_train_raw, X_test_raw, y_train_orig, y_test_final_orig = train_test_split(
            X_original_unscaled, y_original, test_size=0.2, stratify=stratify_param_s2_initial, random_state=42, shuffle=True
        )
        logging.info(f"   SMOTE Scen2: Después de Split inicial. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

        # 2. Escalado SEPARADO y LIMPIO para modelado (y para SMOTE en train)
        X_train_scaled_for_smote = pd.DataFrame()
        y_test_final = y_test_final_orig # y_test no cambia

        if not X_train_raw.empty:
            scaler_s2_modelado = MinMaxScaler()
            X_train_scaled_for_smote = pd.DataFrame(scaler_s2_modelado.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            if not X_test_raw.empty:
                X_test_final = pd.DataFrame(scaler_s2_modelado.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
            logging.info(f"   SMOTE Scen2: Después de Escalado LIMPIO. X_train_scaled_for_smote: {X_train_scaled_for_smote.shape}, X_test_final: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
        
        # 3. SMOTE solo en train (que ya está escalado para modelado)
        X_train_final = X_train_scaled_for_smote.copy() # Inicializar por si SMOTE falla
        y_train_final = y_train_orig.copy()            # Inicializar

        if not X_train_scaled_for_smote.empty and not y_train_orig.empty:
            try:
                # Comprobar si hay al menos una clase minoritaria para SMOTE
                if len(y_train_orig.value_counts().get(1, [])) < smote_sampler.k_neighbors +1 and len(y_train_orig.value_counts()) > 1 : #SMOTE necesita k+1 muestras de la minoría
                     logging.warning(f"   SMOTE Scen2: No hay suficientes muestras en la clase minoritaria del train ({len(y_train_orig.value_counts().get(1,[]))}) para SMOTE con k_neighbors={smote_sampler.k_neighbors}. Usando train escalado sin balanceo.")
                     # X_train_final y y_train_final ya están seteados a los datos pre-balanceo
                elif len(y_train_orig.unique()) < 2:
                    logging.warning("   SMOTE Scen2: Solo una clase presente en y_train_orig. SMOTE no se aplicará.")
                    # X_train_final y y_train_final ya están seteados
                else:
                    X_res_np_train, y_res_np_train = smote_sampler.fit_resample(X_train_scaled_for_smote.to_numpy(), y_train_orig.to_numpy())
                    X_train_final = pd.DataFrame(X_res_np_train, columns=X_train_scaled_for_smote.columns)
                    y_train_final = pd.Series(y_res_np_train, name=y_train_orig.name)
                logging.info(f"   SMOTE Scen2: Después de SMOTE en train. X_train_final (escalado): {X_train_final.shape}, y_train_final dist: {dict(y_train_final.value_counts() if not y_train_final.empty else {})}")
            except ValueError as e:
                logging.error(f"   SMOTE Scen2: Error durante SMOTE.fit_resample en train: {e}. Usando train escalado sin balanceo.")
                # X_train_final y y_train_final ya están seteados a los datos pre-balanceo
        
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
# Funciones de entrenamiento y evaluación (Prácticamente sin cambios de tu script original de SMOTE)
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
        callbacks=[ 
            ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, min_lr=1e-6, verbose=0),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        ]
    )
    # Grillas de hiperparámetros de tu script original de SMOTE (ajustadas ligeramente)
    specs = {
        
        'nn': (
            nn_wrapper,
            {
                'clf__learning_rate':[0.0001], 
                'clf__dropout_rate':[0.3],   
                'clf__batch_size':[32],      
                'clf__epochs':[75],           
                'clf__validation_split': [0.1]   
            }
        ),

        'logreg': (
            LogisticRegression(random_state=42, max_iter=1000), 
            {'clf__penalty':['l1','l2'], 'clf__C':[0.1], 'clf__solver':['liblinear']}
        ),
        'svm': (
            SVC(probability=True, random_state=42, class_weight='balanced'), 
            {'clf__C':[1, 10], 'clf__kernel':['linear']} 
        ),
        'rf': (
            RandomForestClassifier(random_state=42), 
            {'clf__n_estimators':[200], 'clf__max_depth':[20, None]} 
        ),
        'xgb': (
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            {'clf__n_estimators':[100], 'clf__max_depth':[10]} 
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
            # Callbacks ya están en la definición de nn_wrapper
            pass # validation_split se pasa a través de param_grid


        n_cv_splits = 5 # Como en tu script de SMOTE
        min_samples_for_cv = n_cv_splits 
        
        valid_cv = True
        if len(y_train_processed.unique()) > 1:
            min_class_count = min(y_train_processed.value_counts())
            if min_class_count < n_cv_splits:
                logging.warning(f"Clase minoritaria en y_train para {name} tiene {min_class_count} muestras, menos que n_splits={n_cv_splits}. Intentando con n_splits={max(2, min_class_count)}.")
                n_cv_splits = max(2, min_class_count) 
                min_samples_for_cv = n_cv_splits
        else: 
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
        
        cv_obj = StratifiedKFold( n_splits=min(4, min_class_count), shuffle=True, random_state=42)

        grid = GridSearchCV(
            pipe, param_grid,
            cv=cv_obj, scoring='f1', 
            n_jobs=5, # Cambiado de 2 a 1 por consistencia y evitar problemas con Keras
            verbose=2, refit=True, error_score='raise'
        )
        
        best_estimator_for_model = None
        try:
            # Para NN, los callbacks están en nn_wrapper, validation_split en param_grid
            grid.fit(X_train, y_train_processed) 
            best_estimator_for_model = grid.best_estimator_
            best_params_str = str(grid.best_params_)
            best_score_str = f"{grid.best_score_:.4f}"
        except ValueError as ve:
            logging.error(f"Error de ValueError (posiblemente CV) durante GridSearchCV para {name} en {exp_name}: {ve}", exc_info=True)
            best_params_str = "Error en CV"
            best_score_str = "Error en CV"
        except Exception as e:
            logging.error(f"Error general durante GridSearchCV para {name} en {exp_name}: {e}", exc_info=True)
            best_params_str = "Error general"
            best_score_str = "Error general"

        if best_estimator_for_model:
            if name == 'nn':
                scaler_nn = best_estimator_for_model.named_steps['scaler']
                keras_model_nn = best_estimator_for_model.named_steps['clf'].model 
                
                scaler_path = os.path.join(output_dir, f"{exp_name}_{name}_scaler.joblib")
                keras_model_path = os.path.join(output_dir, f"{exp_name}_{name}_keras_model.h5")
                
                joblib.dump(scaler_nn, scaler_path)
                keras_model_nn.save(keras_model_path)
                
                print(f"  ✔ Scaler NN guardado: {scaler_path}")
                print(f"  ✔ Modelo Keras NN guardado: {keras_model_path}")
                best_models[name] = (scaler_path, keras_model_path)
            else:
                model_path = os.path.join(output_dir, f"{exp_name}_{name}_best_pipeline.joblib")
                joblib.dump(best_estimator_for_model, model_path)
                print(f"  ✔ Modelo (pipeline) guardado: {model_path}")
                best_models[name] = best_estimator_for_model
        else:
            best_models[name] = None
            logging.warning(f"No se pudo obtener best_estimator para {name} debido a un error previo.")

        with open(hyper_file, 'a') as hf:
            hf.write(f"{name.upper()} best params: {best_params_str}\n")
            hf.write(f"{name.upper()} best CV ROC-AUC: {best_score_str}\n\n")
        print(f"  ✔ Hiperparámetros guardados para {name.upper()}")
        
    return {k:v for k,v in best_models.items() if v is not None}


def evaluate_and_save_reports(models, X_test, y_test, output_dir):
    # ... (Esta función puede ser idéntica a la de la versión NearMiss_RefinedScaling)
    report_file = os.path.join(output_dir, "classification_reports.txt")
    with open(report_file, "w") as rf:
        rf.write(f"Classification reports for {os.path.basename(output_dir)}\n")
        rf.write("=" * 60 + "\n\n")
    print(f"\n> Reports will be saved to: {report_file}")

    if X_test.empty or y_test.empty:
        logging.error("X_test o y_test están vacíos. No se pueden generar reportes.")
        return

    y_test_processed = y_test.astype(int)

    for name, model_or_paths in models.items():
        print(f"\n> Evaluando {name.upper()}...")
        logging.info(f"Evaluando {name.upper()}...")
        y_pred, y_prob = None, None 

        try:
            if name == "nn":
                scaler_path, keras_model_path = model_or_paths
                scaler = joblib.load(scaler_path)
                keras_model = load_model(keras_model_path, compile=False) 
                # X_test ya está escalado. El scaler cargado es el del pipeline de entrenamiento.
                X_test_for_eval = scaler.transform(X_test) 
                
                y_prob_all_classes = keras_model.predict(X_test_for_eval)

                if y_prob_all_classes.ndim == 1 or y_prob_all_classes.shape[1] == 1: 
                    y_prob = y_prob_all_classes.flatten()
                    y_pred = (y_prob > 0.5).astype(int)
                elif y_prob_all_classes.shape[1] == 2: 
                    y_prob = y_prob_all_classes[:, 1]
                    y_pred = np.argmax(y_prob_all_classes, axis=1)
                else:
                    raise ValueError(f"Forma de salida inesperada del modelo NN: {y_prob_all_classes.shape}")

            else: 
                model_pipeline = model_or_paths
                # El pipeline se encarga de escalar X_test
                y_prob_all_classes = model_pipeline.predict_proba(X_test)
                y_pred = model_pipeline.predict(X_test)
                
                if y_prob_all_classes.shape[1] == 2:
                    y_prob = y_prob_all_classes[:, 1]
                elif y_prob_all_classes.shape[1] == 1:
                    logging.warning(f"predict_proba para {name} devolvió una sola columna. Asumiendo probabilidad de clase positiva.")
                    y_prob = y_prob_all_classes[:,0]
                else:
                    raise ValueError(f"Forma de salida inesperada de predict_proba para {name}: {y_prob_all_classes.shape}")


            auc = roc_auc_score(y_test_processed, y_prob)
            report_str = classification_report(
                y_test_processed, y_pred,
                target_names=["No Fraude", "Fraude"],
                digits=4,
                zero_division=0
            )
            print(f"{name.upper()} ROC-AUC: {auc:.4f}\n{report_str}")

            with open(report_file, "a") as rf:
                rf.write(f"{name.upper()} ROC-AUC: {auc:.4f}\n")
                rf.write(report_str + "\n")
                rf.write("-" * 60 + "\n") # Consistente con tu script original
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
    BASE_OUTPUT = "/scratch/sivar/jarevalo/jsaavedra/resultados_SMOTE" 

    TECHNIQUE_NAME = "SMOTE" # Tu técnica

    for scenario_type in ["scenario1", "scenario2"]:
        experiment_name   = f"{TECHNIQUE_NAME}_Scaling_{scenario_type}" 
        output_dir = os.path.join(BASE_OUTPUT, experiment_name) # <--- Variable definida como output_dir
        log_file_path   = os.path.join(output_dir, f"run_{experiment_name}.log")

        os.makedirs(output_dir, exist_ok=True)
        setup_logging(log_file_path)
        
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando experimento: {experiment_name} ===")
        logging.info(f"Técnica: {TECHNIQUE_NAME} con Escalado Controlado")
        # logging.info(f"Output directory: {output_directory}") # <--- LÍNEA ORIGINAL CON ERROR
        logging.info(f"Output directory: {output_dir}")      # <--- LÍNEA CORREGIDA
        logging.info(f"=================================================")
        print(f"\n=== Iniciando experimento: {experiment_name} ===")

        # ... el resto de tu bloque main ...
        df_original_unscaled = load_and_prepare(CSV_PATH) # Cambiado de load_and_prepare_unscaled si solo hay una
        logging.info(f"Dataset cargado (sin escalar). Forma: {df_original_unscaled.shape}. Clases: {dict(df_original_unscaled['Clase'].value_counts())}")

        if not isinstance(df_original_unscaled, pd.DataFrame) or df_original_unscaled.empty:
            logging.error(f"El DataFrame cargado está vacío o no es un DataFrame. Abortando {experiment_name}.")
            continue
        if 'Clase' not in df_original_unscaled.columns: # Asumiendo que 'Clase' es el target después de renombrar
            logging.error(f"La columna 'Clase' no se encuentra en el DataFrame. Abortando {experiment_name}.")
            continue
        if df_original_unscaled.shape[0] < 10: 
            logging.error(f"Dataset con muy pocas filas ({df_original_unscaled.shape[0]}). Abortando {experiment_name}.")
            continue
        class_counts_original = df_original_unscaled['Clase'].value_counts()
        if len(class_counts_original) < 2 or class_counts_original.get(0,0) == 0 or class_counts_original.get(1,0) == 0:
            logging.error(f"Dataset no tiene ambas clases o una clase está vacía: {class_counts_original.to_dict()}. Abortando {experiment_name}.")
            continue

        X_train_final_data, X_test_final_data, y_train_final_data, y_test_final_data = run_scenario_smote_controlled_scaling(df_original_unscaled.copy(), scenario_type)

        if X_train_final_data.empty or (y_train_final_data is not None and y_train_final_data.empty): # Comprobación robusta
            logging.error(f"X_train o y_train vacíos para {experiment_name} DESPUÉS de run_scenario. Saltando entrenamiento.")
            continue
        
        if y_train_final_data is not None and not y_train_final_data.empty:
            y_train_counts = y_train_final_data.value_counts()
            if len(y_train_counts) < 1: 
                logging.warning(f"y_train no tiene muestras para {experiment_name}. El entrenamiento podría fallar.")
            elif len(y_train_counts) < 2:
                logging.warning(f"y_train tiene solo una clase para {experiment_name}. GridSearchCV se adaptará o podría fallar.")
        elif y_train_final_data is None or y_train_final_data.empty : # Comprobación más explícita
             logging.error(f"y_train_final_data es None o está vacía para {experiment_name}. Saltando entrenamiento.")
             continue

        start_training_time = time.time()
        trained_models_dict = train_and_save_models(X_train_final_data, y_train_final_data, experiment_name, output_dir) # output_dir
        logging.info(f"Entrenamiento para {experiment_name} completado en {time.time() - start_training_time:.2f}s")

        if not trained_models_dict:
            logging.warning(f"No se entrenaron modelos exitosamente para {experiment_name}. Saltando evaluación.")
        else:
            if X_test_final_data is not None and not X_test_final_data.empty and \
               y_test_final_data is not None and not y_test_final_data.empty:
                 evaluate_and_save_reports(trained_models_dict, X_test_final_data, y_test_final_data, output_dir) # output_dir
            else:
                logging.warning(f"X_test o y_test están vacíos o son None para {experiment_name}. Saltando evaluación.")
        
        logging.info(f"=== Fin experimento {experiment_name} ===\n")
        print(f"=== Fin experimento: {experiment_name} ===\n")

    print(f"Todos los experimentos {TECHNIQUE_NAME} con escalado controlado han finalizado.")