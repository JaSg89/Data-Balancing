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
    df = df.rename(columns={'Time':'Tiempo', 'Amount':'Cantidad', 'Class':'Clase'})
    return df

def run_scenario_rus_with_controlled_scaling(df_original_unscaled, scenario):
    target_col_name = 'Clase'
    X_original_unscaled = df_original_unscaled.drop(target_col_name, axis=1)
    y_original = df_original_unscaled[target_col_name]
    
    rus = RandomUnderSampler(random_state=42)
    X_train_final, X_test_final, y_train_final, y_test_final = [pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')]


    if scenario == 'scenario1':
        # 1. Balance en TODO el dataset (df_original_unscaled)
        logging.info(">> ESCENARIO 1: RandomUnderSample en TODO el dataset (sin escalar) -> Split -> Escalado Separado")
        X_res_unscaled_np, y_res_np = rus.fit_resample(X_original_unscaled, y_original)
        
        X_res_unscaled = pd.DataFrame(X_res_unscaled_np, columns=X_original_unscaled.columns)
        y_res = pd.Series(y_res_np, name=y_original.name)
        logging.info(f"   RUS Scen1: Tamaño después de resample (antes de split y escalar): {X_res_unscaled.shape}, Distribución y_res: {dict(y_res.value_counts())}")

        if X_res_unscaled.empty or y_res.empty:
            logging.error("   RUS Scen1: X_res_unscaled o y_res vacíos después del balanceo. Abortando escenario.")
            return X_train_final, X_test_final, y_train_final, y_test_final


        stratify_param_s1 = y_res if not y_res.empty and len(y_res.unique()) > 1 else None
        X_train_raw, X_test_raw, y_train_final, y_test_final = train_test_split(
            X_res_unscaled, y_res, test_size=0.2, stratify=stratify_param_s1, random_state=42
        )
        logging.info(f"   RUS Scen1: Después de Split. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

        if not X_train_raw.empty:
            scaler_s1 = MinMaxScaler()
            X_train_final = pd.DataFrame(scaler_s1.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            if not X_test_raw.empty:
                X_test_final = pd.DataFrame(scaler_s1.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
            logging.info(f"   RUS Scen1: Después de Escalado. X_train_final: {X_train_final.shape}, X_test_final: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
        
    else: # scenario2
        logging.info(">> ESCENARIO 2: Split del dataset (sin escalar) -> Escalado Separado -> RandomUnderSample (solo en train escalado)")
        stratify_param_s2_initial = y_original if not y_original.empty and len(y_original.unique()) > 1 else None
        X_train_raw, X_test_raw, y_train_orig, y_test_final = train_test_split(
            X_original_unscaled, y_original, stratify=stratify_param_s2_initial, test_size=0.2,  random_state=42 )
        logging.info(f"   RUS Scen2: Después de Split inicial. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

        X_train_scaled_for_rus = pd.DataFrame()
        # X_test_final ya se inicializó y se llenará aquí si X_test_raw no es vacío

        if not X_train_raw.empty:
            scaler_s2 = MinMaxScaler()
            X_train_scaled_for_rus = pd.DataFrame(scaler_s2.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            if not X_test_raw.empty:
                X_test_final = pd.DataFrame(scaler_s2.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
            logging.info(f"   RUS Scen2: Después de Escalado. X_train_scaled_for_rus: {X_train_scaled_for_rus.shape}, X_test_final: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
        
        if not X_train_scaled_for_rus.empty and not y_train_orig.empty:
            X_train_res_np, y_train_res_np = rus.fit_resample(X_train_scaled_for_rus, y_train_orig)
            X_train_final = pd.DataFrame(X_train_res_np, columns=X_train_scaled_for_rus.columns)
            y_train_final = pd.Series(y_train_res_np, name=y_train_orig.name)
            logging.info(f"   RUS Scen2: Después de RUS en train. X_train_final: {X_train_final.shape}, y_train_final dist: {dict(y_train_final.value_counts() if not y_train_final.empty else {})}")
        else: # Si X_train_scaled_for_rus o y_train_orig son vacíos
            X_train_final = X_train_scaled_for_rus # Pasa el (posiblemente vacío) X_train escalado
            y_train_final = y_train_orig # Pasa el y_train original

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


def build_nn_model(n_inputs, learning_rate=0.001, dropout_rate=0.5):
    # ... (sin cambios, como lo tenías)
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

def train_and_save_models(X_train, y_train, exp_name, output_dir):
    # ... (sin cambios, como lo tenías, incluyendo el ImbPipeline con MinMaxScaler)
    hyper_file = os.path.join(output_dir, 'hyperparameters.txt')
    with open(hyper_file, 'w') as hf:
        hf.write(f"Hyperparameters for experiment {exp_name}\n")
        hf.write("="*60 + "\n\n")
    print(f"> Hyperparameters will be saved to: {hyper_file}")

    if X_train.empty or X_train.shape[1] == 0:
        logging.error(f"X_train está vacío o no tiene características ANTES de entrenar modelos para {exp_name}. Saltando entrenamiento.")
        return {}

    n_inputs = X_train.shape[1]

    # KerasClassifier como lo tenías, pero verbose=0 y callbacks en fit_params
    nn_wrapper = KerasClassifier(
        build_fn=build_nn_model,
        n_inputs=n_inputs,
        verbose=0, 
    )

    specs = { # Usando tus specs originales
        'nn': (
            nn_wrapper,
            {
                'clf__learning_rate':    [0.001, 0.0001],
                'clf__dropout_rate':     [0.3],
                'clf__batch_size':       [16, 32, 64],
                'clf__epochs':           [50],
                # 'clf__validation_split':[0.1] # Se pasa en fit_params
            }
        ),
        'logreg': (
            LogisticRegression(random_state=42, max_iter=1000), # Sin class_weight, ya que RUS balancea
            {'clf__penalty': ['l1','l2'], 'clf__C': [0.001, 0.01, 0.7, 0.1, 0.2, 1, 10, 100, 1000], 'clf__solver': ['liblinear']}
        ),
        'svm': (
            SVC(probability=True, random_state=42), # Sin class_weight
            {'clf__C': [0.5, 0.7, 0.9, 1, 1.5], 'clf__kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
        ),
        'rf': (
            RandomForestClassifier(random_state=42), # Sin class_weight
            {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 10, 20, 30], 'clf__min_samples_split': [2, 5, 10], 'clf__min_samples_leaf': [1, 2, 4]}
        ),
        'xgb': (
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [3, 5, 7, 10],
                'clf__learning_rate': [0.01, 0.1, 0.2, 0.3],
                'clf__subsample': [0.7, 0.8, 0.9],
                'clf__colsample_bytree': [0.7, 0.8, 1]
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
            fit_params['clf__callbacks'] = [ # Callbacks originales
                ReduceLROnPlateau(monitor='loss', patience=5, factor=0.2, min_lr=1e-6, verbose=0), 
                EarlyStopping(monitor='loss', patience=40, restore_best_weights=True, verbose=0) 
            ]
            fit_params['clf__validation_split'] = 0.1

        n_cv_splits = 4 
        # Lógica de ajuste de CV si es necesario
        if len(y_train_processed.unique()) > 1:
            min_class_count = min(y_train_processed.value_counts())
            if min_class_count < n_cv_splits :
                logging.warning(f"Clase minoritaria en y_train para {name} tiene {min_class_count} muestras, menos que cv={n_cv_splits}. Ajustando cv.")
                n_cv_splits = max(2, min_class_count) if min_class_count >=2 else 1
                if n_cv_splits == 1: # No se puede hacer CV con 1 split
                    logging.error(f"No se puede realizar CV para {name} con n_splits=1. Saltando este modelo.")
                    best_models[name] = None
                    with open(hyper_file, 'a') as hf:
                        hf.write(f"{name.upper()} best params: CV SKIPPED (n_splits=1)\n")
                        hf.write(f"{name.upper()} best CV ROC-AUC: N/A\n\n")
                    continue # Saltar al siguiente modelo
        if X_train.shape[0] < n_cv_splits :
             logging.warning(f"No hay suficientes muestras en X_train ({X_train.shape[0]}) para CV con {n_cv_splits} splits en {name}. Ajustando cv.")
             n_cv_splits = max(2, X_train.shape[0]) if X_train.shape[0] >=2 else 1
             if n_cv_splits == 1:
                logging.error(f"No se puede realizar CV para {name} con n_splits=1 (pocas muestras). Saltando este modelo.")
                best_models[name] = None
                with open(hyper_file, 'a') as hf:
                    hf.write(f"{name.upper()} best params: CV SKIPPED (n_splits=1)\n")
                    hf.write(f"{name.upper()} best CV ROC-AUC: N/A\n\n")
                continue
        
        cv_obj = StratifiedKFold( n_splits=min(4, min_class_count), shuffle=True, random_state=42)

        grid = GridSearchCV(
            pipe, param_grid,
            cv=cv_obj, scoring='roc_auc',
            n_jobs=1, verbose=1, refit=True, error_score='raise' 
        )
        
        best_estimator_for_model = None
        try:
            if name == 'nn':
                grid.fit(X_train, y_train_processed, **fit_params)
            else:
                grid.fit(X_train, y_train_processed)
            best_estimator_for_model = grid.best_estimator_
            best_params_str = str(grid.best_params_)
            best_score_str = f"{grid.best_score_:.4f}"
        except ValueError as ve:
            logging.error(f"ValueError (posiblemente CV) en GridSearchCV para {name}: {ve}", exc_info=True)
            best_params_str = "Error en CV"
            best_score_str = "Error en CV"
        except Exception as e:
            logging.error(f"Error general en GridSearchCV para {name}: {e}", exc_info=True)
            best_params_str = "Error general"
            best_score_str = "Error general"

        if best_estimator_for_model:
            if name == 'nn':
                best_pipeline_nn = best_estimator_for_model
                scaler_nn = best_pipeline_nn.named_steps['scaler']
  
                if hasattr(best_pipeline_nn.named_steps['clf'], 'model_') :
                    keras_model_nn = best_pipeline_nn.named_steps['clf'].model_
                elif hasattr(best_pipeline_nn.named_steps['clf'], 'model'): 
                    keras_model_nn = best_pipeline_nn.named_steps['clf'].model
                else:
                    logging.error(f"No se pudo encontrar el atributo del modelo Keras para {name}")
                    best_models[name] = None
                    with open(hyper_file, 'a') as hf:
                        hf.write(f"{name.upper()} best params: {best_params_str} (Error Keras model attr)\n")
                        hf.write(f"{name.upper()} best CV ROC-AUC: {best_score_str}\n\n")
                    continue

                scaler_path = os.path.join(output_dir, f"{exp_name}_{name}_scaler.joblib")
                keras_model_path = os.path.join(output_dir, f"{exp_name}_{name}_keras_model.h5")
                joblib.dump(scaler_nn, scaler_path)
                keras_model_nn.save(keras_model_path)
                logging.info(f"  ✔ Scaler NN guardado: {scaler_path}")
                logging.info(f"  ✔ Modelo Keras NN guardado: {keras_model_path}")
                best_models[name] = (scaler_path, keras_model_path)
            else:
                model_path = os.path.join(output_dir, f"{exp_name}_{name}_best.joblib")
                joblib.dump(best_estimator_for_model, model_path)
                logging.info(f"  ✔ Modelo guardado: {model_path}")
                best_models[name] = best_estimator_for_model
        else:
            best_models[name] = None
            logging.warning(f"No se pudo obtener best_estimator para {name}.")

        with open(hyper_file, 'a') as hf:
            hf.write(f"{name.upper()} best params: {best_params_str}\n")
            hf.write(f"{name.upper()} best CV ROC-AUC: {best_score_str}\n\n")
        print(f"  ✔ Hiperparámetros (o estado de error) guardados para {name.upper()}")
    
    return {k:v for k,v in best_models.items() if v is not None}


def evaluate_and_save_reports(models, X_test, y_test, output_dir):
    # ... (sin cambios, como lo tenías)
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
                scaler_path, keras_model_path = model_or_paths
                scaler_loaded = joblib.load(scaler_path)
                keras_model = load_model(keras_model_path, compile=False)
                X_test_for_eval = scaler_loaded.transform(X_test)
                y_prob_all_classes = keras_model.predict(X_test_for_eval)
                y_prob = y_prob_all_classes[:, 1]
                y_pred = np.argmax(y_prob_all_classes, axis=1)
            else:
                model_pipeline = model_or_paths
                y_prob_all_classes = model_pipeline.predict_proba(X_test)
                if y_prob_all_classes.shape[1] == 2:
                    y_prob = y_prob_all_classes[:, 1]
                    y_pred = model_pipeline.predict(X_test)
                else:
                    logging.error(f"Forma inesperada de y_prob_all_classes para {name}: {y_prob_all_classes.shape}")
                    y_prob = np.zeros(len(y_test_processed))
                    y_pred = np.zeros(len(y_test_processed))

            auc = float('nan')
            report_str = "N/A"

            if y_pred is not None and y_prob is not None:
                if len(np.unique(y_test_processed)) < 2 :
                    logging.warning(f"Solo una clase presente en y_test para {name}. ROC-AUC no es calculable.")
                elif len(y_prob) == 0:
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
                    target_names=["No Fraude", "Fraude"],
                    digits=4,
                    zero_division=0
                )
            else:
                logging.error(f"Predicciones (y_pred o y_prob) no generadas para {name}.")

            print(f"{name.upper()} ROC-AUC: {auc:.4f}\n{report_str}")
            with open(report_file, "a") as rf:
                rf.write(f"{name.upper()} ROC-AUC: {auc:.4f}\n")
                rf.write(report_str + "\n")
                rf.write("-" * 60 + "\n")
            print(f"  ✔ Reporte guardado para {name.upper()}")
        
        except Exception as e_eval:
            logging.error(f"Error durante la evaluación de {name} en {output_dir}: {e_eval}", exc_info=True)
            print(f"Error durante la evaluación de {name}: {e_eval}")
            with open(report_file, "a") as rf:
                rf.write(f"{name.upper()} - ERROR EN EVALUACIÓN: {e_eval}\n{'-'*60}\n")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    # --- BLOQUE MAIN RESTAURADO A TU ORIGINAL ---
    CSV_PATH    = "C:/Users/saave/Desktop/Master_Thesis/Credit_card_data/creditcard.csv"
    BASE_OUTPUT = "C:/Users/saave/Desktop/data_balance/Resultados_Submuestreo_Aleatorio2"
    technique   = "Undersample" 
    # --- FIN BLOQUE MAIN RESTAURADO ---

    for scenario in ["scenario1", "scenario2"]:
        exp_name   = f"{technique}_{scenario}" # Usa tu variable 'technique' original
        output_dir = os.path.join(BASE_OUTPUT, exp_name) # Usa tu BASE_OUTPUT original
        log_file   = os.path.join(output_dir, f"run_{exp_name}.log") # Nombre de log único

        os.makedirs(output_dir, exist_ok=True)
        setup_logging(log_file)
        
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando experimento: {exp_name} ===")
        logging.info(f"Técnica: Random Under Sampling con Escalado Controlado") # Descripción actualizada
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"=================================================")
        print(f"\n=== Iniciando experimento: {exp_name} ===")

        df_original = load_and_prepare(CSV_PATH)
        logging.info(f"Dataset cargado (sin escalar). Forma: {df_original.shape}. Clases: {dict(df_original['Clase'].value_counts())}")

        if df_original.shape[0] < 10:
            logging.error(f"Dataset con muy pocas filas ({df_original.shape[0]}). Abortando {exp_name}.")
            continue
        class_counts = df_original['Clase'].value_counts()
        if len(class_counts) < 2 or class_counts.get(0,0) == 0 or class_counts.get(1,0) == 0:
            logging.error(f"Dataset no tiene ambas clases o una clase está vacía: {class_counts.to_dict()}. Abortando {exp_name}.")
            continue

        # Llamar a la función de escenario modificada
        # (He renombrado la función a run_scenario_rus_with_controlled_scaling para claridad interna,
        # pero puedes volver a llamarla run_scenario si prefieres y actualizar la llamada aquí)
        X_train, X_test, y_train, y_test = run_scenario_rus_with_controlled_scaling(df_original.copy(), scenario)

        if X_train.empty or y_train.empty:
            logging.error(f"X_train o y_train vacíos para {exp_name} DESPUÉS de run_scenario. Saltando entrenamiento.")
            continue
        if len(y_train.value_counts()) < 1:
            logging.warning(f"y_train no tiene muestras para {exp_name}. El entrenamiento podría fallar.")
        elif len(y_train.value_counts()) < 2:
             logging.warning(f"y_train tiene solo una clase para {exp_name}. GridSearchCV se adaptará o podría fallar.")

        start_time = time.time()
        models = train_and_save_models(X_train, y_train, exp_name, output_dir)
        logging.info(f"Entrenamiento completo en {time.time() - start_time:.1f}s")

        if not models:
            logging.warning(f"No se entrenaron modelos exitosamente para {exp_name}. Saltando evaluación.")
        else:
            if X_test is not None and not X_test.empty and y_test is not None and not y_test.empty:
                 evaluate_and_save_reports(models, X_test, y_test, output_dir)
            else:
                logging.warning(f"X_test o y_test están vacíos para {exp_name} DESPUÉS de run_scenario. Saltando evaluación.")

        logging.info(f"=== Fin experimento {exp_name} ===\n")
        print(f"=== Fin experimento: {exp_name} ===")

    print("Todos los experimentos RUS con escalado controlado han finalizado.")