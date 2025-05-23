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

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'Time':'Tiempo','Amount':'Cantidad','Class':'Clase'})
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
# Funciones de resampling MEUS (Escalado INTERNO de todas las features de clase may. para Mahalanobis)
# Estas funciones devuelven datos en su escala ORIGINAL (la que tenían al entrar)
# -------------------------------------------------------------------
def meus_resample_logic(data_features_orig: pd.DataFrame, # Características de la clase mayoritaria (sin escalar para modelado)
                        n_minority_samples: int,
                        operation_context: str # "Scen1" o "Scen2" para logging
                        ) -> pd.Index: # Devuelve los índices de la clase mayoritaria a conservar
    """
    Lógica central de MEUS para seleccionar muestras de la clase mayoritaria.
    Escala INTERNAMENTE todas las características de 'data_features_orig' para el cálculo de Mahalanobis.
    Devuelve los índices de 'data_features_orig' a conservar.
    """
    if data_features_orig.empty or data_features_orig.shape[1] == 0:
        logging.warning(f"MEUS {operation_context}: La clase mayoritaria no tiene características. "
                        "Devolviendo todos los índices disponibles de la clase mayoritaria (si n_minority lo permite).")
        return data_features_orig.head(n_minority_samples).index # o .index si n_minority_samples > len

    # Escalado INTERNO de TODAS las características de la clase mayoritaria para Mahalanobis
    scaler_internal = MinMaxScaler()
    # data_features_orig ya es SOLO las features de la clase mayoritaria
    X_maj_scaled_for_mahalanobis = pd.DataFrame(scaler_internal.fit_transform(data_features_orig),
                                                columns=data_features_orig.columns,
                                                index=data_features_orig.index)

    mean_vector = X_maj_scaled_for_mahalanobis.mean().values
    
    if X_maj_scaled_for_mahalanobis.shape[0] <= X_maj_scaled_for_mahalanobis.shape[1]:
        logging.warning(f"MEUS {operation_context}: Covarianza para clase mayoritaria podría ser singular (muestras={X_maj_scaled_for_mahalanobis.shape[0]}, feats={X_maj_scaled_for_mahalanobis.shape[1]}). Usando ddof=0.")
        cov_matrix = np.cov(X_maj_scaled_for_mahalanobis.values, rowvar=False, ddof=0)
    else:
        cov_matrix = np.cov(X_maj_scaled_for_mahalanobis.values, rowvar=False)

    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        logging.warning(f"MEUS {operation_context}: Matriz de covarianza de clase mayoritaria singular, usando pseudo-inversa.")
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    diff_values_maj = X_maj_scaled_for_mahalanobis.values - mean_vector
    left_maj = diff_values_maj.dot(inv_cov_matrix)
    distances_sq_maj = np.einsum('ij,ij->i', left_maj, diff_values_maj)
    distances_maj = np.sqrt(np.maximum(distances_sq_maj, 0))
    
    # Añadir distancias al DataFrame de características ORIGINALES de la clase mayoritaria (para usar sus índices)
    data_features_with_dist = data_features_orig.copy() # Evitar SettingWithCopyWarning
    data_features_with_dist['Mahalanobis_Distance_Internal'] = distances_maj
    
    if len(data_features_with_dist) > n_minority_samples:
        selected_indices = data_features_with_dist.sort_values(
            by='Mahalanobis_Distance_Internal', ascending=False
        ).head(n_minority_samples).index
    else:
        selected_indices = data_features_with_dist.index
    
    return selected_indices


def meus_scenario1_apply(df_orig: pd.DataFrame, target_col_name='Clase', majority_class_label=0, minority_class_label=1) -> (pd.DataFrame, pd.Series):
    """
    Aplica MEUS al dataset completo (df_orig, que NO está escalado para modelado).
    Devuelve X_res, y_res en su escala original.
    """
    df = df_orig.copy()
    minority_samples = df[df[target_col_name] == minority_class_label]
    majority_samples_df_orig = df[df[target_col_name] == majority_class_label]
    
    n_minority = len(minority_samples)
    
    if n_minority == 0:
        logging.warning("MEUS Scen1: No hay muestras de la clase minoritaria. Devolviendo datos originales.")
        return df.drop(columns=[target_col_name]), df[target_col_name]
    if len(majority_samples_df_orig) == 0:
        logging.warning("MEUS Scen1: No hay muestras de la clase mayoritaria. Devolviendo datos originales.")
        return df.drop(columns=[target_col_name]), df[target_col_name]

    X_maj_features_orig = majority_samples_df_orig.drop(columns=[target_col_name])
    
    # Fallback si no hay features en la clase mayoritaria (ej. solo columna target)
    if X_maj_features_orig.empty:
        logging.warning("MEUS Scen1: Clase mayoritaria sin features. Submuestreo aleatorio.")
        if len(majority_samples_df_orig) > n_minority:
            selected_majority_indices = majority_samples_df_orig.sample(n=n_minority, random_state=42).index
        else:
            selected_majority_indices = majority_samples_df_orig.index
    else:
        selected_majority_indices = meus_resample_logic(X_maj_features_orig, n_minority, "Scen1")

    minority_indices = minority_samples.index
    balanced_indices = pd.Index(list(selected_majority_indices) + list(minority_indices))
    
    # Seleccionar del DataFrame ORIGINAL (df_orig)
    df_resampled = df_orig.loc[balanced_indices].sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_res = df_resampled.drop(columns=[target_col_name])
    y_res = df_resampled[target_col_name]
    
    logging.info(f"MEUS Scen1: Resampled. Majority selected: {len(selected_majority_indices)}, Minority: {len(minority_indices)}")
    return X_res, y_res


def meus_scenario2_apply(X_train_scaled_for_balance: pd.DataFrame, # X_train YA ESCALADO para modelado
                         y_train_orig: pd.Series,
                         majority_class_label=0, minority_class_label=1) -> (pd.DataFrame, pd.Series):
    """
    Aplica MEUS al conjunto de entrenamiento (X_train_scaled_for_balance, y_train_orig).
    X_train_scaled_for_balance ya está escalado para modelado.
    """
    X_train_current = X_train_scaled_for_balance.copy() # Datos ya escalados para modelado
    y_train = y_train_orig.copy()

    df_train_temp = X_train_current.assign(Clase_temp_target=y_train)
    minority_samples_train = df_train_temp[df_train_temp['Clase_temp_target'] == minority_class_label]
    majority_samples_train_df = df_train_temp[df_train_temp['Clase_temp_target'] == majority_class_label]
    
    n_minority_train = len(minority_samples_train)

    if n_minority_train == 0:
        logging.warning("MEUS Scen2: No hay muestras minoritarias en train. Devolviendo train original (escalado).")
        return X_train_current, y_train
    if len(majority_samples_train_df) == 0:
        logging.warning("MEUS Scen2: No hay muestras mayoritarias en train. Devolviendo train original (escalado).")
        return X_train_current, y_train

    # X_maj_features_train_scaled son las características de la clase mayoritaria, YA ESCALADAS para modelado
    X_maj_features_train_scaled = majority_samples_train_df.drop(columns=['Clase_temp_target'])

    # Fallback si no hay features en la clase mayoritaria
    if X_maj_features_train_scaled.empty:
        logging.warning("MEUS Scen2: Clase mayoritaria en train sin features. Submuestreo aleatorio.")
        if len(majority_samples_train_df) > n_minority_train:
            selected_majority_indices = majority_samples_train_df.sample(n=n_minority_train, random_state=42).index
        else:
            selected_majority_indices = majority_samples_train_df.index
    else:
        # meus_resample_logic recibirá datos ya escalados. Su escalado interno MinMax no cambiará los datos.
        selected_majority_indices = meus_resample_logic(X_maj_features_train_scaled, n_minority_train, "Scen2")

    minority_indices_train = minority_samples_train.index
    balanced_indices_train = pd.Index(list(selected_majority_indices) + list(minority_indices_train))
    
    # Seleccionar de df_train_temp (que contiene X_train_current y y_train)
    df_resampled_train = df_train_temp.loc[balanced_indices_train].sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_train_balanced_scaled = df_resampled_train.drop(columns=['Clase_temp_target'])
    y_train_balanced = df_resampled_train['Clase_temp_target']

    logging.info(f"MEUS Scen2: Resampled train. Majority selected: {len(selected_majority_indices)}, Minority: {len(minority_indices_train)}")
    return X_train_balanced_scaled, y_train_balanced

# -------------------------------------------------------------------
# Función principal de escenario (SOLO PARA MEUS con nueva lógica de escalado)
# -------------------------------------------------------------------
def run_scenario_meus_controlled_scaling(df_original_unscaled, scenario):
    target_col_name = 'Clase'
    X_original_unscaled = df_original_unscaled.drop(target_col_name, axis=1)
    y_original = df_original_unscaled[target_col_name]
    
    majority_class_label = 0 
    minority_class_label = 1

    X_train_final, X_test_final, y_train_final, y_test_final = [None]*4 # Inicializar

    logging.info(f"Usando MEUS (escalado interno para Mahalanobis, escalado de modelado controlado por escenario).")

    if scenario == 'scenario1':
        # 1. Balance en TODO el dataset (df_original_unscaled)
        #    meus_scenario1_apply devuelve X_res_unscaled, y_res
        logging.info(f">> ESCENARIO 1: MEUS en TODO el dataset (sin escalar) -> Split -> Escalado Separado")
        X_res_unscaled, y_res = meus_scenario1_apply(df_original_unscaled.copy(), 
                                                     target_col_name=target_col_name,
                                                     majority_class_label=majority_class_label,
                                                     minority_class_label=minority_class_label)
        logging.info(f"   MEUS Scen1: Tamaño después de resample (antes de split y escalar): {X_res_unscaled.shape}, Distribución y_res: {dict(y_res.value_counts())}")

        if X_res_unscaled.empty or y_res.empty:
            logging.error("   MEUS Scen1: X_res_unscaled o y_res vacíos después del balanceo. Abortando escenario.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

        # 2. Split del dataset balanceado (pero aún no escalado para modelado)
        stratify_param_s1 = y_res if not y_res.empty and len(y_res.unique()) > 1 else None
        X_train_raw, X_test_raw, y_train_final, y_test_final = train_test_split(
            X_res_unscaled, y_res, test_size=0.2, stratify=stratify_param_s1, random_state=42
        )
        logging.info(f"   MEUS Scen1: Después de Split. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

        # 3. Escalado SEPARADO para modelado
        if not X_train_raw.empty:
            scaler_s1 = MinMaxScaler()
            X_train_final = pd.DataFrame(scaler_s1.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            if not X_test_raw.empty:
                X_test_final = pd.DataFrame(scaler_s1.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
            else: X_test_final = pd.DataFrame() # Mantener vacío si raw estaba vacío
            logging.info(f"   MEUS Scen1: Después de Escalado. X_train_final: {X_train_final.shape}, X_test_final: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
        else: # Si X_train_raw es vacío
            X_train_final = pd.DataFrame()
            X_test_final = pd.DataFrame() # X_test_raw también sería vacío

    else: # scenario2
        # 1. Split del dataset ORIGINAL (df_original_unscaled)
        logging.info(f">> ESCENARIO 2: Split del dataset (sin escalar) -> Escalado Separado -> MEUS (solo en train escalado)")
        stratify_param_s2_initial = y_original if not y_original.empty and len(y_original.unique()) > 1 else None
        X_train_raw, X_test_raw, y_train_orig, y_test_final = train_test_split(
            X_original_unscaled, y_original, test_size=0.2, stratify=stratify_param_s2_initial, random_state=42
        )
        logging.info(f"   MEUS Scen2: Después de Split inicial. X_train_raw: {X_train_raw.shape}, X_test_raw: {X_test_raw.shape}")

        # 2. Escalado 
        X_train_scaled_for_meus = pd.DataFrame()
        X_test_final = pd.DataFrame()

        if not X_train_raw.empty:
            scaler_s2 = MinMaxScaler()
            X_train_scaled_for_meus = pd.DataFrame(scaler_s2.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            if not X_test_raw.empty:
                X_test_final = pd.DataFrame(scaler_s2.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
            logging.info(f"   MEUS Scen2: Después de Escalado. X_train_scaled_for_meus: {X_train_scaled_for_meus.shape}, X_test_final: {X_test_final.shape if not X_test_final.empty else '(empty)'}")
        
        # 3. MEUS solo en train (que ya está escalado para modelado)
        #    meus_scenario2_apply recibe X_train escalado y devuelve X_train_balanced también escalado.
        if not X_train_scaled_for_meus.empty and not y_train_orig.empty:
            X_train_final, y_train_final = meus_scenario2_apply(X_train_scaled_for_meus, y_train_orig,
                                                                majority_class_label=majority_class_label,
                                                                minority_class_label=minority_class_label)
            logging.info(f"   MEUS Scen2: Después de MEUS en train. X_train_final: {X_train_final.shape}, y_train_final dist: {dict(y_train_final.value_counts() if not y_train_final.empty else {})}")
        else:
            X_train_final = X_train_scaled_for_meus # Pasa el (posiblemente vacío) X_train escalado
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

# -------------------------------------------------------------------
# Funciones de entrenamiento y evaluación 
# -------------------------------------------------------------------
def train_and_save_models(X_train, y_train, exp_name, output_dir):
    hyper_file = os.path.join(output_dir, 'hyperparameters.txt')
    with open(hyper_file, 'w') as hf:
        hf.write(f"Hyperparameters for experiment {exp_name}\n")
        hf.write("="*60 + "\n\n")
    print(f"> Hyperparameters will be saved to: {hyper_file}")

    if X_train.empty or X_train.shape[1] == 0: # Chequeo adicional
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
            }
        ),

        'logreg': (
            LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            {'clf__penalty':['l1','l2'], 'clf__C':[0.001, 0.01, 0.7, 0.1, 0.2, 1, 10, 100, 1000], 'clf__solver':['liblinear']}
        ),
        'svm': (
            SVC(probability=True, random_state=42, class_weight='balanced'),
            {'clf__C': [0.5, 0.7, 0.9, 1, 1.5], 'clf__kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
        ),
        'rf': (
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            {'clf__n_estimators':[50, 100, 200], 'clf__max_depth':[None, 10, 20, 30], 'clf__min_samples_split': [2, 5, 10], 'clf__min_samples_leaf': [1, 2, 4]}
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
        ),

    }

    best_models = {}
    for name, (clf, param_grid) in specs.items():
        print(f"\n> Entrenando {name.upper()}...")
        logging.info(f"Entrenando {name.upper()}...")
        
        # El pipeline ahora aplica el escalado a X_train_final (que ya debería estar escalado).
        # MinMaxScaler en datos ya escalados [0,1] no los cambia, así que es seguro.
        # Si X_train_final NO estuviera escalado por alguna razón, este scaler lo haría.
        pipe = ImbPipeline([ 
            ('scaler', MinMaxScaler()), 
            ('clf', clf)
        ])
        
        y_train_processed = y_train.astype(int) 

        fit_params = {}
        if name == 'nn':
            fit_params['clf__callbacks'] = [
                ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, min_lr=1e-6, verbose=0),
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0) 
            ]
            fit_params['clf__validation_split'] = 0.1

        n_cv_splits = 4 
        if len(y_train_processed.unique()) > 1:
            min_class_count = min(y_train_processed.value_counts())
            n_cv_splits = min(2, min_class_count) 
            if min_class_count < 2:
                 logging.warning(f"Clase minoritaria en y_train para {name} tiene {min_class_count} muestras. CV no es posible. GridSearchCV podría fallar.")
                 n_cv_splits = 2 # Forzar a 2, pero esperar fallo o manejo por error_score
        
        if X_train.shape[0] < n_cv_splits: # Si hay menos muestras que folds
            logging.warning(f"No hay suficientes muestras en X_train ({X_train.shape[0]}) para CV con {n_cv_splits} splits en {name}. Saltando GridSearchCV.")
            best_params_str = "CV skipped (pocas muestras)"
            best_score_str = "N/A (CV skipped)"
            # Podrías entrenar un modelo por defecto aquí si quieres
            best_models[name] = None # Marcar como no entrenado
            with open(hyper_file, 'a') as hf: # Guardar estado
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
            if name == 'nn':
                grid.fit(X_train, y_train_processed, **fit_params)
            else:
                grid.fit(X_train, y_train_processed)
            
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
                # ... (código de guardado NN igual)
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
                # ... (código de guardado otros modelos igual)
                model_path = os.path.join(output_dir, f"{exp_name}_{name}_best_pipeline.joblib")
                joblib.dump(best_estimator_for_model, model_path)
                logging.info(f"  ✔ Modelo (pipeline) guardado: {model_path}")
                best_models[name] = best_estimator_for_model
        else:
            best_models[name] = None # Marcar explícitamente que no se entrenó/guardó
            logging.warning(f"No se pudo obtener best_estimator para {name} debido a un error previo.")

        with open(hyper_file, 'a') as hf:
            hf.write(f"{name.upper()} best params: {best_params_str}\n")
            hf.write(f"{name.upper()} best CV ROC-AUC: {best_score_str}\n\n")
        print(f"  ✔ Hiperparámetros (o estado de error) guardados para {name.upper()}")
        
    return {k: v for k, v in best_models.items() if v is not None} # Devolver solo modelos exitosos


def evaluate_and_save_reports(models, X_test, y_test, output_dir):
   
    report_file = os.path.join(output_dir, "classification_reports.txt")
    with open(report_file, "w") as rf:
        rf.write(f"Classification reports for {os.path.basename(output_dir)}\n")
        rf.write("=" * 60 + "\n\n")
    print(f"\n> Reports will be saved to: {report_file}")

    if X_test.empty or y_test.empty: # X_test puede ser vacío si X_train_raw era vacío
        logging.error("X_test o y_test están vacíos. No se pueden generar reportes.")
        print("X_test o y_test están vacíos. Saltando evaluación.")
        return

    y_test_processed = y_test.astype(int)

    for name, model_or_paths in models.items():
        # El modelo ya viene con su scaler si no es NN, o las rutas para NN
        print(f"\n> Evaluando {name.upper()}...")
        logging.info(f"Evaluando {name.upper()}...")
        y_pred, y_prob = None, None 

        try:
            if name == "nn":
                # model_or_paths son (scaler_path, keras_model_path)
                scaler_path, keras_model_path = model_or_paths
                scaler_loaded = joblib.load(scaler_path)
                keras_model = load_model(keras_model_path, compile=False)
                X_test_for_eval = scaler_loaded.transform(X_test) 
                
                y_prob_all_classes = keras_model.predict(X_test_for_eval)
                y_prob = y_prob_all_classes[:, 1]
                y_pred = np.argmax(y_prob_all_classes, axis=1)
            else: 
                # model_or_paths es el pipeline completo (scaler + clf)
                model_pipeline = model_or_paths
                # El pipeline se encargará de escalar X_test (ya debería estar escalado, pero es seguro)
                y_prob_all_classes = model_pipeline.predict_proba(X_test) 
                y_pred = model_pipeline.predict(X_test)
                
                if y_prob_all_classes.shape[1] == 2:
                    y_prob = y_prob_all_classes[:, 1]
                elif y_prob_all_classes.shape[1] == 1: 
                    clf_step = model_pipeline.named_steps.get('clf')
                    if clf_step and hasattr(clf_step, 'classes_') and len(clf_step.classes_) == 1:
                        if clf_step.classes_[0] == 1: y_prob = y_prob_all_classes[:, 0]
                        else: y_prob = 1.0 - y_prob_all_classes[:, 0]
                    else: y_prob = np.zeros(len(y_test_processed)) 
                else: y_prob = y_prob_all_classes[:, -1]

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
                        logging.error(f"Error al calcular ROC-AUC para {name}: {e_auc}. y_test unique: {np.unique(y_test_processed)}, y_prob unique (first 5): {np.unique(y_prob[:5]) if len(y_prob)>0 else 'empty'}")
                
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
    CSV_PATH    = "C:/Users/saave/Desktop/Master_Thesis/Credit_card_data/creditcard.csv" 
    BASE_OUTPUT = "C:/Users/saave/Desktop/data_balance/Resultados_MEUS"
    
    for scenario in ["scenario1", "scenario2"]:
        exp_name   = f"MEUS_MajClass_Scaling_{scenario}" 
        output_dir = os.path.join(BASE_OUTPUT, exp_name)
        log_file   = os.path.join(output_dir, f"run_{exp_name}.log")

        os.makedirs(output_dir, exist_ok=True)
        setup_logging(log_file)
        
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando experimento: {exp_name} ===")
        logging.info(f"Técnica: MEUS (Mahalanobis Undersampling en Clase Mayoritaria con Escalado Controlado)")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"=================================================")
        print(f"\n=== Iniciando experimento: {exp_name} ===")

        df_original = load_and_prepare(CSV_PATH) # Carga datos SIN escalar
        logging.info(f"Dataset cargado (sin escalar). Forma: {df_original.shape}. Clases: {dict(df_original['Clase'].value_counts())}")

        if df_original.shape[0] < 10: 
            logging.error(f"Dataset con muy pocas filas ({df_original.shape[0]}). Abortando {exp_name}.")
            continue
        class_counts = df_original['Clase'].value_counts()
        if len(class_counts) < 2 or class_counts.get(0,0) == 0 or class_counts.get(1,0) == 0:
            logging.error(f"Dataset no tiene ambas clases o una clase está vacía: {class_counts.to_dict()}. Abortando {exp_name}.")
            continue

        X_train, X_test, y_train, y_test = run_scenario_meus_controlled_scaling(df_original.copy(), scenario)

        if X_train.empty or y_train.empty: # X_train podría ser vacío si X_res_unscaled es vacío
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
            # Asegurarse de que X_test no sea None o vacío antes de evaluar
            if X_test is not None and not X_test.empty and y_test is not None and not y_test.empty:
                 evaluate_and_save_reports(trained_models, X_test, y_test, output_dir)
            else:
                logging.warning(f"X_test o y_test están vacíos para {exp_name}. Saltando evaluación.")
        
        logging.info(f"=== Fin experimento {exp_name} ===\n")
        print(f"=== Fin experimento: {exp_name} ===\n")

    print("Todos los experimentos MEUS con escalado controlado han finalizado.")