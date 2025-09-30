import os
import logging
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

# Parámetros globales
CSV_PATH        = "C:/Users/saave/Desktop/Master_Thesis/Credit_card_data/creditcard.csv"
TARGET_COLUMN   = "Class"
RANDOM_SEED     = 42
OUTPUT_DIR      = "C:/Users/saave/Desktop/data_balance/FIGURAS/"
TSNE_BALANCED   = os.path.join(OUTPUT_DIR, "tSNE_MEUS.png")


# Paso 2: Función MEUS (matching 1-a-1 real con Mahalanobis)
def apply_meus(X: pd.DataFrame, y: pd.Series, seed: int):
    data = X.copy()
    data[y.name] = y

    df_min = data[data[y.name] == 1].drop(columns=[y.name])
    df_maj = data[data[y.name] == 0].drop(columns=[y.name])
    if df_min.empty or df_maj.empty:
        logging.warning("Clases insuficientes; retorno original.")
        return X, y

    # 2.1 Calcular Sigma⁻¹
    cov     = np.cov(X.values, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    # 2.2 Ajustar K al tamaño de la mayoría
    K = min(150, len(df_maj))
    nn = NearestNeighbors(n_neighbors=K,
                          metric='mahalanobis',
                          metric_params={'VI': inv_cov})
    nn.fit(df_maj.values)

    # 2.3 Buscar vecinos para cada minoría
    _, neigh_idx = nn.kneighbors(df_min.values)

    # 2.4 Claiming 1-a-1
    claimed  = set()
    selected = []
    for i in range(len(df_min)):
        for j in neigh_idx[i]:
            idx = df_maj.index[j]
            if idx not in claimed:
                claimed.add(idx)
                selected.append(idx)
                break

    df_maj_sel = df_maj.loc[selected]

    # 2.5 Reconstruir y barajar
    X_res = pd.concat([df_min, df_maj_sel], axis=0)
    y_res = pd.Series([1]*len(df_min) + [0]*len(df_maj_sel),
                      name=y.name, index=X_res.index)

    np.random.seed(seed)
    perm    = np.random.permutation(X_res.index)
    X_res   = X_res.loc[perm].reset_index(drop=True)
    y_res   = y_res.loc[perm].reset_index(drop=True)

    logging.info(f"MEUS → minoría: {y_res.sum()}, mayoría: {len(y_res)-y_res.sum()}")
    return X_res, y_res


# Paso 3: Función de visualización t-SNE (solo balanced, sin título)
def plot_tsne_balanced(X: pd.DataFrame, y: pd.Series, out_path: str):
    if X.empty or y.empty:
        logging.error("No hay datos para plotear.")
        return

    # 3.1 Cálculo t-SNE
    start = time.time()
    tsne  = TSNE(n_components=2,
                 perplexity=25,
                 n_iter=1500,
                 random_state=RANDOM_SEED,
                 verbose=1)
    coords = tsne.fit_transform(X)
    logging.info(f"t-SNE completado en {time.time() - start:.2f}s")

    # 3.2 Dibujar scatter
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap   = {0: '#0072B2', 1:'#D55E00' }
    labels = {0: 'Majority Class (Non-Fraud)', 1: 'Minority Class (Fraud)'}

    for cls in [0, 1]:
        mask = (y.values == cls)
        ax.scatter(coords[mask, 0],
                   coords[mask, 1],
                   c=cmap[cls],
                   label=labels[cls],
                   alpha=0.8,
                   edgecolors='w',
                   s=50)

    # 3.3 Etiquetas de ejes en inglés y con fuente grande
    ax.set_xlabel('First t-SNE Dimension', fontsize=21)
    ax.set_ylabel('Second t-SNE Dimension', fontsize=21)
    # NO title
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.legend(title="Classes", fontsize=14, title_fontsize=15,
              frameon=True, facecolor='white', framealpha=1.0)
    ax.grid(True, linestyle='-', alpha=0.6)

    # 3.4 Guardar figura
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    logging.info(f"Figura guardada en: {out_path}")

    plt.show()
    plt.close(fig)


# Paso 4: Bloque principal
if __name__ == "__main__":
    # 4.1 Carga y split
    df         = pd.read_csv(CSV_PATH)
    X, y       = df.drop(TARGET_COLUMN, axis=1), df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED
    )

    # 4.2 Escalado
    scaler         = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # 4.3 Submuestreo MEUS
    logging.info("Aplicando MEUS para balancear clases…")
    X_bal, y_bal = apply_meus(X_train_scaled, y_train, seed=RANDOM_SEED)

    # 4.4 Visualizar SOLO el conjunto balanceado
    logging.info("Generando t-SNE de datos balanceados…")
    plot_tsne_balanced(X_bal, y_bal, TSNE_BALANCED)

    logging.info("Fin del proceso.")