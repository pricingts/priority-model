import pandas as pd
from typing import Mapping
from src.standarization import standardize_prop_clients
from typing import Dict
import numpy as np

def apply_weight_mapping(
    df: pd.DataFrame,
    key_col: str,
    mapping: Mapping[str, float],
    weight_col: str
) -> pd.DataFrame:
    """Asigna un peso basado en un dict de mapeo, rellenando 0 si falta clave."""
    df[weight_col] = df[key_col].map(mapping).fillna(0.0)
    return df

def compute_proportion_map(
    df_all_clients: pd.DataFrame,
    df_clients_original: pd.DataFrame,
    threshold: int = 65,
    method: str = None,     
    beta: float = 1.0        
) -> Dict[str, float]:

    df = df_all_clients.rename(columns={'client': 'cliente'})
    df_std = standardize_prop_clients(
        df, df_clients_original,
        prop_col='cliente', orig_col='Cliente',
        threshold=threshold
    ).dropna(subset=['cliente_std'])

    # 2) Cálculo de count por cliente
    df_counts = (
        df_std
        .groupby('cliente_std')
        .size()
        .reset_index(name='count')
    )

    if df_counts.empty:
        return {}

    max_count = df_counts['count'].max()

    # 3) Transformación según método
    if method == 'proportion':
        df_counts['raw_score'] = df_counts['count']
    elif method == 'power':
        df_counts['raw_score'] = df_counts['count'] ** beta
    elif method == 'relative_power':
        df_counts['raw_score'] = (df_counts['count'] / max_count) ** beta
    elif method == 'log':
        df_counts['raw_score'] = np.log1p(df_counts['count'])
    else:
        raise ValueError(f"Método desconocido: {method!r}")

    # total = df_counts['raw_score'].sum()
    # df_counts['prop_weight'] = df_counts['raw_score'] / total

    return df_counts.set_index('cliente_std')['raw_score'].to_dict()


def compute_client_map(
    df_clients: pd.DataFrame,
    client_col: str = 'cliente',
    rank_col: str = 'ranking',
    alpha: float = None
) -> Dict[str, float]:

    df = (
        df_clients[[client_col, rank_col]]
        .dropna()
        .assign(**{rank_col: pd.to_numeric(df_clients[rank_col], errors='coerce')})
        .dropna(subset=[rank_col])
        .copy()
    )

    N = len(df)
    ranks = df[rank_col].values

    df['weight'] = np.exp(-alpha * (ranks - 1)/N)

    return dict(zip(df[client_col], df['weight']))