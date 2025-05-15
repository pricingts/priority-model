import pandas as pd
import re
from unidecode import unidecode
from rapidfuzz import process, fuzz
from typing import Dict, Any

def clean_name(s: str) -> str:
    """Quita tildes, pasa a minúsculas, elimina no-alfa numéricos y unifica espacios."""
    s = unidecode(str(s))
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return re.sub(r'\s+', ' ', s)

def clean_name(s: str) -> str:
    """Quita tildes, pasa a minúsculas, elimina no-alfa numéricos y unifica espacios."""
    s = unidecode(str(s))
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return re.sub(r'\s+', ' ', s)

def standardize_prop_clients(
    df_prop:    pd.DataFrame,
    df_original: pd.DataFrame,
    prop_col:   str = 'cliente',
    orig_col:   str = 'cliente',
    threshold:  int = 65
) -> pd.DataFrame:
    """
    Añade a df_prop columnas de estandarización y score:
    - 'cliente_std': nombre estandarizado o NaN si debajo de threshold.
    - 'match_key': clave limpia sugerida.
    - 'score': similitud WRatio.
    Permite revisar manualmente cada caso.
    """
    # Copiar datos
    df_prop = df_prop.copy()
    df_original = df_original.copy()

    # 1) Limpia nombres
    df_prop['_key']     = df_prop[prop_col].map(clean_name)
    df_original['_key'] = df_original[orig_col].map(clean_name)

    # 2) Merge exacto
    merged = df_prop.merge(
        df_original[['_key', orig_col]],
        on='_key', how='left', suffixes=('', '_orig')
    )
    merged['cliente_std'] = merged[orig_col]

    # 3) Encuentra no-matches exactos
    mask = merged['cliente_std'].isna()
    if mask.any():
        choices = df_original['_key'].tolist()
        # Extrae mejor match y score para cada clave
        results = merged.loc[mask, '_key'].map(
            lambda k: process.extractOne(k, choices, scorer=fuzz.WRatio) or (None, 0, None)
        )
        # Desempaqueta en DataFrame temporal
        tmp = pd.DataFrame(results.tolist(), index=merged[mask].index)
        tmp.columns = ['match_key', 'score', '_extra']
        merged = merged.join(tmp)

        # 4) Lookup único por _key
        lookup: Dict[str, Any] = (
            df_original
            .drop_duplicates('_key')
            .set_index('_key')[orig_col]
            .to_dict()
        )
        # Asigna nombre estandarizado sugerido
        merged.loc[mask, 'cliente_std'] = merged.loc[mask, 'match_key'].map(lookup)

        # 5) Vacía donde score < threshold
        low_conf = merged['score'] < threshold
        merged.loc[low_conf, 'cliente_std'] = pd.NA

    # 6) Limpia solo la _extra
    merged = merged.drop(columns=['_extra'])

    return merged
