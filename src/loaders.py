import pandas as pd
from typing import List, Optional, Dict
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging
import pandas as pd

def load_top_clients(path: str) -> pd.DataFrame:
    """Lee ranking de clientes y comercial asociado."""
    df = pd.read_excel(path)
    return df[['cliente', 'ranking', 'comercial']].dropna()

def load_operations_data(paths: List[str]) -> pd.DataFrame:
    """Concatena archivos Excel donde cada fila es una operación."""
    dfs = [pd.read_excel(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def get_gsheet_client(creds_json_path: str):
    """
    Crea un cliente de gspread a partir de un JSON de service account.
    """
    scope = [
        'https://www.googleapis.com/auth/spreadsheets.readonly',
        'https://www.googleapis.com/auth/drive.readonly'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json_path, scope)
    return gspread.authorize(creds)

def load_all_clients_from_sheets(
    sheet_ids: List[str],
    creds_json_path: str,
    *,
    date_col: str = "date",   # nombre de la columna con la fecha (ej. "time")
    year: int = 2025          # año que quieres conservar
) -> pd.DataFrame:
    """
    Lee la(s) Google Sheet(s) indicadas, conserva solo las filas cuyo año == `year`
    y devuelve un DataFrame con una sola columna 'client'.

    - `date_col` debe contener fechas en formato reconocible por pandas.
    - Lanza RuntimeError si alguna hoja no tiene 'client' o `date_col`.
    """
    client = get_gsheet_client(creds_json_path)
    parts: list[pd.DataFrame] = []
    failed: list[str] = []

    for sid in sheet_ids:
        try:
            ss = client.open_by_key(sid)
            ws = ss.get_worksheet(0)
            df = pd.DataFrame(ws.get_all_records())
            df.columns = df.columns.str.strip().str.lower()
        except Exception as e:
            logging.error(f"Error leyendo sheet {sid}: {e}")
            failed.append(sid)
            continue

        # Validaciones
        if "client" not in df.columns:
            logging.error(f"La hoja {sid} no contiene columna 'client'")
            failed.append(sid)
            continue
        if date_col.lower() not in df.columns:
            logging.error(f"La hoja {sid} no contiene columna '{date_col}'")
            failed.append(sid)
            continue

        df[date_col] = pd.to_datetime(df[date_col],
                                format="%d/%m/%Y",
                                errors="coerce")

        mask_2025 = df[date_col].dt.year == year
        filtered = df.loc[mask_2025, ["client"]]

        parts.append(filtered)

    if failed:
        raise RuntimeError(f"No se pudieron procesar estas sheets: {failed}")

    return pd.concat(parts, ignore_index=True)

def compute_proportion_map_from_sheets(
    sheet_ids: List[str],
    creds_json_path: str
) -> Dict[str, float]:
    """
    Lee todas las sheets y construye un dict cliente→proporción,
    aplicando estandarización de nombre de columna ('client' → 'cliente').
    Reporta cualquier sheet_id con error de apertura.
    """
    client = get_gsheet_client(creds_json_path)
    all_clients: List[pd.DataFrame] = []
    failed_ids: List[str] = []

    for sid in sheet_ids:
        try:
            ss = client.open_by_key(sid)
        except Exception as e:
            logging.error(f"No se pudo abrir la hoja con ID {sid}: {e}")
            failed_ids.append(sid)
            continue

        ws = ss.get_worksheet(0)
        df = pd.DataFrame(ws.get_all_records())
        df.columns = df.columns.str.strip().str.lower()
        if 'client' in df.columns:
            df = df.rename(columns={'client': 'cliente'})

        if 'cliente' not in df.columns:
            logging.error(f"La hoja {sid} no contiene columna 'cliente' ni 'client'")
            failed_ids.append(sid)
            continue

        all_clients.append(df[['cliente']])

    if failed_ids:
        raise PermissionError(
            f"Error al leer estas hojas (permiso o columna faltante): {failed_ids}"
        )

    df_all = pd.concat(all_clients, ignore_index=True)

    return df_all['cliente'].value_counts(normalize=True).to_dict()