from pathlib import Path
import pandas as pd
from src.prioridad.utils               import load_config
from src.prioridad.loaders            import load_all_clients_from_sheets
from src.prioridad.transformers       import compute_client_map, compute_proportion_map
from src.prioridad.priority_calculator import PriorityCalculator
import matplotlib.pyplot as plt

def main():
    # ─── Parámetros de entorno ────────────────────────────────────────────────
    config_path   = Path('config/weights.yaml')
    clients_file  = Path('data/top_profit.xlsx')
    original_csv  = Path('Duracion Envio Solicitudes - clientes.csv')
    creds_json    = 'credentials.json'
    sheet_ids     = [
        '1vFf06EI_0PZcVeqN0ovUtWe-pyD65mEO9MfeQKZFEOQ',  # Sharon
        '1ntDsrnH3LR3XW2pj_tl76pPsoqY5IW_4SL4VF7gJrvk',  # Pipe
        '1-GXJdHoutTKU9Jqrh0NoLizAQAc3_Yw8ijsmrM5fTr4',  # Johnny
        '17Y7R6GoPrxH4mS3Mi4B1HoVktZpX3uleeRdj8215xU4',  # Andres
        '18O47Xkdnpoco069nChTrs7XrGr-UjL_UZ-DMsqTBhfs',  # Irina
        '1kpLuMOHRp3gK32EoUPxV_AzV_EYF8y8Np0U_JjOlNPM',  # Ivan
        '1GkOHm_IP3pBnTt0yb7bdJIEgzJENvu5965RK-wSonC8',  # Pedro
        '1-os5Ig5UzKmt8S0NP0qVwyiJWFn9BeqkbKU5GTy23Yo',  # Jorge y Steph
    ]

    # ─── 1) Cargo configuración y pesos básicos ───────────────────────────────
    cfg = load_config(config_path)
    w1, w2, w3   = cfg['weights']['w1'], cfg['weights']['w2'], cfg['weights']['w3']
    inc_map      = cfg['incoterm_weights']
    mod_map      = cfg['modality_weights']
    orig_map     = cfg['origin_weights']
    dest_map     = cfg['destination_weights']

    df_clients   = pd.read_excel(clients_file)
    df_original  = pd.read_csv(original_csv)

    df_all   = load_all_clients_from_sheets(sheet_ids, creds_json, date_col="date",  year=2025)
    prop_map = compute_proportion_map(df_all, df_original, threshold=65, method='relative_power', beta=2.0)

    test_requests = [
        {
            'cliente':     'BATERIAS WILLARD S.A',
            'incoterm':    'DDP',
            'modalidad':   'FCL',
            'origin':      'Asia',
            'destination': 'Asia'
        },
        {
            'cliente':     'GELCO SAS',
            'incoterm':    'DDP',
            'modalidad':   'FCL',
            'origin':      'Asia',
            'destination': 'Asia'
        },
        {
            'cliente':     'ROYCE CORPORATION INC',
            'incoterm':    'DDP',
            'modalidad':   'FCL',
            'origin':      'Asia',
            'destination': 'Asia'
        },
        {
            'cliente':     'AMI TRADING USA INC.',
            'incoterm':    'DDP',
            'modalidad':   'FCL',
            'origin':      'Asia',
            'destination': 'Asia'
        },
    ]
    df_prop = (
    pd.DataFrame.from_dict(prop_map, orient='index', columns=['proportion'])
        .rename_axis('cliente')
        .reset_index()
        .sort_values('proportion', ascending=False)   # ↓ orden descendente
        .reset_index(drop=True)                       # opcional: reindexa 0..n-1
)

    print(df_prop)

    #clients_to_check = [r['cliente'] for r in test_requests]
    #df_test = df_prop[df_prop['cliente'].isin(clients_to_check)]
    #print(df_prop['cliente'].unique())

    # ─── 5) Pruebo varios alphas y acumulo resultados ─────────────────────────
    alphas      = [4.0]
    all_results = []

    for alpha in alphas:
        # 5.1) Mapa de clientes con decaimiento exponencial
        client_map = compute_client_map(df_clients, alpha=alpha)

        calc = PriorityCalculator(
            client_map, prop_map,
            inc_map, mod_map, orig_map, dest_map,
            w1, w2, w3
        )

        # 5.3) Aplico el modelo a cada solicitud
        df_scored = pd.DataFrame(test_requests)
        df_scored['priority_score'] = df_scored.apply(
            lambda row: calc.calculate(
                row['cliente'],
                row['incoterm'],
                row['modalidad'],
                row['origin'],
                row['destination']
            ),
            axis=1
        )
        df_scored['alpha'] = alpha
        all_results.append(df_scored)

    # ─── 6) Uno resultados y pivoteo para comparar ────────────────────────────
    df_all_alphas = pd.concat(all_results, ignore_index=True)
    pivot = df_all_alphas.pivot_table(
        index='cliente',
        columns='alpha',
        values='priority_score'
    )

    print("\nComparativa de priority_score según alpha:\n")
    print(pivot.to_string())

if __name__ == '__main__':
    main()