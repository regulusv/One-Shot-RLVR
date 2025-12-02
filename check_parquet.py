import pandas as pd
try:
    df = pd.read_parquet('data/train/one_shot_rlvr/pi1_r128.parquet')
    print("Columns:", df.columns)
    if 'data_source' in df.columns:
        print("Data Sources:", df['data_source'].unique())
    else:
        print("No data_source column")
except Exception as e:
    print(e)

