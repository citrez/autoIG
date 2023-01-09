from pathlib import Path
import pandas as pd
from autoIG.utils import TMP_DIR
pd.DataFrame().to_csv(TMP_DIR/"raw_stream.csv",index=False,header=False)
pd.DataFrame().to_csv(TMP_DIR/"sold.csv",index=False,header=False)
pd.DataFrame().to_csv(TMP_DIR/"position_metrics.csv",index=False,header=False)
Path(TMP_DIR/'autoIG.sqlite').unlink()
(TMP_DIR/'autoIG.sqlite').touch()
print('Whiped temporary data')