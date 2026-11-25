# Settings: change fees, trading thresholds, LSTM windows, #CV folds
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

SEED = 42          #reproducible seed --> can be changed with each run
FEE = 0.001        #0.1% charged when position changes
THRESHOLD = 0.5    #go long if P(up) > THRESHOLD, else hold cash
WINDOW = 14        #LSTM look-back in days
CV_SPLITS = 5      #walk-forward folds

#Raw price levels are non-stationary ($3k --> $125k); replaced by ratios in data.make_X
LEVEL_COLS = ["open", "high", "low", "close", "ma_7d", "ma_30d"]
#Heavy right-tailed on-chain levels; log-transformed
LOG_COLS = ["active_addresses", "transaction_count", "hash_rate", "transaction_fees"]