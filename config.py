"""戦略パラメータの一元管理。調整はここだけで完結する。"""
import os

# ── パス ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.getenv("STOCK_ALERT_HOME", PROJECT_DIR)
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.expanduser("~/stock-alert")

# ── 相場判定 ──────────────────────────────────────────────────────────────
BEAR_MARKET_THRESHOLD = -5.0    # 日経20日リターンがこれ以下で下落相場と判定
HOT_MARKET_THRESHOLD  = 15.0   # 日経60日リターンがこれ以上で急騰相場と判定（新規見送り推奨）

# ── 市場タイミングフィルター（マーケットタイミング）──────────────────────────
# 日経225が63日SMAを下回っているときはシグナルを出さない
# 暴落時の損失を避け、長期複利を守る
MARKET_TIMING_ENABLED    = True   # True: 下落相場でシグナル停止
MARKET_TIMING_SMA_DAYS   = 63     # 何日SMAと比較するか
MARKET_TIMING_20D_THRESH = -3.0   # 日経20日リターンがこれ以下でも停止（急落キャッチ用）

# ── スクリーナー条件（screener.py / backtest.py / rf_train_v3.py と同値に保つ） ──
SCREENER_MOM_3M_MIN   =  8.0   # 3ヶ月モメンタム下限（5.0→8.0: 10期間BTで勝率+7pp）
SCREENER_MOM_3M_MAX   = 30.0
SCREENER_MOM_20D_MIN  =  0.0   # 20日モメンタム下限（-3.0→0.0: avg+1%改善）
SCREENER_VOL_MIN      = 22.0   # 年率ボラ下限（20.0→22.0）
SCREENER_VOL_MAX      = 50.0
SCREENER_RSI_MIN      = 45.0   # RSI下限（40→45: 勝率+4pp）
SCREENER_RSI_MAX      = 70.0
