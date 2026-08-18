"""戦略パラメータの一元管理。調整はここだけで完結する。"""
import os

# ── パス ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.getenv("STOCK_ALERT_HOME", PROJECT_DIR)
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.expanduser("~/stock-alert")

# ── 相場判定 ──────────────────────────────────────────────────────────────
BEAR_MARKET_THRESHOLD = -5.0    # 日経20日リターンがこれ以下で下落相場と判定

# ── 市場タイミングフィルター（マーケットタイミング）──────────────────────────
# 暴落時の損失を避け、長期複利を守る
MARKET_TIMING_20D_THRESH = -3.0   # 日経20日リターンがこれ以下ならシグナル停止（急落キャッチ用）
