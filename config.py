"""戦略パラメータの一元管理。調整はここだけで完結する。"""
import os

# ── パス ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.getenv("STOCK_ALERT_HOME", PROJECT_DIR)
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.expanduser("~/stock-alert")

# ── 相場判定 ──────────────────────────────────────────────────────────────
BEAR_MARKET_THRESHOLD = -5.0    # 日経20日リターンがこれ以下で下落相場と判定

# ── モデル出力 ─────────────────────────────────────────────────────────────
FORECAST        = 63            # 予測ホライズン（営業日）
RISE_THRESHOLD  = 15.0          # 上昇判定の閾値（%）
NET_SELL_THRESHOLD = -5.0       # 保有株の売りシグナル閾値

# ── 新規候補フィルター ─────────────────────────────────────────────────────
NEW_CANDIDATE_NET_MIN        = 8.0   # ネットスコア下限
NEW_CANDIDATE_NET_MAX        = 13.0  # ネットスコア上限（過熱回避）
CANDIDATE_DROP_PROB_MAX      = 10.0  # 下落確率上限（【回避】ライン）
CANDIDATE_EARNINGS_SKIP_DAYS = 7     # 決算N日以内の新規候補を除外
