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

# ── モデル出力 ─────────────────────────────────────────────────────────────
FORECAST        = 63            # 予測ホライズン（営業日）
RISE_THRESHOLD  = 15.0          # 上昇判定の閾値（%）
# 保有株売りシグナル閾値（下降シグナル=net<-10に統一。バックテスト75%勝率+9.4%）
SELL_DAYS_MID           =  3    # 使用継続（将来拡張用）
SELL_DAYS_LATE          = 63    # 使用継続（将来拡張用）
NET_SELL_THRESHOLD_MID  = -10.0  # 下降シグナル基準（net<-10）
NET_SELL_THRESHOLD_LATE = -10.0  # 同上

# ── スクリーナー条件（screener.py / backtest.py / rf_train_v3.py と同値に保つ） ──
SCREENER_MOM_3M_MIN   =  8.0   # 3ヶ月モメンタム下限（5.0→8.0: 10期間BTで勝率+7pp）
SCREENER_MOM_3M_MAX   = 30.0
SCREENER_MOM_20D_MIN  =  0.0   # 20日モメンタム下限（-3.0→0.0: avg+1%改善）
SCREENER_VOL_MIN      = 22.0   # 年率ボラ下限（20.0→22.0）
SCREENER_VOL_MAX      = 50.0
SCREENER_RSI_MIN      = 45.0   # RSI下限（40→45: 勝率+4pp）
SCREENER_RSI_MAX      = 70.0

# ── 新規候補フィルター ─────────────────────────────────────────────────────
NEW_CANDIDATE_NET_MIN        = 10.0  # ネットスコア下限（8.0→10.0: 弱い帯を除外）
NEW_CANDIDATE_NET_MAX        = 13.0  # ネットスコア上限（過熱回避）
CANDIDATE_DROP_PROB_MAX      =  8.0  # 下落確率上限（12.0→8.0: 高リスク帯を除外）
CANDIDATE_EARNINGS_SKIP_DAYS = 7     # 決算N日以内の新規候補を除外
# conflict除外: net高くてdrop_probも無視できない「矛盾シグナル」を除外
CANDIDATE_CONFLICT_NET_MIN   = 10.0  # net がこれ以上かつ
CANDIDATE_CONFLICT_DROP_MIN  =  5.0  # drop_prob がこれ以上なら候補から除外
