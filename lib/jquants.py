"""
lib/jquants.py
J-Quants API v2 経由でデータ取得し stock_alert.db に保存するモジュール。

必要なもの:
  - 環境変数 JQUANTS_API_KEY（または ~/.jquants-api/jquants-api.toml）
  - pip install jquants-api-client
  - J-Quants Standard プラン以上（信用取引・空売りデータに必要）

提供する関数:
  fetch_margin_history(start, end)   → kabutan_jquants_margin テーブルに保存
  fetch_short_history(start, end)    → short_interest テーブルに保存
  get_margin_ratio(code, date)       → float | None
  get_short_balance(code, date)      → float | None
"""

import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# クライアント生成
# ------------------------------------------------------------------ #

def _get_client():
    """ClientV2 を返す。APIキー未設定なら ValueError。"""
    try:
        from jquantsapi import ClientV2
    except ImportError:
        raise ImportError("jquants-api-client 未インストール: pip install jquants-api-client")

    api_key = os.environ.get("JQUANTS_API_KEY", "")
    if not api_key:
        raise ValueError(
            "JQUANTS_API_KEY 環境変数が未設定です。"
            "J-Quants Standard プランのAPIキーを設定してください。"
        )
    return ClientV2(api_key=api_key)


# ------------------------------------------------------------------ #
# 信用残高（週次）fetch → DB 保存
# ------------------------------------------------------------------ #

def fetch_margin_history(
    start: str = "20200101",
    end: Optional[str] = None,
    codes: Optional[list] = None,
) -> int:
    """
    信用取引週末残高を J-Quants から一括取得して margin_data テーブルへ保存。

    Args:
        start: 取得開始日 YYYYMMDD（デフォルト 2020-01-01）
        end:   取得終了日 YYYYMMDD（デフォルト 今日）
        codes: 対象銘柄リスト。None なら全銘柄。

    Returns:
        保存行数
    """
    from lib.db import upsert_margin_data

    if end is None:
        end = date.today().strftime("%Y%m%d")

    cli = _get_client()
    logger.info(f"信用残高取得: {start} → {end}")

    try:
        df = cli.get_mkt_margin_interest_range(start_dt=start, end_dt=end)
    except Exception as e:
        logger.error(f"get_mkt_margin_interest_range 失敗: {e}")
        raise

    if df is None or df.empty:
        logger.warning("信用残高データが空でした")
        return 0

    # カラム名の正規化（J-Quants v2 の実際のカラム名に合わせる）
    df.columns = [c.strip() for c in df.columns]
    logger.info(f"取得カラム: {list(df.columns)}")
    logger.info(f"行数: {len(df)}")

    # 必要なカラムを特定
    # 例: Code, Date, LongSelling(買い残), ShortSelling(売り残), Ratio(倍率) など
    code_col  = _find_col(df, ['Code', 'code', 'コード'])
    date_col  = _find_col(df, ['Date', 'date', 'PublishedDate', '日付'])
    buy_col   = _find_col(df, ['LongSelling', 'BuyBalance', 'buy_balance', '信用買残', '買い残'])
    sell_col  = _find_col(df, ['ShortSelling', 'SellBalance', 'sell_balance', '信用売残', '売り残'])
    ratio_col = _find_col(df, ['Ratio', 'ratio', '倍率', 'CreditRatio'])

    if not code_col or not date_col:
        logger.error(f"必要なカラムが見つかりません: {list(df.columns)}")
        return 0

    saved = 0
    for _, row in df.iterrows():
        try:
            code = str(row[code_col]).strip().zfill(4)
            if codes and code not in codes:
                continue
            week_date = _normalize_date(row[date_col])
            buy  = float(row[buy_col])  if buy_col  and pd.notna(row[buy_col])  else None
            sell = float(row[sell_col]) if sell_col and pd.notna(row[sell_col]) else None
            ratio = float(row[ratio_col]) if ratio_col and pd.notna(row[ratio_col]) else None
            if ratio is None and buy and sell and sell > 0:
                ratio = buy / sell
            upsert_margin_data(code, week_date, buy, sell, ratio)
            saved += 1
        except Exception:
            continue

    logger.info(f"信用残高 保存完了: {saved}行")
    return saved


# ------------------------------------------------------------------ #
# 空売り残高 fetch → DB 保存
# ------------------------------------------------------------------ #

def fetch_short_history(
    start: str = "20200101",
    end: Optional[str] = None,
    codes: Optional[list] = None,
) -> int:
    """
    空売り残高報告を J-Quants から一括取得して jpx_short_interest テーブルへ保存。

    Returns:
        保存行数
    """
    from lib.db import bulk_upsert_short_interest

    if end is None:
        end = date.today().strftime("%Y%m%d")

    cli = _get_client()
    logger.info(f"空売り残高取得: {start} → {end}")

    try:
        df = cli.get_mkt_short_sale_report_range(start_dt=start, end_dt=end)
    except Exception as e:
        logger.error(f"get_mkt_short_sale_report_range 失敗: {e}")
        raise

    if df is None or df.empty:
        logger.warning("空売り残高データが空でした")
        return 0

    df.columns = [c.strip() for c in df.columns]
    logger.info(f"取得カラム: {list(df.columns)}")
    logger.info(f"行数: {len(df)}")

    code_col    = _find_col(df, ['Code', 'code', 'コード'])
    date_col    = _find_col(df, ['DisclosedDate', 'Date', 'date', '開示日', 'PublishedDate'])
    bal_col     = _find_col(df, ['ShortBalance', 'short_balance', '残高株数', 'ShortSelling'])
    amount_col  = _find_col(df, ['ShortAmount', 'short_amount', '残高金額', 'Amount'])

    if not code_col or not date_col:
        logger.error(f"必要なカラムが見つかりません: {list(df.columns)}")
        return 0

    rows = []
    for _, row in df.iterrows():
        try:
            code = str(row[code_col]).strip().zfill(4)
            if codes and code not in codes:
                continue
            week_date = _normalize_date(row[date_col])
            bal = float(row[bal_col]) if bal_col and pd.notna(row[bal_col]) else None
            amt = float(row[amount_col]) if amount_col and pd.notna(row[amount_col]) else None
            if bal is not None:
                rows.append((code, week_date, bal, amt))
        except Exception:
            continue

    if rows:
        bulk_upsert_short_interest(rows)

    logger.info(f"空売り残高 保存完了: {len(rows)}行")
    return len(rows)


# ------------------------------------------------------------------ #
# ライブ取得ヘルパー（rank_stocks.py / backtest.py から呼ばれる）
# ------------------------------------------------------------------ #

def get_margin_ratio(code: str, as_of: date) -> Optional[float]:
    """
    指定日時点での信用倍率を DB から取得。
    直近5週以内の最新値を返す（weekly データなので）。
    """
    from lib.db import get_margin_ratio_at
    return get_margin_ratio_at(code, as_of.isoformat())


def get_short_pct(code: str, as_of: date, shares_outstanding: float) -> Optional[float]:
    """
    指定日時点での空売り残高比率（空売り株数 / 発行済株式数）を返す。
    """
    from lib.db import get_short_balance_at
    rec = get_short_balance_at(code, as_of.isoformat())
    if rec is None or shares_outstanding <= 0:
        return None
    return rec["short_balance"] / shares_outstanding


# ------------------------------------------------------------------ #
# ユーティリティ
# ------------------------------------------------------------------ #

def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """候補カラム名リストから実際にある最初のカラム名を返す。"""
    for c in candidates:
        if c in df.columns:
            return c
    # 部分一致
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None


def _normalize_date(val) -> str:
    """J-Quants が返す日付を YYYY-MM-DD 文字列に変換。"""
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s[:10]  # 既に YYYY-MM-DD
