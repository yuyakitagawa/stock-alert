"""
tools/fetch_jquants_fin.py
J-Quants Free プランで利用可能な fins/summary を取得して DB に保存。

取得範囲: 2024-03-17 ～ 2026-03-17 (Free プランの利用可能期間)
API制限: 5件/分 → 12秒スリープ
所要時間: 約65分（発表日329日分）

保存先: jquants_fin_summary テーブル
  code, disc_date, doc_type, fy_end, np, cfo, ta, equity, eps, bps,
  div_ann, payout_ratio, sh_out, tr_sh

使い方:
    python3 tools/fetch_jquants_fin.py          # 全期間フェッチ（nohup推奨）
    python3 tools/fetch_jquants_fin.py --resume # 未取得日のみフェッチ
"""
import sys, os, time, argparse, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import date, datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/fetch_jquants_fin.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)

# Free プラン利用可能期間
AVAIL_START = "2024-03-17"
AVAIL_END   = "2026-03-17"

RATE_SLEEP = 13  # 12秒+余裕 (5件/分)


def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def get_target_dates() -> list[str]:
    """
    DBの fundamentals_annual から実際に発表があった日付を取得。
    未取得分のみ返す（--resume 時は取得済みをスキップ）。
    """
    from lib.db import DB_PATH, init_db
    init_db()
    con = sqlite3.connect(DB_PATH)

    # 発表があった日付（利用可能範囲内）
    rows = con.execute("""
        SELECT DISTINCT announce_date
        FROM fundamentals_annual
        WHERE announce_date IS NOT NULL
          AND announce_date >= ?
          AND announce_date <= ?
        ORDER BY announce_date
    """, (AVAIL_START, AVAIL_END)).fetchall()

    all_dates = [r[0] for r in rows]
    con.close()
    return all_dates


def get_fetched_dates() -> set[str]:
    """既にDBに保存済みの disc_date 一覧を返す。"""
    from lib.db import DB_PATH, init_db
    init_db()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT DISTINCT disc_date FROM jquants_fin_summary").fetchall()
    con.close()
    return {r[0] for r in rows}


def parse_row(row: pd.Series) -> dict:
    """J-Quants のレコード1行 → DB保存用 dict に変換。"""

    def _f(v):
        try:
            return float(v) if v is not None and str(v) not in ("nan", "None", "") else None
        except (ValueError, TypeError):
            return None

    # code: 5桁 "72030" → 4桁 "7203"
    code_raw = str(row.get("Code", "")).strip()
    code = code_raw[:4] if len(code_raw) == 5 else code_raw.zfill(4)

    disc_date = str(row.get("DiscDate", ""))[:10]

    doc_type_raw = str(row.get("DocType", ""))
    if "FYFin" in doc_type_raw:
        doc_type = "FY"
    elif "1QFin" in doc_type_raw:
        doc_type = "1Q"
    elif "2QFin" in doc_type_raw:
        doc_type = "2Q"
    elif "3QFin" in doc_type_raw:
        doc_type = "3Q"
    else:
        doc_type = doc_type_raw[:10]

    fy_end_raw = row.get("CurFYEn")
    fy_end = str(fy_end_raw)[:10] if fy_end_raw is not None else None

    return {
        "code":         code,
        "disc_date":    disc_date,
        "doc_type":     doc_type,
        "fy_end":       fy_end,
        "np":           _f(row.get("NP")),
        "cfo":          _f(row.get("CFO")),
        "ta":           _f(row.get("TA")),
        "equity":       _f(row.get("Eq")),
        "eps":          _f(row.get("EPS")),
        "bps":          _f(row.get("BPS")),
        "div_ann":      _f(row.get("DivAnn")),
        "payout_ratio": _f(row.get("PayoutRatioAnn")),
        "sh_out":       _f(row.get("ShOutFY")),
        "tr_sh":        _f(row.get("TrShFY")),
        # 通期会社予想（進捗率＝累計実績NP÷FNP の計算に使用）
        "fnp":          _f(row.get("FNP")),
        "fop":          _f(row.get("FOP")),
        "fsales":       _f(row.get("FSales")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="取得済み日付をスキップ")
    args = parser.parse_args()

    load_env()

    api_key = os.environ.get("JQUANTS_API_KEY", "")
    if not api_key:
        print("ERROR: JQUANTS_API_KEY が設定されていません")
        sys.exit(1)

    from jquantsapi import ClientV2
    from lib.db import bulk_upsert_jquants_fin_summary, init_db
    init_db()

    cli = ClientV2(api_key=api_key)

    target_dates = get_target_dates()
    logger.info(f"対象日数: {len(target_dates)} 日 ({AVAIL_START} ～ {AVAIL_END})")
    logger.info(f"予想時間: {len(target_dates) * RATE_SLEEP // 60} 分")

    if args.resume:
        fetched = get_fetched_dates()
        target_dates = [d for d in target_dates if d not in fetched]
        logger.info(f"未取得: {len(target_dates)} 日 (取得済みスキップ)")

    total_rows = 0
    errors = 0

    for i, d in enumerate(target_dates):
        date_str = d.replace("-", "")
        try:
            df = cli.get_fin_summary(date_yyyymmdd=date_str)
            if df is not None and len(df) > 0:
                rows = [parse_row(row) for _, row in df.iterrows()]
                # 財務諸表（FY/1Q/2Q/3Q）のみ保存。配当予想・業績修正等は除外
                rows = [r for r in rows if r["code"] and r["disc_date"]
                        and r["doc_type"] in ("FY", "1Q", "2Q", "3Q")
                        and r["cfo"] is not None]
                bulk_upsert_jquants_fin_summary(rows)
                total_rows += len(rows)
                logger.info(f"[{i+1}/{len(target_dates)}] {d}: {len(rows)}銘柄 (累計: {total_rows:,})")
            else:
                logger.info(f"[{i+1}/{len(target_dates)}] {d}: 0銘柄")
        except Exception as e:
            logger.error(f"[{i+1}/{len(target_dates)}] {d}: ERROR {e}")
            errors += 1

        if i < len(target_dates) - 1:
            time.sleep(RATE_SLEEP)

    logger.info(f"完了: {total_rows:,}行保存 / {errors}エラー")
    logger.info("次: python3 core/rf_train_v3.py  ← CFO/TA/自社株買い特徴量込みで再学習")


if __name__ == "__main__":
    main()
