"""
Task7: サバイバーシップバイアス確認スクリプト

現在の学習データには「生き残った銘柄」しか含まれていない。
上場廃止（倒産・MBO・合併）した銘柄は取得できず、学習から除外されている。
このスクリプトはバイアスの規模を推定し、現在のラベル分布の妥当性を評価する。
"""
import numpy as np
import requests
import time
from datetime import date, timedelta

SAVE_DIR = __import__("os").path.expanduser("~/stock-alert")
TRAIN_CUTOFF = date(2025, 1, 1)
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ===== TSE上場廃止統計（推定値） =====
# JPX統計によると、東証全体で年間100〜150社が上場廃止
# うち自己都合廃止（倒産・整理）は約20〜40社、残りは合併・MBO
# 合併/MBOの場合は廃止時に株価が高いことが多い（バイアス方向は正）
# 倒産・整理は大きな下落を示す（バイアス方向は負、モデルに入れるべきデータ）
ANNUAL_DELISTINGS_TOTAL = 130        # 年間上場廃止総数（推定）
ANNUAL_DELISTINGS_NEGATIVE = 30      # 倒産・整理など（下落を伴う）
ANNUAL_DELISTINGS_POSITIVE = 100     # MBO・合併など（株価プレミアム）
TRAINING_YEARS = 2.3                 # 学習データの年数 (2022-09 ~ 2024-12)

def load_training_data():
    import os
    npz = np.load(os.path.join(SAVE_DIR, "training_data.npz"), allow_pickle=True)
    X, yr, yd, dates_raw = npz["X"], npz["yr"], npz["yd"], npz["dates"]
    dates = [date.fromisoformat(str(d)) for d in dates_raw]
    tr_mask = [d < TRAIN_CUTOFF for d in dates]
    X_tr = X[tr_mask]; yr_tr = yr[tr_mask]; yd_tr = yd[tr_mask]
    dates_tr = sorted(set(d for d, m in zip(dates, tr_mask) if m))
    return X_tr, yr_tr, yd_tr, dates_tr

def check_delisted_availability(sample_codes):
    """指定コードがYahoo Financeで取得できるか確認"""
    available = []
    for code in sample_codes:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.T?interval=1d&range=1d"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=5)
            result = resp.json().get("chart", {}).get("result")
            if result:
                available.append(code)
        except Exception:
            pass
        time.sleep(0.3)
    return available

def main():
    print("=" * 60)
    print("Task7: サバイバーシップバイアス分析")
    print("=" * 60)

    # 1. 学習データの基本統計
    print("\n[1] 現在の学習データ統計")
    X_tr, yr_tr, yd_tr, dates_tr = load_training_data()
    n_samples = len(X_tr)
    n_dates = len(dates_tr)
    avg_stocks = n_samples / n_dates if n_dates > 0 else 0
    rise_rate = yr_tr.mean() * 100
    drop_rate = yd_tr.mean() * 100
    print(f"  学習サンプル数: {n_samples:,}")
    print(f"  サンプル日数:   {n_dates}日")
    print(f"  1日平均銘柄数:  {avg_stocks:.0f}銘柄")
    print(f"  上昇ラベル率:   {rise_rate:.1f}%  (63日後+15%以上)")
    print(f"  下落ラベル率:   {drop_rate:.1f}%  (63日後-15%以下)")
    print(f"  対象期間:       {dates_tr[0]} ～ {dates_tr[-1]}")

    # 2. 欠損銘柄の推定
    print("\n[2] 欠損銘柄の推定（サバイバーシップバイアス）")
    missing_negative = int(ANNUAL_DELISTINGS_NEGATIVE * TRAINING_YEARS)
    missing_positive = int(ANNUAL_DELISTINGS_POSITIVE * TRAINING_YEARS)
    missing_total    = missing_negative + missing_positive
    print(f"  学習期間: {TRAINING_YEARS}年")
    print(f"  推定欠損（倒産・整理系）: 約{missing_negative}銘柄  ← ドロップ率を押し上げる方向")
    print(f"  推定欠損（MBO・合併系）:  約{missing_positive}銘柄  ← 上昇率を押し上げる方向")
    print(f"  推定欠損総数:             約{missing_total}銘柄 / {int(avg_stocks + missing_total/n_dates):.0f}銘柄中")

    # 3. バイアスの方向と規模
    print("\n[3] バイアス方向の推定")
    # 倒産銘柄: 廃止前の期間、-15%超下落するサンプルが多い
    est_delisted_neg_drop_rate = 60.0   # 倒産銘柄のdrop label比率(%)
    est_delisted_pos_drop_rate = 5.0    # MBO銘柄のdrop label比率(%)（低い）
    est_delisted_neg_rise_rate = 5.0    # 倒産銘柄のrise label比率(%)（低い）
    est_delisted_pos_rise_rate = 30.0   # MBO銘柄のrise label比率(%)（プレミアム）

    # サンプルレベルで補正: 各銘柄は平均 n_samples/n_unique_stocks のサンプルを生成
    n_unique_stocks = n_samples / (n_samples / n_dates / n_dates * n_dates) if n_dates > 0 else 4000
    # 簡易推定: 4036銘柄が確認されているので固定値を使う
    n_unique_stocks = 4036
    samples_per_stock = n_samples / n_unique_stocks  # ≈27サンプル/銘柄

    add_neg_samples = missing_negative * samples_per_stock
    add_pos_samples = missing_positive * samples_per_stock
    total_samples_corrected = n_samples + add_neg_samples + add_pos_samples

    corrected_drop = (
        n_samples * drop_rate
        + add_neg_samples * est_delisted_neg_drop_rate
        + add_pos_samples * est_delisted_pos_drop_rate
    ) / total_samples_corrected

    corrected_rise = (
        n_samples * rise_rate
        + add_neg_samples * est_delisted_neg_rise_rate
        + add_pos_samples * est_delisted_pos_rise_rate
    ) / total_samples_corrected

    print(f"  現在の下落ラベル率: {drop_rate:.1f}%")
    print(f"  補正後の推定下落率: {corrected_drop:.1f}%  （差: {corrected_drop - drop_rate:+.1f}%）")
    print(f"  現在の上昇ラベル率: {rise_rate:.1f}%")
    print(f"  補正後の推定上昇率: {corrected_rise:.1f}%  （差: {corrected_rise - rise_rate:+.1f}%）")

    # 4. Yahoo Financeでの廃止銘柄アクセス確認
    print("\n[4] 上場廃止銘柄のデータ取得可否テスト")
    # 2023年前後に上場廃止した代表的銘柄
    known_delisted = {
        "6502": "東芝（2023年12月廃止）",
        "7047": "ポート（2023年廃止）",
        "4592": "サンバイオ（株価急落後、現在も上場中だがテスト用）",
    }
    print("  Yahoo Finance での取得可否:")
    for code, name in known_delisted.items():
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.T?interval=1d&range=1mo"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=5)
            result = resp.json().get("chart", {}).get("result")
            status = "✅ 取得可" if result else "❌ 取得不可（廃止済み）"
        except Exception:
            status = "❌ エラー"
        print(f"    {name}: {status}")
        time.sleep(0.3)

    # 5. 結論と対策
    print("\n[5] 結論と対策")
    print(f"""
  【バイアスの規模】
  - 欠損銘柄: 推定{missing_total}銘柄（全上場銘柄の約{missing_total/(n_unique_stocks+missing_total)*100:.1f}%）
  - 下落率への影響: {drop_rate:.1f}% → 推定{corrected_drop:.1f}%（+{corrected_drop-drop_rate:.1f}%ポイント）
  - 上昇率への影響: {rise_rate:.1f}% → 推定{corrected_rise:.1f}%（{corrected_rise-rise_rate:+.1f}%ポイント）

  【バイアスの方向】
  - 倒産・整理銘柄（約{missing_negative}社分）が未学習 → モデルが大暴落を過小評価
  - MBO・合併銘柄（約{missing_positive}社分）が未学習 → 影響は小さい（プレミアム付与で実際はプラス）

  【対策】
  1. ✅ 現在実施済み: scale_pos_weight で クラス不均衡を部分補正
  2. ✅ 現在実施済み: down_streak/drawdown60 フィルターで崩れ銘柄を除外
  3. ⚠️  未実施: 廃止銘柄データの追加（Yahoo Finance では不可、有償DBが必要）
  4. ✅ 推奨: 下落モデルスコアを保守的に解釈（高下落確率は特に重視）

  【結論】
  バイアスの規模は下落率で+{corrected_drop-drop_rate:.1f}%pt程度と推定される。
  無料データソースでは廃止銘柄の追加は困難なため、現状では文書化と
  保守的解釈での運用が現実的。モデル下落スコアは実態より楽観的な可能性あり。
""")


if __name__ == "__main__":
    main()
