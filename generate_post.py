import pandas as pd
import glob
import os
from datetime import datetime

TOP_X = 5       # X投稿に載せる銘柄数
TOP_NOTE = 20   # note記事に載せる銘柄数


def load_latest_ranking():
    files = glob.glob(os.path.expanduser("~/stock-alert/ranking_*.csv"))
    if not files:
        print("ERROR: ranking CSVが見つかりません。先にrank_stocks.pyを実行してください")
        return None
    latest = max(files, key=os.path.getmtime)
    print(f"読み込み: {latest}")
    return pd.read_csv(latest)


def generate_x_post(df):
    top = df.head(TOP_X)
    today = datetime.now().strftime("%Y/%m/%d")
    lines = []
    lines.append(f"📈【今週の注目株 Top{TOP_X}】{today}")
    lines.append("")
    for _, row in top.iterrows():
        lines.append(f"#{int(row['銘柄コード'])} {row['銘柄名']}")
        lines.append(f"  上昇確率 {row['上昇確率(%)']:.1f}%  株価 ¥{int(row['直近株価(円)']):,}")
    lines.append("")
    lines.append("東証3,756銘柄をスクリーニング＋機械学習で絞り込み。")
    lines.append("※過去データ基づく参考情報です。投資は自己責任で。")
    lines.append("")
    lines.append("#日本株 #株式投資 #スクリーニング")
    return "\n".join(lines)


def generate_note_post(df):
    top = df.head(TOP_NOTE)
    today = datetime.now().strftime("%Y年%m月%d日")
    lines = []
    lines.append(f"# 今週の注目株ランキング Top{TOP_NOTE}【{today}】")
    lines.append("")
    lines.append("## スクリーニング方法")
    lines.append("")
    lines.append("東証上場3,756銘柄（ETF除く）を2段階でスクリーニングしています。")
    lines.append("")
    lines.append("**Step 1：技術指標フィルター**")
    lines.append("- R²（トレンド安定度）0.70以上")
    lines.append("- 3ヶ月モメンタム +5%以上")
    lines.append("- ボラティリティ 50%以下")
    lines.append("")
    lines.append("**Step 2：機械学習スコアリング**")
    lines.append("- Random Forestモデルで「3ヶ月後に+15%以上上昇する確率」を算出")
    lines.append("- 特徴量：直近リターン・移動平均・RSI・ボラティリティ・52週レンジ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## 上位{TOP_NOTE}銘柄")
    lines.append("")
    lines.append("| 順位 | コード | 銘柄名 | 株価 | 上昇確率 |")
    lines.append("|------|--------|--------|------|----------|")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        lines.append(
            f"| {i} | {int(row['銘柄コード'])} | {row['銘柄名']} | "
            f"¥{int(row['直近株価(円)']):,} | {row['上昇確率(%)']:.1f}% |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 免責事項")
    lines.append("")
    lines.append("このランキングは過去の株価データに基づく統計的スクリーニングの結果です。将来の株価上昇を保証するものではありません。掲載情報はデータ提供を目的としており、投資助言ではありません。投資判断はご自身の責任で行ってください。")
    return "\n".join(lines)


def main():
    df = load_latest_ranking()
    if df is None:
        return

    x_post = generate_x_post(df)
    note_post = generate_note_post(df)

    date_str = datetime.now().strftime("%Y%m%d")
    x_path = os.path.expanduser(f"~/stock-alert/post_x_{date_str}.txt")
    note_path = os.path.expanduser(f"~/stock-alert/post_note_{date_str}.md")

    with open(x_path, "w", encoding="utf-8") as f:
        f.write(x_post)
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note_post)

    print("\n=== X投稿文 ===")
    print(x_post)
    print(f"\n文字数: {len(x_post)}文字")
    print(f"\n保存: {x_path}")
    print(f"保存: {note_path}")


if __name__ == "__main__":
    main()