"""
lib/publish_ledger.py

「候補はあったのに記事が0件」を、正常な見送りと異常に切り分けて記録する台帳。

なぜ件数だけでは足りないか（2026-08-26）:
  edinet_blog.yml は平日13便回る。素材（開示）は毎日数十件あるが、そのほとんどは
  基準未満・既報・比率変化なしで記事にならない。実際 8/26 の21時便は候補31件→公開0件で
  「0件処理しました。」だけを出して正常終了しているが、これは仕様どおりの正常な便である。
  一方で同じ「候補>0・公開0」でも、Anthropic APIの上限で本文生成が全滅した日
  （2026-08-24）や microCMS が401を返す日は、直さないと記事が永久に出ない異常である。
  件数だけを見る監視は、この2つを区別できないので「毎便鳴る（＝誰も見なくなる）」か
  「一度も鳴らない」のどちらかにしかならない。

設計:
  候補1件ごとに必ず結末を1つ記録し、理由を「正常な見送り」と「異常」に分類する。
  - 正常な見送り（EXPECTED）: 記事が出ないのが仕様どおり。何件あっても通知しない。
  - 異常（ANOMALY）: 素材があるのに出せていない。1件でもあればLINE通知＋終了コード。
  - 未分類: 記録されないまま脱落した候補。将来 `continue` を足して記録を忘れたときに
    ここに落ちるので、監視自体が腐らない。異常として扱う。

使い方:
    ledger = PublishLedger("publish_blog_articles")
    ledger.start(len(candidates))
    for c in candidates:
        if ...:
            ledger.skip(SKIP_BELOW_THRESHOLD, f"{name}({code})")
            continue
        ...
        ledger.publish(f"{name}({code})")
    ledger.report()                 # 内訳を1行で出す（成功時も出す＝日次レビューが読む）
    sys.exit(ledger.finish())       # 異常があればLINE通知して終了コード4
"""
from collections import Counter

from lib import notify

# 異常があったときの終了コード。x_metrics(3)・scan_large_holdings(3)と区別する。
EXIT_ANOMALY = 4

# ── 正常な見送り（記事が出ないのが仕様どおり）──────────────────────────
SKIP_ALREADY_PUBLISHED = "already_published"
# 開示側の台帳(article_published_at)に記録があり、記事を意図的に削除した開示
SKIP_ALREADY_ARTICLED = "already_articled"
SKIP_BELOW_THRESHOLD = "below_threshold"
SKIP_NO_RATIO_CHANGE = "no_ratio_change"
SKIP_WAIT_NEXT_RUN = "wait_next_run"
SKIP_NO_PRIOR_RATIO = "no_prior_ratio"
SKIP_NO_AMOUNT = "no_amount"
SKIP_NO_STOCK_NAME = "no_stock_name"
SKIP_MAX_ARTICLES = "max_articles"

# ── 異常（素材があるのに出せていない）────────────────────────────────
FAIL_GENERATION = "generation_failed"
FAIL_PUBLISH = "publish_failed"
FAIL_PERMISSION = "permission_error"
FAIL_UNCLASSIFIED = "unclassified"

EXPECTED = {
    SKIP_ALREADY_PUBLISHED: "既報",
    SKIP_ALREADY_ARTICLED: "作成済み開示",
    SKIP_BELOW_THRESHOLD: "基準未満",
    SKIP_NO_RATIO_CHANGE: "比率変化なし",
    SKIP_WAIT_NEXT_RUN: "次便へ持ち越し",
    SKIP_NO_PRIOR_RATIO: "直前保有割合が取れない",
    SKIP_NO_AMOUNT: "金額を概算できない",
    SKIP_NO_STOCK_NAME: "銘柄名が取れない",
    SKIP_MAX_ARTICLES: "1回の上限に到達",
}

ANOMALY = {
    FAIL_GENERATION: "記事生成に失敗",
    FAIL_PUBLISH: "microCMSへの投稿に失敗",
    FAIL_PERMISSION: "microCMSの権限エラー",
    FAIL_UNCLASSIFIED: "理由が記録されないまま脱落",
}

# 異常の例を通知に載せる件数（LINEは4,000字だが、読ませたいのは先頭数件だけ）
_MAX_EXAMPLES = 5


class PublishLedger:
    """候補1件ごとの結末を集計する。"""

    def __init__(self, source: str):
        self.source = source
        self.total = 0
        self.published_count = 0
        self.reasons: Counter = Counter()
        self.examples: dict[str, list[str]] = {}
        # max_articles や権限エラーで打ち切った場合、残りの候補は「評価していない」だけで
        # 脱落ではない。未分類の計算から除くために覚えておく。
        self.stopped_early = False

    def start(self, total_candidates: int) -> None:
        self.total = total_candidates

    def publish(self, label: str = "") -> None:
        self.published_count += 1

    def skip(self, reason: str, label: str = "") -> None:
        """候補1件を見送ったことを記録する。未知の reason は異常として扱う。"""
        self.reasons[reason] += 1
        if reason not in EXPECTED:
            self.examples.setdefault(reason, [])
            if len(self.examples[reason]) < _MAX_EXAMPLES and label:
                self.examples[reason].append(label)

    def stop_early(self, reason: str, label: str = "") -> None:
        """残りの候補を評価せずに打ち切ったことを記録する。"""
        self.stopped_early = True
        self.skip(reason, label)

    # ── 判定 ────────────────────────────────────────────────────
    @property
    def unclassified(self) -> int:
        """理由が記録されないまま脱落した候補数。負にはしない。"""
        if self.stopped_early:
            return 0
        accounted = self.published_count + sum(self.reasons.values())
        return max(0, self.total - accounted)

    def anomaly_counts(self) -> dict[str, int]:
        out = {r: n for r, n in self.reasons.items() if r not in EXPECTED and n}
        if self.unclassified:
            out[FAIL_UNCLASSIFIED] = self.unclassified
        return out

    def has_anomaly(self) -> bool:
        return bool(self.anomaly_counts())

    # ── 出力 ────────────────────────────────────────────────────
    def _label(self, reason: str) -> str:
        return EXPECTED.get(reason) or ANOMALY.get(reason) or reason

    def summary(self) -> str:
        """候補と結末の内訳を1行で。日次ログレビューが読む前提で日本語のまま出す。"""
        parts = [f"{self._label(r)}{n}" for r, n in self.reasons.most_common()]
        if self.unclassified:
            parts.append(f"{ANOMALY[FAIL_UNCLASSIFIED]}{self.unclassified}")
        breakdown = f"（{' / '.join(parts)}）" if parts else ""
        return (f"[{self.source}] 候補{self.total}件 → 公開{self.published_count}件{breakdown}")

    def report(self) -> None:
        print(self.summary())
        for reason, n in self.anomaly_counts().items():
            examples = self.examples.get(reason) or []
            tail = f": {', '.join(examples)}" if examples else ""
            print(f"  🚨 {self._label(reason)} {n}件{tail}")

    def notify_message(self) -> str:
        lines = [
            f"候補{self.total}件のうち{self.published_count}件しか公開できていません。", "",
        ]
        for reason, n in self.anomaly_counts().items():
            examples = self.examples.get(reason) or []
            tail = f"（{', '.join(examples)}）" if examples else ""
            lines.append(f"・{self._label(reason)}: {n}件{tail}")
        return "\n".join(lines)

    def finish(self) -> int:
        """内訳を出し、異常があればLINE通知して終了コードを返す。"""
        self.report()
        if not self.has_anomaly():
            return 0
        notify.error(self.source, self.notify_message())
        return EXIT_ANOMALY
