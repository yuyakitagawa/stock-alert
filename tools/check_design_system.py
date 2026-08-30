"""kujira-watch のコードがデザインシステムに従っているかを検査する（CI・手動兼用）。

台帳は kujira-watch/docs/design_system.md。トークンを用意しても、コンポーネント側で
場当たりの値を書けば意味が無い。実際 2026-08-30 の導入直後に、任意値サイズ15箇所と
Tailwind標準色6箇所が残っていた（自分で導入したシステムに自分で違反していた）ため、
機械で落とせるようにした。

実行: python3 tools/check_design_system.py
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "kujira-watch" / "src"
LEDGER = "kujira-watch/docs/design_system.md"

# Tailwind標準パレット。ブランドトークン（brand-*/ink-*/surface-*/rule-*/gain/loss）以外で
# 色を指定していたら違反。
TW_PALETTE = (
    "slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|"
    "teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose"
)

RULES = [
    (
        re.compile(r"\b(?:text|border|bg)-foreground/\d+\b"),
        "文字色は用途で選ぶ4段を使う: text-ink / -secondary / -tertiary / -muted",
        ("*.ts", "*.tsx"),
    ),
    (
        re.compile(r"\b(?:text|leading|tracking|rounded|shadow)-\[[^\]]+\]"),
        "任意値ユーティリティでスケールを迂回しない。必要な段が無ければ先に台帳へ足す",
        ("*.tsx",),
    ),
    (
        re.compile(rf"\b(?:text|bg|fill|stroke|border)-(?:{TW_PALETTE})-\d{{2,3}}\b"),
        "Tailwind標準色を直接使わない。上昇/下落は text-gain / text-loss（fill-/stroke-も同じ）",
        ("*.tsx",),
    ),
    (
        re.compile(r'fontSize: *"[0-9]'),
        "フォントサイズを直書きしない。var(--text-*) かMUIのtypography variantを使う",
        ("*.tsx",),
    ),
]

# 台帳そのものを持つファイルは対象外（ここに実値が並ぶのが正しい）。
EXEMPT = {"theme.ts"}


def main() -> int:
    # ディレクトリが無いと rglob が空を返して「違反なし」で通ってしまう。
    # 移動・改名で検査が黙って無効化されるのを防ぐ。
    if not SRC.is_dir():
        print(f"検査対象が見つかりません: {SRC}")
        print(f"移動・改名したなら tools/check_design_system.py の SRC を直すこと。台帳: {LEDGER}")
        return 1

    violations = []
    for pattern, message, globs in RULES:
        for g in globs:
            for path in sorted(SRC.rglob(g)):
                if path.name in EXEMPT:
                    continue
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    for m in pattern.finditer(line):
                        rel = path.relative_to(SRC.parent.parent)
                        violations.append((str(rel), i, m.group(0), message))

    if not violations:
        print("デザインシステム検査: 違反なし ✅")
        return 0

    print(f"デザインシステム検査: {len(violations)}件の違反\n")
    for rel, line, found, message in violations:
        print(f"  {rel}:{line}  {found}")
        print(f"    → {message}")
    print(f"\n台帳: {LEDGER}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
