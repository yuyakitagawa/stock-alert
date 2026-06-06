#!/usr/bin/env python3
"""オーナー(Human)が自然文で言った方針・指示を、FM(LLM)が受け取り、
   feedback.md の「今週のコメント」セクションに要点整理して書き込む。

   オーナーは Markdown を直接編集しなくてよい。チャットで方針を言うだけ。
   FM がそれを翻訳・整理し、チームが実行できる形に落とし込む。

   使い方:
     python3 pdca/owner_directive.py "来週はbear相場での防御を最優先にして"
"""
import sys, os, re, subprocess
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

import activity

BASE_DIR = Path(__file__).resolve().parent.parent
PDCA_DIR = Path(__file__).resolve().parent
FEEDBACK = PDCA_DIR / "feedback.md"
TODAY    = date.today().strftime("%Y/%m/%d")

load_dotenv(BASE_DIR / ".env", override=True)

import anthropic
client = anthropic.Anthropic()

SECTION_RE = re.compile(r"(## 今週のコメント\n)(.*?)(?=\n## |\n---|\Z)", re.DOTALL)


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=str(BASE_DIR), capture_output=True)


def main():
    if len(sys.argv) < 2:
        print('使い方: python3 pdca/owner_directive.py "オーナーの方針・指示"')
        sys.exit(1)
    owner_msg = " ".join(sys.argv[1:]).strip()

    fb = FEEDBACK.read_text(encoding="utf-8")
    m  = SECTION_RE.search(fb)
    current = m.group(2).strip() if m else ""

    # 文脈として最終目標部分（冒頭）も渡す
    head = fb[:fb.find("## 投資フェーズ")] if "## 投資フェーズ" in fb else fb[:800]

    prompt = f"""あなたは株式予測モデルチームのファンドマネージャー(FM)です。
オーナーから方針・指示を口頭で受け取りました。あなたの仕事は、これを
feedback.md の「今週のコメント」セクションに、チームが実行できる要点として整理して書き込むことです。

【オーナーの方針・指示】
{owner_msg}

【チームの最終目標（参考）】
{head.strip()}

【現在の「今週のコメント」】
{current}

ルール:
- オーナーの指示が判断を仰ぐ性質（「〜するか判断して」等）なら、FMとして結論を下し、それを反映する
- オーナーの意図を保ちつつ、箇条書き中心で簡潔に
- チーム(Quant数量アナリスト / Consultantマーケットコンサル / Engineer)が何をすべきか明確にする
- 既存の方針と矛盾する新指示が来た場合は新しい方を優先し、古い記述を更新する
- 数値目標（avg / win / big_win_rate / アルファ）に変更があれば正確に反映する
- マークダウンの見出し(##)は付けず、本文のみを出力する

新しい「今週のコメント」セクションの本文だけを出力してください（前後の説明文は不要）。"""

    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    new_body = resp.content[0].text.strip()

    if m:
        new_fb = fb[:m.start(2)] + new_body + "\n" + fb[m.end(2):]
    else:
        new_fb = fb.rstrip() + f"\n\n## 今週のコメント\n\n{new_body}\n"
    new_fb = re.sub(r"最終更新: .*", f"最終更新: {TODAY}", new_fb, count=1)
    FEEDBACK.write_text(new_fb, encoding="utf-8")

    print("=== FM が feedback.md の『今週のコメント』を更新しました ===\n")
    print(new_body)

    activity.record("FM", "オーナーの方針を feedback.md に整理・反映", "done", new_body)

    # コミット&プッシュ（翌営業日のサイクルが読めるように）
    git("config", "user.email", "pdca-bot@github-actions")
    git("config", "user.name",  "PDCA Bot")
    git("add", "pdca/feedback.md")
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(BASE_DIR))
    if r.returncode != 0:
        git("commit", "-m", f"pdca: FMがオーナー方針をfeedback.mdに反映 ({TODAY}) [skip ci]")
        git("pull", "--rebase", "origin", "main")
        git("push")
        print("\n→ コミット&プッシュ完了。活動ログに記録しました。")
    else:
        print("\n→ 変更なし。")


if __name__ == "__main__":
    main()
