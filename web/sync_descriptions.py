"""
sync_descriptions.py
スプレッドシート「📝 会社説明」で手動管理した会社説明を Supabase に同期する。

- シートが無ければ作成し、保有株＋値上げ力ウォッチリスト銘柄を seed（説明は一部プリセット）。
- 既存シートは **絶対に clear しない**（手動編集を保護）。未登録の銘柄だけ行を追記する。
- 説明が記入されている行を Supabase `ai_analyses`(model_version=company-desc-v1, date=1970-01-01)
  へ upsert。既存の `/api/stock/[code]/description` がこれをキャッシュとして返すため、
  手動説明が最優先・AIは未記入銘柄のフォールバックになる。

使い方:
  python3 web/sync_descriptions.py          # シート整備 + Supabase同期
  python3 web/sync_descriptions.py --dry-run # 同期せず内容確認のみ
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

import gspread
import requests
from google.oauth2.service_account import Credentials

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")
GCP_KEY_PATH   = os.path.join(BASE_DIR, "gcp_key.json")
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

SHEET_TITLE   = "📝 会社説明"
HEADER        = ["コード", "銘柄名", "説明"]
MODEL_VER     = "company-desc-v1"   # description API の getCached と一致させる
CACHE_DATE    = "1970-01-01"        # description API の saveCache と一致させる

SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# 値上げ力ウォッチリスト銘柄の初期説明（オーナーが編集可能なテンプレート）
SEED_DESCRIPTIONS = {
    "2811": "トマトを中心とした野菜加工メーカー。トマトケチャップ・野菜飲料で国内首位。健康志向を追い風に値上げ転嫁が効きやすい。海外では業務用トマト事業も展開。",
    "2229": "ポテトチップス・じゃがりこなど国内最大手のスナック菓子メーカー。シリアル「フルグラ」も展開し、北米・アジアへ海外進出を進める。",
    "2897": "「チキンラーメン」「カップヌードル」で知られる即席麺の世界的パイオニア。国内首位で、海外でも即席麺事業を拡大している。",
    "2502": "「スーパードライ」を中核とする酒類最大手。ビール・飲料・食品を展開し、欧州・豪州のビール事業買収でグローバル化を進める。",
    "8113": "紙おむつ・生理用品・ペットケアの衛生用品大手。「ムーニー」「ソフィ」などを展開し、アジア新興国で高シェア・高成長。",
    "4452": "洗剤・トイレタリー・化粧品の国内最大手。「アタック」「メリーズ」「ビオレ」などブランド多数。化学品事業も併営する。",
    "4911": "国内最大手の化粧品メーカー。高価格帯ブランドを中核に、日本・中国・トラベルリテールで展開する。",
    "4527": "目薬で国内首位（「Vロート」）。スキンケア「肌ラボ」や胃腸薬など一般用医薬品・化粧品を幅広く展開する。",
    "7956": "哺乳びん・ベビーケア用品で国内首位。育児用品ブランド「Pigeon」で中国・アジアにも展開する。",
    "7309": "自転車部品・釣具の世界的メーカー。ロードバイク用変速機（コンポーネント）で世界シェア圧倒的首位を誇る。",
}


def _client():
    creds = Credentials.from_service_account_file(GCP_KEY_PATH, scopes=SCOPE)
    gc    = gspread.authorize(creds)
    return gc.open_by_key(SPREADSHEET_ID)


def _holdings_seed(sh) -> list[tuple[str, str]]:
    """保有株シートの (コード, 銘柄名) を seed 候補として返す。"""
    out = []
    try:
        ws = sh.worksheet("保有株")
        for row in ws.get_all_records():
            code = str(row.get("コード", "")).strip().zfill(4)
            name = str(row.get("銘柄名", "")).strip()
            if code and code != "0000":
                out.append((code, name))
    except Exception as e:
        print(f"[warn] 保有株 seed 取得失敗（無視）: {e}")
    return out


def ensure_sheet(sh):
    """「📝 会社説明」シートを取得（無ければ作成＋seed）。既存はclearしない。"""
    try:
        ws = sh.worksheet(SHEET_TITLE)
        created = False
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_TITLE, rows=200, cols=3)
        ws.update(range_name="A1", values=[HEADER])
        created = True

    rows = ws.get_all_values()
    if not rows:
        ws.update(range_name="A1", values=[HEADER])
        rows = [HEADER]

    existing_codes = {str(r[0]).strip().zfill(4) for r in rows[1:] if r and r[0].strip()}

    # seed 候補: ウォッチリスト10 + 保有株。未登録のみ追記。
    seed_rows = []
    # ウォッチリスト（説明プリセット）
    name_map = {c: n for c, n in _holdings_seed(sh)}
    wl_names = {
        "2811": "カゴメ", "2229": "カルビー", "2897": "日清食品HD", "2502": "アサヒGHD",
        "8113": "ユニ・チャーム", "4452": "花王", "4911": "資生堂", "4527": "ロート製薬",
        "7956": "ピジョン", "7309": "シマノ",
    }
    for code, desc in SEED_DESCRIPTIONS.items():
        if code not in existing_codes:
            seed_rows.append([code, wl_names.get(code, name_map.get(code, "")), desc])
            existing_codes.add(code)
    # 保有株（説明は空欄、オーナーが記入）
    for code, name in _holdings_seed(sh):
        if code not in existing_codes:
            seed_rows.append([code, name, ""])
            existing_codes.add(code)

    if seed_rows:
        start = len(rows) + 1
        ws.update(range_name=f"A{start}", values=seed_rows)
        print(f"[sheet] {SHEET_TITLE}: {len(seed_rows)}行を追記（既存は保護）"
              + ("（新規作成）" if created else ""))
    else:
        print(f"[sheet] {SHEET_TITLE}: 追記なし（全銘柄登録済み）")
    return ws


def read_descriptions(ws) -> list[dict]:
    """説明が入っている行のみ {code, name, description} で返す。"""
    out = []
    for row in ws.get_all_records():
        code = str(row.get("コード", "")).strip().zfill(4)
        desc = str(row.get("説明", "")).strip()
        name = str(row.get("銘柄名", "")).strip()
        if code and code != "0000" and desc:
            out.append({"code": code, "name": name, "description": desc})
    return out


def upsert_supabase(items: list[dict]) -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[supabase] SUPABASE_URL / SERVICE_KEY 未設定。同期スキップ。")
        return
    rows = [{
        "code":          it["code"],
        "date":          CACHE_DATE,
        "summary":       it["description"],
        "bull_points":   [],
        "bear_points":   [],
        "model_version": MODEL_VER,
    } for it in items]
    if not rows:
        print("[supabase] 同期対象なし（説明が記入された銘柄が0件）。")
        return
    url = f"{SUPABASE_URL}/rest/v1/ai_analyses"
    headers = {
        "apikey":        SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates",
    }
    resp = requests.post(url, headers=headers, json=rows, timeout=30)
    if resp.ok:
        print(f"[supabase] {len(rows)}件 upsert 完了 (model_version={MODEL_VER})")
    else:
        print(f"[supabase] upsert失敗: {resp.status_code} {resp.text[:300]}")


def main():
    dry = "--dry-run" in sys.argv
    if not SPREADSHEET_ID or not os.path.exists(GCP_KEY_PATH):
        print("[error] SPREADSHEET_ID / gcp_key.json が必要です。")
        sys.exit(1)

    sh = _client()
    ws = ensure_sheet(sh)
    items = read_descriptions(ws)
    print(f"[read] 説明記入済み: {len(items)}銘柄")
    for it in items:
        print(f"  {it['code']} {it['name']}: {it['description'][:30]}…")

    if dry:
        print("[dry-run] Supabase同期はスキップしました。")
        return
    upsert_supabase(items)


if __name__ == "__main__":
    main()
