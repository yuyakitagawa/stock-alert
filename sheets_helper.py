"""
sheets_helper.py
Google Spreadsheetからwatch_list.csvの代わりにチェック銘柄を読み込む。

【初回セットアップ手順】
1. https://console.cloud.google.com/ にアクセス
2. 新しいプロジェクトを作成（例: stock-alert）
3. 「APIとサービス」→「ライブラリ」→ Google Sheets API を有効化
4. 「APIとサービス」→「認証情報」→「サービスアカウント」を作成
5. 作成したサービスアカウントのキー（JSON）をダウンロード
6. ダウンロードしたJSONを ~/stock-alert/gcp_key.json に置く
7. Googleスプレッドシートを新規作成し、サービスアカウントのメールアドレスを「編集者」として共有
8. スプレッドシートのURLから SPREADSHEET_ID をコピーして .env に追加:
   SPREADSHEET_ID=1aBcDeFgHiJkLmNoPqRsTuVwXyZ...
9. スプレッドシートの1行目に「コード」「銘柄名」と入力（ヘッダー）
10. 2行目以降に銘柄を入力（例: 9434, ソフトバンク）

【スプレッドシートの形式】
| コード | 銘柄名          |
|--------|-----------------|
| 9434   | ソフトバンク    |
| 4689   | LY Corporation  |
| 6098   | リクルート      |
| ...    | ...             |
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")
GCP_KEY_PATH   = os.path.expanduser("~/stock-alert/gcp_key.json")
WATCH_CSV_PATH = os.path.expanduser("~/stock-alert/watch_list.csv")

SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


def _load_from_sheets():
    """Google Sheetsからチェック銘柄を読み込む"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_file(GCP_KEY_PATH, scopes=SCOPE)
        gc    = gspread.authorize(creds)
        sh    = gc.open_by_key(SPREADSHEET_ID)
        ws    = sh.get_worksheet(0)          # 1枚目のシート
        data  = ws.get_all_records()         # ヘッダー行を自動認識
        df = pd.DataFrame(data, dtype=str)
        # 列名を正規化（前後スペース除去）
        df.columns = df.columns.str.strip()
        df["コード"]  = df["コード"].str.strip().str.zfill(4)
        df["銘柄名"] = df["銘柄名"].str.strip()
        return df[["コード", "銘柄名"]]
    except ImportError:
        raise ImportError("gspread / google-auth が未インストールです: pip install gspread google-auth")
    except Exception as e:
        raise RuntimeError(f"Google Sheets 読み込み失敗: {e}")


def _load_from_csv():
    """フォールバック: watch_list.csvから読み込む"""
    if not os.path.exists(WATCH_CSV_PATH):
        raise FileNotFoundError(f"watch_list.csv が見つかりません: {WATCH_CSV_PATH}")
    df = pd.read_csv(WATCH_CSV_PATH, dtype=str)
    df.columns = df.columns.str.strip()
    df["コード"]  = df["コード"].str.strip().str.zfill(4)
    df["銘柄名"] = df["銘柄名"].str.strip()
    return df[["コード", "銘柄名"]]


def load_watch_list() -> dict:
    """
    チェック銘柄を {コード: 銘柄名} の辞書で返す。
    - SPREADSHEET_ID が設定済み + gcp_key.json が存在 → Google Sheets を使用
    - それ以外 → watch_list.csv にフォールバック
    """
    use_sheets = bool(SPREADSHEET_ID) and os.path.exists(GCP_KEY_PATH)

    if use_sheets:
        try:
            df = _load_from_sheets()
            print(f"[INFO] Google Sheetsからチェック銘柄を読み込みました ({len(df)}銘柄)")
            return dict(zip(df["コード"], df["銘柄名"]))
        except Exception as e:
            print(f"[WARN] Google Sheets読み込み失敗、CSVにフォールバック: {e}")

    df = _load_from_csv()
    print(f"[INFO] watch_list.csvからチェック銘柄を読み込みました ({len(df)}銘柄)")
    return dict(zip(df["コード"], df["銘柄名"]))


# 動作確認用
if __name__ == "__main__":
    stocks = load_watch_list()
    print("\n── チェック銘柄 ──")
    for code, name in stocks.items():
        print(f"  {code}: {name}")