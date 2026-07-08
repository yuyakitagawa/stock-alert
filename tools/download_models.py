"""Supabase StorageからモデルファイルをダウンロードしてREPO_ROOTに保存する。

daily_alert.yml のモデル学習ステップの代わりに呼ばれる。
Modal上で金曜に学習・アップロードされたものを取得する。
"""
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
BUCKET = "models"
MODEL_FILES = [
    "rf_model.pkl",
    "rf_drop_model.pkl",
    "baseline_auc.json",
    "feature_importance.json",
    "optimal_thresholds.json",
]

SAVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_file(filename: str) -> bool:
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{filename}"
    headers = {
        "apikey": SERVICE_KEY,
        "Authorization": f"Bearer {SERVICE_KEY}",
    }
    resp = requests.get(url, headers=headers, timeout=120)
    if resp.status_code == 200:
        dest = os.path.join(SAVE_DIR, filename)
        with open(dest, "wb") as f:
            f.write(resp.content)
        print(f"  ✅ {filename} ({len(resp.content)//1024}KB)")
        return True
    elif resp.status_code == 404:
        print(f"  ⚠️  {filename} not found in Storage (初回学習前?)")
        return False
    else:
        print(f"  ❌ {filename}: {resp.status_code} {resp.text[:200]}")
        return False


def main():
    if not SUPABASE_URL or not SERVICE_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定")
        sys.exit(1)

    print("Downloading models from Supabase Storage ...")
    missing_critical = False
    for fname in MODEL_FILES:
        ok = download_file(fname)
        if not ok and fname in ("rf_model.pkl", "rf_drop_model.pkl"):
            missing_critical = True

    if missing_critical:
        print("❌ 必須モデルが取得できませんでした。Modalで学習済みか確認してください。")
        sys.exit(1)

    print("Done ✅")


if __name__ == "__main__":
    main()
