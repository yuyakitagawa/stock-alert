"""Modal app: 金曜にXGBoostモデルを学習してSupabase Storageに保存する。

セットアップ:
  pip install modal
  modal secret create stock-alert-secrets \
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... \
    JQUANTS_API_KEY=... EDINET_API_KEY=... \
    ANTHROPIC_API_KEY=... \
    GH_TOKEN=...   # リポジトリclone用(privateなら必要)

デプロイ:
  modal deploy modal_train.py

手動実行:
  modal run modal_train.py::train
"""
import modal

REPO_URL = "https://github.com/yuyakitagawa/stock-alert.git"
BRANCH = "main"
STORAGE_BUCKET = "models"
MODEL_FILES = ["rf_model.pkl", "rf_drop_model.pkl", "baseline_auc.json",
               "feature_importance.json", "optimal_thresholds.json"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "requests", "pandas", "numpy", "scikit-learn", "joblib",
        "python-dotenv", "openpyxl", "xlrd", "xgboost", "lightgbm",
        "gspread", "google-auth", "yfinance", "anthropic",
        "jquants-api-client",
    )
)

app = modal.App("stock-alert-train", image=image)
secrets = [modal.Secret.from_name("stock-alert-secrets")]


def _upload_to_supabase_storage(local_path: str, filename: str, supabase_url: str, service_key: str) -> None:
    """Supabase Storage REST API でファイルをアップロード（upsert）。"""
    import requests, os
    with open(local_path, "rb") as f:
        data = f.read()
    # Content-Typeを拡張子で判定
    ct = "application/octet-stream" if local_path.endswith(".pkl") else "application/json"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": ct,
        "x-upsert": "true",
    }
    url = f"{supabase_url}/storage/v1/object/{STORAGE_BUCKET}/{filename}"
    resp = requests.post(url, headers=headers, data=data, timeout=120)
    if resp.ok:
        print(f"  ✅ uploaded {filename} ({len(data)//1024}KB)")
    else:
        print(f"  ❌ upload failed {filename}: {resp.status_code} {resp.text[:200]}")
        raise RuntimeError(f"Storage upload failed: {filename}")


@app.function(
    secrets=secrets,
    timeout=3600,
    # 金曜 10:00 UTC = 金曜 19:00 JST
    schedule=modal.Cron("0 10 * * 5"),
)
def train():
    import os, subprocess, sys, tempfile

    supabase_url = os.environ["SUPABASE_URL"]
    service_key = os.environ["SUPABASE_SERVICE_KEY"]
    gh_token = os.environ.get("GH_TOKEN", "")

    with tempfile.TemporaryDirectory() as workdir:
        # リポジトリをclone
        clone_url = REPO_URL
        if gh_token:
            clone_url = REPO_URL.replace("https://", f"https://{gh_token}@")
        print(f"Cloning {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth=1", "--branch", BRANCH, clone_url, workdir],
            check=True, capture_output=True
        )

        # 環境変数を .env に書き出す
        env_keys = [
            "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "JQUANTS_API_KEY",
            "EDINET_API_KEY", "ANTHROPIC_API_KEY",
        ]
        with open(os.path.join(workdir, ".env"), "w") as f:
            for k in env_keys:
                v = os.environ.get(k, "")
                if v:
                    f.write(f"{k}={v}\n")

        # 学習実行
        print("Running rf_train_v3.py ...")
        result = subprocess.run(
            [sys.executable, "core/rf_train_v3.py"],
            cwd=workdir,
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {result.returncode}")

        # 成果物をSupabase Storageにアップロード
        print("Uploading model files to Supabase Storage ...")
        for fname in MODEL_FILES:
            fpath = os.path.join(workdir, fname)
            if os.path.exists(fpath):
                _upload_to_supabase_storage(fpath, fname, supabase_url, service_key)
            else:
                print(f"  ⚠️  {fname} not found, skipping")

    print("Done ✅")


if __name__ == "__main__":
    # ローカルテスト用: modal run modal_train.py
    with app.run():
        train.remote()
