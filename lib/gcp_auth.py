"""GCPサービスアカウントのアクセストークン取得（GA4 / Search Console 共通）。

鍵はファイル（ローカルの `gcp_key.json`）と環境変数 `GCP_SERVICE_ACCOUNT_JSON`
（GitHub Actions用。鍵ファイルは.gitignoreでリポジトリに入れていないため、CIでは
SecretのJSON文字列を渡す）の両方から読む。スコープは呼び出し側のAPIごとに違うため引数で渡す。
"""
import json
import os


def credentials_path() -> str:
    return os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcp_key.json")


def client_email() -> str:
    """鍵に入っているサービスアカウントのメールアドレス。権限エラーの案内文に使う。"""
    try:
        with open(credentials_path()) as f:
            return json.load(f).get("client_email", "")
    except Exception:
        return ""


def access_token(scope: str) -> "str | None":
    """アクセストークン。鍵が見つからなければNone。"""
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(raw), scopes=[scope])
    else:
        path = credentials_path()
        if not os.path.exists(path):
            return None
        creds = service_account.Credentials.from_service_account_file(path, scopes=[scope])
    creds.refresh(Request())
    return creds.token
