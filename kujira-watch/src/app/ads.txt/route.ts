import { ADSENSE_PUBLISHER_ID } from "@/lib/adsense";

// AdSenseで収益化するにはルート直下の /ads.txt に販売者情報が必要。
// パブリッシャーID未設定（審査前・ローカル開発）では404を返し、空・不正な内容を配信しない。
// `f08c47fec0942fa0` は Google の TAG-ID（全パブリッシャー共通の固定値）。
export function GET() {
  if (!ADSENSE_PUBLISHER_ID) {
    return new Response("Not Found", { status: 404 });
  }

  return new Response(`google.com, ${ADSENSE_PUBLISHER_ID}, DIRECT, f08c47fec0942fa0\n`, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}
