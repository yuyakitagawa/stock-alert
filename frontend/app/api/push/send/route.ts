import { NextRequest, NextResponse } from "next/server";
import webpush from "web-push";

// ビルド時（VAPIDキー未設定）にトップレベルで例外を投げないよう、
// 実行時にVAPIDを初期化する。キーが揃っているときのみ設定。
let _vapidReady = false;
function ensureVapid(): boolean {
  if (_vapidReady) return true;
  const pub = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;
  const priv = process.env.VAPID_PRIVATE_KEY;
  if (!pub || !priv) return false;
  webpush.setVapidDetails(
    `mailto:${process.env.VAPID_CONTACT_EMAIL ?? "admin@example.com"}`,
    pub,
    priv
  );
  _vapidReady = true;
  return true;
}

const SB_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SB_KEY = process.env.SUPABASE_SERVICE_KEY!;

function sbHeaders(extra: Record<string, string> = {}) {
  return {
    apikey: SB_KEY,
    Authorization: `Bearer ${SB_KEY}`,
    "Content-Type": "application/json",
    ...extra,
  };
}

export async function POST(req: NextRequest) {
  const secret = req.headers.get("x-internal-secret");
  if (secret !== process.env.INTERNAL_SEND_SECRET) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  if (!ensureVapid()) {
    return NextResponse.json({ error: "VAPID keys not configured" }, { status: 503 });
  }

  // 最新日付を取得
  const latestRes = await fetch(
    `${SB_URL}/rest/v1/web_rankings?select=date&order=date.desc&limit=1`,
    { headers: sbHeaders() }
  );
  const latest = latestRes.ok ? await latestRes.json() : [];
  if (!latest.length) return NextResponse.json({ sent: 0, skipped: 0 });
  const date = latest[0].date;

  // シグナル取得
  const rankRes = await fetch(
    `${SB_URL}/rest/v1/web_rankings?date=eq.${date}&select=name,code,recommend,net`,
    { headers: sbHeaders() }
  );
  const rankings = rankRes.ok ? await rankRes.json() : [];

  const sBuy  = rankings.filter((r: { recommend: string }) => r.recommend.includes("S買い"));
  const sells = rankings.filter((r: { recommend: string }) => r.recommend.includes("売り"));

  const lines: string[] = [];
  if (sBuy.length)  lines.push(`🟢 S買い ${sBuy.length}銘柄`);
  if (sells.length) lines.push(`🔴 売り検討 ${sells.length}銘柄`);

  if (lines.length === 0) return NextResponse.json({ sent: 0, skipped: 0 });

  const payload = JSON.stringify({
    title: `📈 StockSignal ${date}`,
    body:  lines.join(" / "),
    url:   process.env.NEXT_PUBLIC_SITE_URL ?? "/",
  });

  // サブスクリプション取得
  const subsRes = await fetch(
    `${SB_URL}/rest/v1/push_subscriptions?enabled=eq.true&select=endpoint,keys`,
    { headers: sbHeaders() }
  );
  const subs = subsRes.ok ? await subsRes.json() : [];
  if (!subs.length) return NextResponse.json({ sent: 0, skipped: 0 });

  let sent = 0;
  let skipped = 0;

  await Promise.all(
    subs.map(async (sub: { endpoint: string; keys: { p256dh: string; auth: string } }) => {
      try {
        await webpush.sendNotification({ endpoint: sub.endpoint, keys: sub.keys }, payload);
        sent++;
      } catch (err: unknown) {
        const status = (err as { statusCode?: number }).statusCode;
        if (status === 410 || status === 404) {
          await fetch(
            `${SB_URL}/rest/v1/push_subscriptions?endpoint=eq.${encodeURIComponent(sub.endpoint)}`,
            { method: "DELETE", headers: sbHeaders() }
          );
        }
        skipped++;
      }
    })
  );

  return NextResponse.json({ sent, skipped });
}
