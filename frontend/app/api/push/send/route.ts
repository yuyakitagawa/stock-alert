import { NextRequest, NextResponse } from "next/server";
import webpush from "web-push";
import { supabaseAdmin } from "@/lib/supabase";

webpush.setVapidDetails(
  "mailto:dosankoure@gmail.com",
  process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY!,
  process.env.VAPID_PRIVATE_KEY!
);

export async function POST(req: NextRequest) {
  const secret = req.headers.get("x-internal-secret");
  if (secret !== process.env.INTERNAL_SEND_SECRET) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const sb = supabaseAdmin();

  // 最新日付を取得
  const { data: latest } = await sb
    .from("web_rankings")
    .select("date")
    .order("date", { ascending: false })
    .limit(1)
    .single();

  if (!latest) return NextResponse.json({ sent: 0, skipped: 0 });

  // 当日のシグナルを取得
  const { data: rankings } = await sb
    .from("web_rankings")
    .select("name, code, recommend, net")
    .eq("date", latest.date)
    .in("recommend", ["S買い", "A買い", "売り検討"]);

  const sBuy  = rankings?.filter((r) => r.recommend === "S買い")  ?? [];
  const aBuy  = rankings?.filter((r) => r.recommend === "A買い")  ?? [];
  const sells = rankings?.filter((r) => r.recommend === "売り検討") ?? [];

  const lines: string[] = [];
  if (sBuy.length)  lines.push(`🟢 S買い ${sBuy.length}銘柄`);
  if (aBuy.length)  lines.push(`🟩 A買い ${aBuy.length}銘柄`);
  if (sells.length) lines.push(`🔴 売り検討 ${sells.length}銘柄`);

  if (lines.length === 0) {
    return NextResponse.json({ sent: 0, skipped: 0 });
  }

  const payload = JSON.stringify({
    title: `📈 StockSignal ${latest.date}`,
    body:  lines.join(" / "),
    url:   process.env.NEXT_PUBLIC_SITE_URL ?? "/",
  });

  // サブスクリプション取得
  const { data: subs } = await sb
    .from("push_subscriptions")
    .select("endpoint, keys")
    .eq("enabled", true);

  if (!subs || subs.length === 0) {
    return NextResponse.json({ sent: 0, skipped: 0 });
  }

  let sent = 0;
  let skipped = 0;

  await Promise.all(
    subs.map(async (sub) => {
      try {
        await webpush.sendNotification(
          { endpoint: sub.endpoint, keys: sub.keys as { p256dh: string; auth: string } },
          payload
        );
        sent++;
      } catch (err: unknown) {
        const status = (err as { statusCode?: number }).statusCode;
        if (status === 410 || status === 404) {
          // 期限切れ登録を削除
          await sb.from("push_subscriptions").delete().eq("endpoint", sub.endpoint);
        }
        skipped++;
      }
    })
  );

  return NextResponse.json({ sent, skipped });
}
