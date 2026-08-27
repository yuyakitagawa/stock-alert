import Link from "next/link";
import { getHoldingAmountsInRange, getHoldingsInRange } from "@/lib/investors";
import { buildTrendingIssuers, selectDirection } from "@/lib/trendingStats";

// TOPの本文に、銘柄ランキング(/trending)の上位だけを抜き出して置く。
//
// なぜ必要か（2026-08-27のGA4クリックログ実測、28日間）:
//   TOPは閲覧者102人のうち何かを押したのが18人（17.6%）で全ページ中最低。一方
//   /trendingは閲覧者8人全員（100%）、/rankingは83.3%が押している。「中に入れば触られるが
//   入り口で止まっている」状態で、28日で/trendingへ到達したのは364人中8人だった。
//   ヘッダーのタブには同じリンクが並んでいるので、リンクをもう一度置いても意味は薄い。
//   足りないのは押す理由＝中身なので、実際の銘柄名と金額をTOPに出す。
//
// 効果は `python3 tools/ga4_clicks.py` のCTA別クリック（label=銘柄名/「ランキングを…」）で検証する。
const WINDOW_DAYS = 7;
const PREVIEW_COUNT = 3;

function daysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

export default async function TopTrendingPreview() {
  const currentFrom = daysAgo(WINDOW_DAYS - 1);
  const rangeFrom = daysAgo(WINDOW_DAYS * 2 - 1);
  const rangeTo = daysAgo(0);

  // TOPは最も見られるページなので、この枠が取れなくても本体（記事一覧）は必ず出す。
  const [rows, amountByDocId] = await Promise.all([
    getHoldingsInRange(rangeFrom, rangeTo).catch(() => []),
    getHoldingAmountsInRange(rangeFrom, rangeTo).catch(() => ({})),
  ]);
  const top = selectDirection(buildTrendingIssuers(rows, currentFrom, amountByDocId), "both").slice(
    0,
    PREVIEW_COUNT
  );
  if (top.length === 0) return null;

  return (
    <section className="mb-8 rounded border border-foreground/15 p-4">
      {/* スマホでは見出しが折り返してリンクが見出しの途中に挟まって見えるため、縦に積む。 */}
      <div className="mb-3 flex flex-col items-start gap-1 sm:flex-row sm:items-baseline sm:justify-between sm:gap-2">
        <h2 className="text-base font-bold text-brand-navy">直近7日で開示が増えた銘柄</h2>
        <Link href="/trending" className="shrink-0 text-sm text-brand-blue hover:underline">
          ランキングをすべて見る ›
        </Link>
      </div>
      <ol className="space-y-2">
        {top.map((entry, i) => (
          <li key={entry.key} className="flex items-baseline gap-2 text-sm">
            <span className="w-4 shrink-0 font-bold text-foreground/40">{i + 1}</span>
            <Link href={`/stocks/${entry.key}`} className="text-brand-blue hover:underline">
              {entry.label}
            </Link>
            <span className="ml-auto shrink-0 text-foreground/70">
              {entry.count}件
              {entry.amount > 0 && ` / 約${Math.round(entry.amount).toLocaleString()}億円`}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
