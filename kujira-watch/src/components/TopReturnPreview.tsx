import Link from "next/link";
import { getLatestReturnCohort, formatSignedPercent } from "@/lib/investorReturns";
import { displayFilerName, formatDate } from "@/lib/format";
import { getPublishedStockCodes } from "@/lib/publishedPages";

// TOPに「3ヶ月前の開示は、その後どうなったか」を置く。
//
// なぜ必要か: 競合の大量保有アラート（アクティビストウォッチャー等）は株価データを持たず、
// 「誰がいくら買ったか」で止まる。開示日を基準にした3ヶ月後リターンはこのサイトにしか無い
// 数字なのに、これまで /ranking/returns と投資家ページの中にしか出ていなかった。
//
// 見せ方の規律: 上位3件だけを並べると勝った銘柄しか出ない。同じ開示日の全件の平均と勝率を
// 必ず先に書き、上位はその内訳として出す（都合の良い数字だけを見せない）。
export default async function TopReturnPreview() {
  // TOPは最も見られるページなので、この枠が取れなくても本体（記事一覧）は必ず出す。
  const [cohort, publishedCodes] = await Promise.all([
    getLatestReturnCohort().catch(() => null),
    // 銘柄ページを公開していない銘柄はリンクにしない（lib/publishedPages.ts）。
    // 薄い集約ページは404にしているので、リンクだけ残すとリンク切れになる。
    getPublishedStockCodes().catch(() => new Set<string>()),
  ]);
  if (!cohort || cohort.top.length === 0) return null;

  return (
    <section className="mb-8 rounded border border-foreground/15 p-4">
      <div className="mb-3 flex flex-col items-start gap-1 sm:flex-row sm:items-baseline sm:justify-between sm:gap-2">
        <h2 className="text-base font-bold text-brand-navy">
          {formatDate(cohort.discDate)}の買い開示は、その後どうなったか
        </h2>
        <Link href="/ranking/returns" className="shrink-0 text-sm text-brand-blue hover:underline">
          投資家別の成績を見る ›
        </Link>
      </div>
      <p className="mb-3 text-sm leading-relaxed text-foreground/70">
        この日の買い開示{cohort.count}件を、開示日の終値で買って3ヶ月（63営業日）持ったと仮定すると、
        {formatDate(cohort.date3m)}時点で平均
        <span className={cohort.avgReturn >= 0 ? "font-bold text-brand-blue" : "font-bold text-red-600"}>
          {formatSignedPercent(cohort.avgReturn)}
        </span>
        ・勝率{cohort.winRate}%でした。内訳の上位3件は次のとおりです。
      </p>
      <ol className="space-y-2">
        {cohort.top.map((entry, i) => (
          <li key={entry.docId} className="flex items-baseline gap-2 text-sm">
            <span className="w-4 shrink-0 font-bold text-foreground/40">{i + 1}</span>
            <span className="min-w-0">
              {publishedCodes.has(entry.issuerCode) ? (
                <Link
                  href={`/stocks/${entry.issuerCode}`}
                  className="text-brand-blue hover:underline"
                >
                  {entry.issuerName}
                </Link>
              ) : (
                <span>{entry.issuerName}</span>
              )}
              <span className="text-foreground/50">（{displayFilerName(entry.filerName)}）</span>
            </span>
            <span
              className={`ml-auto shrink-0 font-medium ${
                entry.ret3m >= 0 ? "text-brand-blue" : "text-red-600"
              }`}
            >
              {formatSignedPercent(entry.ret3m)}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
