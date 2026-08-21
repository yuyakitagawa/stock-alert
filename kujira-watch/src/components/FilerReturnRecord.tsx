import Link from "next/link";
import { formatDate } from "@/lib/format";
import {
  getFilerReturnSummary,
  formatSignedPercent,
  formatSignedPoint,
  MIN_POSITIONS,
  RETURN_TRADING_DAYS,
} from "@/lib/investorReturns";

// 投資家ページの「買い開示の3ヶ月後リターン」パネル（サーバーコンポーネント）。
// /ranking/returns と同じ行（investor_returns_3m）を読むので、ランキングの数字と必ず一致する。
// 買い開示が MIN_POSITIONS 件に満たない投資家はビューに載っていないため何も描画しない
// （n=1・2の「実績」を出さないのは、2026-08-18に撤去した旧filer_win_rateの反省）。

function Stat({
  label,
  value,
  tone,
  note,
}: {
  label: string;
  value: string;
  tone?: "gain" | "loss";
  note?: string;
}) {
  const color = tone === "gain" ? "text-gain" : tone === "loss" ? "text-loss" : "text-foreground/80";
  return (
    <div>
      <dt className="text-[11px] text-foreground/40">{label}</dt>
      <dd className={`m-0 mt-0.5 text-sm font-bold tabular-nums ${color}`}>
        {value}
        {note && <span className="ml-1 text-xs font-normal text-foreground/50">{note}</span>}
      </dd>
    </div>
  );
}

export default async function FilerReturnRecord({ filerName }: { filerName: string }) {
  const summary = await getFilerReturnSummary(filerName);
  if (!summary) return null;

  const { row, rank, filerCount } = summary;
  const tone = (v: number) => (v >= 0 ? ("gain" as const) : ("loss" as const));
  const wins = Math.round((row.winRate / 100) * row.positionCount);

  return (
    <section className="mb-8 rounded-md border border-rule bg-section-tint p-4">
      <h2 className="text-sm font-bold text-brand-navy">買い開示の3ヶ月後リターン</h2>
      <dl className="m-0 mt-3 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        <Stat
          label={`平均（買い開示${row.positionCount}件）`}
          value={formatSignedPercent(row.avgReturn)}
          tone={tone(row.avgReturn)}
        />
        <Stat label="中央値" value={formatSignedPercent(row.medianReturn)} tone={tone(row.medianReturn)} />
        <Stat label="勝率" value={`${row.winRate}%`} note={`${row.positionCount}件中${wins}件`} />
        {row.avgExcessReturn !== null && (
          <Stat
            label="日経平均比"
            value={formatSignedPoint(row.avgExcessReturn)}
            tone={tone(row.avgExcessReturn)}
          />
        )}
      </dl>
      <p className="mb-0 mt-3 text-[11px] leading-relaxed text-foreground/40">
        買い増し・新規取得の開示1件ごとに、開示日（休場の場合は直後の営業日）の終値から
        {RETURN_TRADING_DAYS}営業日後の終値までの騰落率を計算し、等ウェイトで平均した値です。
        対象は{formatDate(row.firstBuyDate)}〜{formatDate(row.latestBuyDate)}の開示で、
        3ヶ月が経過していない開示・売却の開示は含みません。実際の取得単価や売却時期は反映していないため
        この投資家の損益そのものではなく、将来の値動きを示すものでもありません。
        {" "}
        <Link href="/ranking/returns" className="text-brand-blue hover:underline">
          3ヶ月リターンランキング
        </Link>
        では買い開示{MIN_POSITIONS}件以上の{filerCount}名中{rank}位です。
      </p>
    </section>
  );
}
