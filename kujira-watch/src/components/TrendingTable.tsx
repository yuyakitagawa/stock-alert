import Link from "next/link";
import { displayFilerName } from "@/lib/format";
import type { TrendingEntry } from "@/lib/trendingStats";

// 期間比較（直近N日 vs 前N日）の増加件数ランキング。
// /trending（銘柄）と/ranking/trending（投資家）で共用する。
// 表だと375pxで横スクロールが必要（minWidth 420px）だったため、1件=1カードにして
// 数値をカード内に折り返す。見出し行が無くなるぶん、各数値にラベルを付けている。
export default function TrendingTable({
  entries,
  headLabel,
  windowDays,
  hrefOf,
  noteOf,
}: {
  entries: TrendingEntry[];
  headLabel: string;
  windowDays: number;
  hrefOf: (entry: TrendingEntry) => string | null;
  // 社名だけでは何の会社か分からないため、銘柄一覧では事業内容(無ければ業種)を1行添える。
  noteOf?: (entry: TrendingEntry) => string | null;
}) {
  return (
    <>
      <p className="kicker mb-2 text-foreground/50">{headLabel}</p>
      <ul className="card-grid card-grid-wide">
        {entries.map((entry) => {
          const href = hrefOf(entry);
          const label = (
            <span className="block">
              {displayFilerName(entry.label)}
              {entry.isNew && <span className="kicker ml-2 whitespace-nowrap text-brand-gold">NEW</span>}
            </span>
          );
          const note = noteOf?.(entry);
          const body = (
            <>
              {note && (
                <span className="mt-1 block text-xs font-normal leading-relaxed text-foreground/60">
                  {note}
                </span>
              )}
              <span className="mt-auto flex flex-wrap items-baseline gap-x-3 gap-y-0.5 pt-1.5 font-normal text-foreground/50">
                <span className="text-sm font-bold text-brand-navy">直近{windowDays}日 {entry.count}件</span>
                <span className="text-xs">前{windowDays}日 {entry.prevCount}件</span>
                <span className="text-xs text-brand-gold">増加 +{entry.delta}件</span>
              </span>
            </>
          );
          return (
            <li key={entry.key}>
              {href ? (
                <Link href={href} className="card flex h-full flex-col font-medium text-brand-blue">
                  {label}
                  {body}
                </Link>
              ) : (
                <span className="card flex h-full flex-col font-medium text-foreground/80">
                  {label}
                  {body}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </>
  );
}
