"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { displayFilerName, formatAmountParts } from "@/lib/format";
import type { TrendingCounts } from "@/lib/trendingStats";
import SectorIcon from "./SectorIcon";

// 表示に必要なものはサーバー側で解決して渡す。hrefやnoteを関数propsで受け取ると
// クライアントコンポーネントの境界を越えられないため。
export type TrendingItem = TrendingCounts & {
  key: string;
  label: string;
  href: string | null;
  // 社名だけでは何の会社か分からないため、銘柄一覧では事業内容(無ければ業種)を1行添える。
  note?: string | null;
  // 業種アイコン（SectorIcon）用。
  sector?: string | null;
};

// 初回に描画する件数と、スクロールで追加する単位。
// SSRされるのは初回分だけなので、この値がクローラーから見える件数の下限になる。
const INITIAL_COUNT = 30;
const STEP = 30;

// 金額はランキングの並べ替え軸なので各カードの先頭に出す。桁だけ読めれば十分（億円未満は出さない）。
// 推定できない開示（訂正報告書・株価や発行済株式数が取れない銘柄）は0億円扱いになるため、
// 合計が0のときは「推定0億円」ではなく金額不明として出す（最下位に並ぶ理由が読めるように）。
function amountLabel(amountOku: number): string | null {
  if (amountOku <= 0) return null;
  // 1兆円以上は兆表記に繰り上がる（億円は整数、兆円は小数第1位まで）。
  const { value, unit } = formatAmountParts(amountOku);
  return `推定${value}${unit}`;
}

// 期間比較（直近N日 vs 前N日）で開示が増えた銘柄を、推定売買金額の大きい順に並べたランキング。
// /trending（銘柄）の一覧表。
// 表だと375pxで横スクロールが必要（minWidth 420px）だったため、1件=1カードにして
// 数値をカード内に折り返す。見出し行が無くなるぶん、各数値にラベルを付けている。
// 件数制限を外して全件出すようにしたところ/trendingが466件・HTML 1.08MB（gzip 106KB）に
// なり、perf_check.ymlの閾値100KBを超えたため、下端まで来たら30件ずつ足すオートページャーにする。
// 全件を一度に描画しないだけで、データ自体は最初から手元にあるので追加取得は発生しない。
export default function TrendingTable({
  items,
  headLabel,
  windowDays,
}: {
  items: TrendingItem[];
  headLabel: string;
  windowDays: number;
}) {
  const [shown, setShown] = useState(INITIAL_COUNT);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const hasMore = shown < items.length;

  const showMore = useCallback(() => {
    setShown((prev) => Math.min(prev + STEP, items.length));
  }, [items.length]);

  // 画面下端のsentinelが見えたら次の30件を描画する（ページネーションの代わり）。
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || !hasMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) showMore();
      },
      { rootMargin: "600px" }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, showMore]);

  return (
    <>
      <p className="kicker mb-2 text-foreground/50">
        {headLabel}
        <span className="ml-2 font-normal normal-case tracking-normal">
          {items.length}件中{Math.min(shown, items.length)}件
        </span>
      </p>
      <ul className="card-grid card-grid-wide">
        {items.slice(0, shown).map((entry) => {
          const currentAmount = amountLabel(entry.amount);
          const prevAmount = amountLabel(entry.prevAmount);
          const label = (
            <span className="flex items-start gap-2">
              <SectorIcon sector={entry.sector} />
              <span className="min-w-0">
                {displayFilerName(entry.label)}
                {entry.isNew && <span className="kicker ml-2 whitespace-nowrap text-brand-gold">NEW</span>}
              </span>
            </span>
          );
          const body = (
            <>
              {entry.note && (
                <span className="mt-1 block text-xs font-normal leading-relaxed text-foreground/60">
                  {entry.note}
                </span>
              )}
              <span className="mt-auto flex flex-wrap items-baseline gap-x-3 gap-y-0.5 pt-1.5 font-normal text-foreground/50">
                <span className="text-sm font-bold text-brand-navy">
                  直近{windowDays}日 {currentAmount ?? "金額不明"}
                  <span className="ml-1 font-semibold">{entry.count}件</span>
                </span>
                <span className="text-xs">
                  前{windowDays}日 {prevAmount ?? "―"} {entry.prevCount}件
                </span>
                <span className="text-xs text-brand-gold">増加 +{entry.delta}件</span>
              </span>
            </>
          );
          return (
            <li key={entry.key}>
              {entry.href ? (
                <Link href={entry.href} className="card flex h-full flex-col font-medium text-brand-blue">
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
      {hasMore && (
        // sentinelは「見えたら自動で足す」ためのものだが、IntersectionObserverが働かない
        // 環境（バックグラウンドタブ・一部の埋め込みビューなど）でも先に進めるよう、
        // 同じ要素を押せるボタンにしておく。通常はスクロールだけで自動的に足される。
        <div ref={sentinelRef} className="py-6 text-center">
          <button
            type="button"
            onClick={showMore}
            className="text-sm text-brand-blue hover:underline"
          >
            もっと見る（残り{items.length - shown}件）
          </button>
        </div>
      )}
    </>
  );
}
