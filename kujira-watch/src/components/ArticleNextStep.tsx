import Link from "next/link";
import { displayFilerName, formatDate } from "@/lib/format";

// 記事の要点（ファクトボックス）直後に置く、次に見るページへの導線。
//
// なぜここに置くか（2026-08-27のGA4実測、28日間）:
//   記事は入口160セッションでTOPに次ぐ2番目の入口なのに、1人あたりの滞在は16秒しかない
//   （データ/一覧ページは75秒）。回遊導線は本文の下（同じ銘柄の他の記事／投資家／関連ランキング）
//   にしか無く、16秒では到達しない。送り先の銘柄ページは入口の直帰率13.7%と全種別で最も低い＝
//   そこへ着いた人はよく回るので、記事を読み切らなかった人にも見える位置に出す。
//
// href は呼び出し側が lib/publishedPages.ts で解決して渡す。公開していない集約ページ
// （薄い銘柄・投資家・取引日）はnullで来るので、そのボタンは出さない。
export default function ArticleNextStep({
  stockName,
  stockHref,
  filerName,
  filerHref,
  dealDate,
  dateHref,
}: {
  stockName: string;
  stockHref?: string | null;
  filerName?: string;
  filerHref?: string | null;
  dealDate?: string;
  dateHref?: string | null;
}) {
  const dateOnly = dealDate ? dealDate.slice(0, 10) : "";
  if (!stockHref && !filerHref && !dateHref) return null;

  return (
    <nav
      aria-label="この記事の関連ページ"
      className="mb-6 flex flex-wrap gap-2 border-b border-rule pb-4 text-sm"
    >
      {stockHref && (
        <Link
          href={stockHref}
          className="rounded border border-brand-blue px-3 py-1.5 font-bold text-brand-blue hover:bg-brand-blue hover:text-white"
        >
          {stockName}の大株主構成 ›
        </Link>
      )}
      {filerName && filerHref && (
        <Link
          href={filerHref}
          className="rounded border border-rule px-3 py-1.5 text-foreground/70 hover:border-brand-blue hover:text-brand-blue"
        >
          {displayFilerName(filerName)}の保有銘柄 ›
        </Link>
      )}
      {dateOnly && dateHref && (
        <Link
          href={dateHref}
          className="rounded border border-rule px-3 py-1.5 text-foreground/70 hover:border-brand-blue hover:text-brand-blue"
        >
          {formatDate(dateOnly)}の全開示 ›
        </Link>
      )}
    </nav>
  );
}
