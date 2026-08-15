import type { FilerWinRate } from "@/lib/investors";
import { formatSignedOku } from "@/lib/format";

// 記事・銘柄ページで提出投資家名の横に出す「乗っかりリターン実績」の要約チップ。
// 算出方法の詳細な注記は/investors/[filer]と/rankingにあるため、ここは
// 「乗っかって儲かってきた投資家か」を移動せず一目で判別できる最小表示に留める。
export default function FilerTrackRecordChip({ winRate }: { winRate: FilerWinRate }) {
  const positive = winRate.totalReturnOku >= 0;
  return (
    <span
      title={`乗っかりリターン実績（推定）: この投資家の買い増し開示${winRate.n}回に開示日から乗っかり、${winRate.holdDays}営業日（約3ヶ月）保有した場合の推定損益の合計。終値ベースの参考値で、将来の成績を保証するものではありません`}
      className={`inline-flex items-center whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-bold ${
        positive
          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
          : "border-rose-200 bg-rose-50 text-rose-700"
      }`}
    >
      乗っかり実績 {formatSignedOku(winRate.totalReturnOku)}・{winRate.n}回
    </span>
  );
}
