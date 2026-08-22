import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import { DEAL_TYPE_COLORS, DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { formatAmountParts, formatDealAmount } from "@/lib/format";
import type { DealType } from "@/types/article";

// 投資家分類ごとの週次トレンド（/weekly）。分類1つにつき AmountTrendChart と同じ
// 「買いは上・売りは下」の上下対称の小さな棒グラフを1枚ずつ描き、グリッドに並べる。
// 以前は買い用・売り用の2枚のヒートマップ表に分けていたが、「ある分類が買っているのか
// 売っているのか」を読むには同じ分類の買いと売りが隣にある方が早いため、分類単位で
// 買い・売りを1枚にまとめた（2026-08-22）。縦軸のスケールは分類ごとに取る。当初は
// 全分類共通にしていたが、最大の分類（数千億円規模）に引っ張られて他の分類の棒が
// 数px以下になりほぼ見えなかったため同日中に分類ごとへ変更。分類間の規模の誤読は
// 各カード右上の期間計（買い越し/売り越し額）と「目盛り」表示で補う。

export type CategoryWeekCell = {
  buyCount: number;
  buyAmount: number; // 億円
  sellCount: number;
  sellAmount: number; // 億円
};

export type CategoryTrendRow = {
  dealType: DealType;
  cells: CategoryWeekCell[]; // weeksと同じ並び（古い週→新しい週）
  buyTotal: number;
  sellTotal: number;
};

export type CategoryWeekColumn = {
  weekStart: string;
  axisLabel: string; // 横軸ラベル（例: "8/17"）
  tableLabel: string; // ツールチップ・表用（例: "8/17〜8/23"）
  isPartial: boolean;
};

const WIDTH = 300;
const HEIGHT = 150;
const PAD_TOP = 28; // 目盛り表示＋直接ラベル分
const PAD_BOTTOM = 30; // 横軸ラベル＋売り側の直接ラベル分
const PAD_X = 6;

// 億円単位の短縮表記（1兆円以上は「1.2兆」）。
function shortAmount(amountOku: number): string {
  const { value, unit } = formatAmountParts(amountOku, 0);
  return `${value}${unit === "兆円" ? "兆" : ""}`;
}

function CategoryChart({
  row,
  weeks,
}: {
  row: CategoryTrendRow;
  weeks: CategoryWeekColumn[];
}) {
  const max = Math.max(...row.cells.flatMap((c) => [c.buyAmount, c.sellAmount]), 1);
  const half = (HEIGHT - PAD_TOP - PAD_BOTTOM) / 2;
  const baseY = PAD_TOP + half;
  const step = (WIDTH - PAD_X * 2) / weeks.length;
  const barWidth = Math.min(step - 6, 28);
  const net = row.buyTotal - row.sellTotal;
  const dotColor = (DEAL_TYPE_COLORS[row.dealType] ?? DEAL_TYPE_COLORS["その他"]).dot;

  return (
    <div className="rounded-md border border-rule p-3">
      <div className="flex items-baseline justify-between gap-2">
        <span
          className="text-sm font-bold text-brand-navy"
          title={DEAL_TYPE_DESCRIPTIONS[row.dealType]}
        >
          <span
            className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
            style={{ background: dotColor }}
          />
          {row.dealType}
        </span>
        <span
          className="whitespace-nowrap text-xs font-bold"
          style={{ color: net >= 0 ? "var(--brand-blue)" : "var(--loss)" }}
          title={`期間計: 買い${formatDealAmount(row.buyTotal)} / 売り${formatDealAmount(row.sellTotal)}`}
        >
          {net >= 0 ? "買い越し +" : "売り越し -"}
          {shortAmount(Math.abs(net))}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${row.dealType}の週別の買い・売り推定取引金額の棒グラフ`}
        className="mt-1 h-auto w-full"
      >
        <text x={WIDTH - PAD_X} y={9} textAnchor="end" fontSize={9} fill="var(--foreground)" opacity={0.45}>
          目盛り {formatDealAmount(max)}
        </text>
        <line x1={PAD_X} y1={baseY} x2={WIDTH - PAD_X} y2={baseY} stroke="var(--rule)" strokeWidth={1} />
        {row.cells.map((cell, i) => {
          const w = weeks[i];
          const x = PAD_X + i * step + (step - barWidth) / 2;
          const buyHeight = cell.buyAmount > 0 ? Math.max((cell.buyAmount / max) * half, 2) : 0;
          const sellHeight = cell.sellAmount > 0 ? Math.max((cell.sellAmount / max) * half, 2) : 0;
          // React 19は<title>のchildrenが単一の文字列でないと中身を落とすため、先に組み立てる。
          const tooltip =
            `${row.dealType} ${w.tableLabel}: 買い${formatDealAmount(cell.buyAmount)}（${cell.buyCount}件） / ` +
            `売り${formatDealAmount(cell.sellAmount)}（${cell.sellCount}件）${w.isPartial ? "（集計中）" : ""}`;
          const opacity = w.isPartial ? 0.35 : 1;
          return (
            <g key={w.weekStart}>
              {buyHeight > 0 && (
                <rect x={x} y={baseY - buyHeight} width={barWidth} height={buyHeight} rx={Math.min(3, buyHeight)} fill="var(--brand-blue)" opacity={opacity}>
                  <title>{tooltip}</title>
                </rect>
              )}
              {sellHeight > 0 && (
                <rect x={x} y={baseY} width={barWidth} height={sellHeight} rx={Math.min(3, sellHeight)} fill="var(--loss)" opacity={opacity}>
                  <title>{tooltip}</title>
                </rect>
              )}
              {cell.buyAmount > 0 && (
                <text x={x + barWidth / 2} y={baseY - buyHeight - 4} textAnchor="middle" fontSize={10} fill="var(--foreground)" opacity={0.75}>
                  {shortAmount(cell.buyAmount)}
                </text>
              )}
              {cell.sellAmount > 0 && (
                <text x={x + barWidth / 2} y={baseY + sellHeight + 11} textAnchor="middle" fontSize={10} fill="var(--foreground)" opacity={0.75}>
                  {shortAmount(cell.sellAmount)}
                </text>
              )}
              <text x={x + barWidth / 2} y={HEIGHT - 6} textAnchor="middle" fontSize={9} fill="var(--foreground)" opacity={0.5}>
                {w.axisLabel}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function CategoryTrendGrid({
  rows,
  weeks,
}: {
  rows: CategoryTrendRow[];
  weeks: CategoryWeekColumn[];
}) {
  if (rows.length === 0 || weeks.length === 0) return null;

  const hasPartial = weeks.some((w) => w.isPartial);

  return (
    <figure className="my-4">
      <div className="mb-2 flex items-center gap-4 text-xs text-foreground/60">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-3 rounded-sm" style={{ background: "var(--brand-blue)" }} />
          買い（上）
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-3 rounded-sm" style={{ background: "var(--loss)" }} />
          売り（下）
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {rows.map((row) => (
          <CategoryChart key={row.dealType} row={row} weeks={weeks} />
        ))}
      </div>
      <figcaption className="mt-2 text-xs text-foreground/50">
        解説記事化した開示の推定取引金額を投資家分類×週（月曜始まり）で買い・売りに分けて集計。
        単位は億円。縦軸のスケールは分類ごと（各カード右上の「目盛り」が棒の最大値）なので、分類間の規模は右上の期間計で比べてください。
        並びは期間計（買い＋売り）の大きい順。
        {hasPartial && "薄い棒は今週（集計中）です。"}
      </figcaption>
      <details className="mt-2 text-sm text-foreground/60">
        <summary className="cursor-pointer">数値を表で見る</summary>
        <TableContainer>
          <Table size="small" sx={{ minWidth: 120 + weeks.length * 90, "& .MuiTableCell-root": { borderColor: "divider", px: 1 } }}>
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: "text.secondary", whiteSpace: "nowrap" }}>投資家分類</TableCell>
                {weeks.map((w) => (
                  <TableCell key={w.weekStart} align="right" sx={{ color: w.isPartial ? "text.disabled" : "text.secondary", whiteSpace: "nowrap" }}>
                    {w.axisLabel}〜
                    <span className="block text-[0.625rem] font-normal">買い / 売り</span>
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={row.dealType}>
                  <TableCell sx={{ color: "primary.main", whiteSpace: "nowrap" }}>{row.dealType}</TableCell>
                  {row.cells.map((cell, i) => (
                    <TableCell key={weeks[i].weekStart} align="right" sx={{ whiteSpace: "nowrap", lineHeight: 1.3 }}>
                      <span style={{ color: cell.buyCount > 0 ? "var(--brand-blue)" : undefined }}>
                        {cell.buyCount > 0 ? shortAmount(cell.buyAmount) : "–"}
                      </span>
                      {" / "}
                      <span style={{ color: cell.sellCount > 0 ? "var(--loss)" : undefined }}>
                        {cell.sellCount > 0 ? shortAmount(cell.sellAmount) : "–"}
                      </span>
                      <span className="block text-[0.625rem] text-foreground/70">
                        {cell.buyCount + cell.sellCount}件
                      </span>
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </details>
    </figure>
  );
}
