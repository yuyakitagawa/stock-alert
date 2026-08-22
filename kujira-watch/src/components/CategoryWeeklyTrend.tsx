import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import { DEAL_TYPE_COLORS, DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";
import { formatAmountParts, formatDealAmount } from "@/lib/format";
import type { DealType } from "@/types/article";

// 「どの投資家分類が、どの週に活発だったか」を1枚で見せる週次のヒートマップ表（/weekly）。
// 分類は13種類あり系列ごとの折れ線・積み上げ棒にすると色が破綻するため、行=分類・
// 列=週のマトリクスにして、セルの濃さで金額規模を表す。数字も併記するので濃淡が
// 読めなくても（印刷・色覚特性）情報は落ちない。
// 買いと売りは同じ分類でも意味が正反対（買い増しと利益確定を足すと打ち消し合う）なので、
// 1つの表に混ぜず、買い用・売り用の2枚を並べて使う。濃淡の色は方向を表す（買い=青／売り=赤）。

export type CategoryWeekCell = { count: number; amount: number };

export type CategoryWeeklyRow = {
  dealType: DealType;
  cells: CategoryWeekCell[]; // weeksと同じ並び（古い週→新しい週）
  total: number;
};

export type CategoryWeekColumn = {
  weekStart: string;
  axisLabel: string; // 列見出し（例: "8/17"）
  tableLabel: string; // ツールチップ用（例: "8/17〜8/23"）
  isPartial: boolean;
};

// 億円単位の短縮表記（セル用。1兆円以上は「1.2兆」）。
function shortAmount(amountOku: number): string {
  const { value, unit } = formatAmountParts(amountOku, 0);
  return `${value}${unit === "兆円" ? "兆" : ""}`;
}

// 方向色（買い=青／売り=赤）をセル背景へ薄く敷く。金額の比は桁で開くため平方根で圧縮し、
// 小さい週も「動きがあった」ことが分かる程度には色が付くようにする。
// 最大でも0.35に抑えるのは、濃いセルの上でも金額・件数の文字のコントラストを保つため。
function tint(hex: string, ratio: number): string {
  const alpha = ratio > 0 ? 0.06 + 0.29 * Math.sqrt(ratio) : 0;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
}

// 分類名の列は横スクロールしても残す。スマホでは分類名（最長「国内アセットマネジメント」）を
// そのまま1行で出すと画面の6割を占めて週の列が2つしか見えないため、折り返して幅を抑える。
const stickyLabelCell = {
  position: "sticky",
  left: 0,
  zIndex: 1,
  bgcolor: "background.default",
  maxWidth: { xs: 104, sm: "none" },
  whiteSpace: { xs: "normal", sm: "nowrap" },
  fontSize: { xs: "0.6875rem", sm: "0.8125rem" },
} as const;

// 買い＝サイトのブランド青、売り＝損失色。AmountTrendChartの買い/売りと同じ配色にそろえる。
const TONE_HEX = { buy: "#0068b7", sell: "#be123c" } as const;

export default function CategoryWeeklyTrend({
  rows,
  weeks,
  tone,
}: {
  rows: CategoryWeeklyRow[];
  weeks: CategoryWeekColumn[];
  tone: "buy" | "sell";
}) {
  if (rows.length === 0 || weeks.length === 0) return null;

  const maxCell = Math.max(...rows.flatMap((r) => r.cells.map((c) => c.amount)), 1);
  const label = tone === "buy" ? "買い" : "売り";

  return (
    <figure className="my-4">
      <TableContainer>
        <Table
          size="small"
          sx={{
            minWidth: 120 + weeks.length * 62 + 76,
            "& .MuiTableCell-root": { borderColor: "divider", px: 1 },
          }}
        >
          <TableHead>
            <TableRow>
              <TableCell sx={{ ...stickyLabelCell, color: "text.secondary" }}>投資家分類</TableCell>
              {weeks.map((w) => (
                <TableCell
                  key={w.weekStart}
                  align="right"
                  sx={{ color: w.isPartial ? "text.disabled" : "text.secondary", whiteSpace: "nowrap" }}
                >
                  {w.axisLabel}
                  {w.isPartial && (
                    <span className="block text-[0.625rem] font-normal">集計中</span>
                  )}
                </TableCell>
              ))}
              <TableCell align="right" sx={{ color: "text.secondary", whiteSpace: "nowrap" }}>
                期間計
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => {
              // 分類が想定外の値（CMS側の選択肢追加など）でも落とさない。
              const dotColor = (DEAL_TYPE_COLORS[row.dealType] ?? DEAL_TYPE_COLORS["その他"]).dot;
              return (
                <TableRow key={row.dealType}>
                  <TableCell
                    title={DEAL_TYPE_DESCRIPTIONS[row.dealType]}
                    sx={{ ...stickyLabelCell, color: "primary.main" }}
                  >
                    <span
                      className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
                      style={{ background: dotColor }}
                    />
                    {row.dealType}
                  </TableCell>
                  {row.cells.map((cell, i) => (
                    <TableCell
                      key={weeks[i].weekStart}
                      align="right"
                      title={
                        cell.count > 0
                          ? `${row.dealType} ${weeks[i].tableLabel}の${label}: ${cell.count}件・${formatDealAmount(cell.amount)}`
                          : `${row.dealType} ${weeks[i].tableLabel}の${label}: 開示なし`
                      }
                      sx={{
                        bgcolor: tint(TONE_HEX[tone], cell.amount / maxCell),
                        color: cell.count > 0 ? "text.primary" : "text.disabled",
                        whiteSpace: "nowrap",
                        lineHeight: 1.3,
                      }}
                    >
                      {cell.count > 0 ? (
                        <>
                          {shortAmount(cell.amount)}
                          <span className="block text-[0.625rem] text-foreground/70">{cell.count}件</span>
                        </>
                      ) : (
                        "–"
                      )}
                    </TableCell>
                  ))}
                  <TableCell
                    align="right"
                    sx={{ fontWeight: 700, color: "primary.main", whiteSpace: "nowrap" }}
                  >
                    {shortAmount(row.total)}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      <figcaption className="mt-1 text-xs text-foreground/50">
        解説記事化した{label}の開示について、推定取引金額を投資家分類×週（月曜始まり）で集計。
        単位は億円で、色が濃いセルほどその週に大きく{label}へ動いた分類です。行は期間計の大きい順。
      </figcaption>
    </figure>
  );
}
