import { formatAmountParts, formatDealAmount } from "@/lib/format";

// 週別の売買金額トレンド（/weekly）用の上下対称の棒グラフ。
// DisclosureTrendChart（開示件数・単一系列）と違い、買い・売りの2系列なので
// ベースラインを中央に置き、買いは上・売りは下へ伸ばして「どちらに傾いた週か」が
// 一目で分かる形にする。外部チャートライブラリは使わずインラインSVGで自前描画する
// （サイト内の他のグラフと同じ方針）。数値は全部を直接ラベルすると潰れるため
// 最新週だけ直接ラベルし、全数値はグラフ下の<details>内テーブルで参照できるようにする。

export type AmountTrendBar = {
  key: string;
  axisLabel: string; // 棒の下の短いラベル（例: "8/11"）
  tableLabel: string; // ツールチップ・テーブルの行ラベル（例: "8/11〜8/17"）
  count: number;
  buyAmount: number; // 億円
  sellAmount: number; // 億円
  isPartial: boolean; // 集計中の週（今週）
};

const WIDTH = 640;
const HEIGHT = 260;
const PAD_TOP = 20; // 直接ラベルの分
const PAD_BOTTOM = 40; // 期間ラベル＋売り側の直接ラベル分
const PAD_X = 8;

// 億円単位の短縮表記（棒に直接置くラベル用。1兆円以上は「1.2兆」）。
function shortAmount(amountOku: number): string {
  const { value, unit } = formatAmountParts(amountOku, 0);
  return `${value}${unit === "兆円" ? "兆" : ""}`;
}

export default function AmountTrendChart({ bars }: { bars: AmountTrendBar[] }) {
  if (bars.length < 2) return null;

  const max = Math.max(...bars.flatMap((b) => [b.buyAmount, b.sellAmount]), 1);
  const plotHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;
  // 買い側・売り側それぞれに使える高さ（ベースラインは中央）。
  const half = plotHeight / 2;
  const baseY = PAD_TOP + half;
  const step = (WIDTH - PAD_X * 2) / bars.length;
  const barWidth = Math.min(step - 8, 34);
  const lastIndex = bars.length - 1;
  // 期間ラベルは重なりを避けて必要なら1本おき（末尾は必ず出す）。
  const labelEvery = bars.length > 10 ? 2 : 1;

  return (
    <figure className="my-4">
      <div className="mb-1 flex items-center gap-4 text-xs text-foreground/60">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-3 rounded-sm" style={{ background: "var(--brand-blue)" }} />
          買い
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-3 rounded-sm" style={{ background: "var(--loss)" }} />
          売り
        </span>
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="週別の買い・売り推定取引金額の棒グラフ"
        className="h-auto w-full"
      >
        <line
          x1={PAD_X}
          y1={baseY}
          x2={WIDTH - PAD_X}
          y2={baseY}
          stroke="var(--rule)"
          strokeWidth={1}
        />
        {bars.map((b, i) => {
          const x = PAD_X + i * step + (step - barWidth) / 2;
          const buyHeight = b.buyAmount > 0 ? Math.max((b.buyAmount / max) * half, 2) : 0;
          const sellHeight = b.sellAmount > 0 ? Math.max((b.sellAmount / max) * half, 2) : 0;
          const net = b.buyAmount - b.sellAmount;
          const showValue = i === lastIndex;
          const showAxisLabel = (bars.length - 1 - i) % labelEvery === 0;
          // React 19は<title>のchildrenが単一の文字列でないと中身を落とすため、先に組み立てる。
          const tooltip =
            `${b.tableLabel}: 買い${formatDealAmount(b.buyAmount)} / 売り${formatDealAmount(b.sellAmount)}` +
            `（差し引き${net >= 0 ? "+" : "-"}${formatDealAmount(Math.abs(net))}・${b.count}件）` +
            `${b.isPartial ? "（集計中）" : ""}`;
          const opacity = b.isPartial ? 0.35 : 1;
          return (
            <g key={b.key}>
              {buyHeight > 0 && (
                <rect
                  x={x}
                  y={baseY - buyHeight}
                  width={barWidth}
                  height={buyHeight}
                  rx={Math.min(3, buyHeight)}
                  fill="var(--brand-blue)"
                  opacity={opacity}
                >
                  <title>{tooltip}</title>
                </rect>
              )}
              {sellHeight > 0 && (
                <rect
                  x={x}
                  y={baseY}
                  width={barWidth}
                  height={sellHeight}
                  rx={Math.min(3, sellHeight)}
                  fill="var(--loss)"
                  opacity={opacity}
                >
                  <title>{tooltip}</title>
                </rect>
              )}
              {showValue && b.buyAmount > 0 && (
                <text
                  x={x + barWidth / 2}
                  y={baseY - buyHeight - 5}
                  textAnchor="middle"
                  fontSize={11}
                  fill="var(--foreground)"
                  opacity={0.75}
                >
                  {shortAmount(b.buyAmount)}
                </text>
              )}
              {showValue && b.sellAmount > 0 && (
                <text
                  x={x + barWidth / 2}
                  y={baseY + sellHeight + 13}
                  textAnchor="middle"
                  fontSize={11}
                  fill="var(--foreground)"
                  opacity={0.75}
                >
                  {shortAmount(b.sellAmount)}
                </text>
              )}
              {showAxisLabel && (
                <text
                  x={x + barWidth / 2}
                  y={HEIGHT - 8}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--foreground)"
                  opacity={0.5}
                >
                  {b.axisLabel}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-1 text-xs text-foreground/50">
        解説記事化した開示の推定取引金額を週別（月曜始まり）に買い・売りへ分けた推移。単位は億円。
        {bars[lastIndex].isPartial && "薄い棒は今週（集計中）です。"}
      </figcaption>
    </figure>
  );
}
