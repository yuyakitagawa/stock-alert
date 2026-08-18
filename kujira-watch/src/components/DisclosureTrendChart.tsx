// 開示件数トレンドの棒グラフ。/trending（月別）と/weekly（週別）で共用する。
// /stocks/[code]の株価グラフと同じく外部チャートライブラリを使わず
// インラインSVGで自前描画するサーバーコンポーネント。
// 単一系列なので凡例は置かず、色はブランドネイビー1色（集計中の期間のみ薄色）。
// 数値ラベルは全部の棒には付けず、最大値と最新期間だけ直接ラベルする。
// ホバー時は各棒のSVG <title>（ブラウザ標準ツールチップ）で期間と件数を出し、
// 全数値はグラフ下の<details>内テーブル（クローラーも読める）で参照できるようにする。

export type TrendBar = {
  key: string;
  axisLabel: string; // 棒の下の短いラベル（例: "25/07"・"8/11週"）
  tableLabel: string; // <details>内テーブルの行ラベル（例: "2025年7月"・"8/11〜8/17"）
  count: number;
  isPartial: boolean; // 集計中の期間（当月・今週）
};

const WIDTH = 640;
const HEIGHT = 200;
const PAD_TOP = 22; // 直接ラベルの分だけ上を空ける
const PAD_BOTTOM = 26; // 期間ラベル分
const PAD_X = 8;

export default function DisclosureTrendChart({
  bars,
  ariaLabel,
  caption,
  periodHeadLabel,
  partialNote,
}: {
  bars: TrendBar[];
  ariaLabel: string;
  caption: string;
  periodHeadLabel: string; // テーブル見出し（例: "月"・"週"）
  partialNote: string; // 集計中の棒がある場合の注記（例: "薄い棒は当月（集計中）です。"）
}) {
  if (bars.length < 2) return null;

  const max = Math.max(...bars.map((b) => b.count), 1);
  const plotHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;
  const step = (WIDTH - PAD_X * 2) / bars.length;
  const barWidth = Math.min(step - 6, 32);
  const maxIndex = bars.findIndex((b) => b.count === max);
  const lastIndex = bars.length - 1;
  // 期間ラベルは重なりを避けて1本おき（末尾は必ず出す）。
  const labelEvery = bars.length > 8 ? 2 : 1;

  return (
    <figure className="my-4">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={ariaLabel}
        className="h-auto w-full"
      >
        {/* ベースライン */}
        <line
          x1={PAD_X}
          y1={HEIGHT - PAD_BOTTOM}
          x2={WIDTH - PAD_X}
          y2={HEIGHT - PAD_BOTTOM}
          stroke="var(--rule)"
          strokeWidth={1}
        />
        {bars.map((b, i) => {
          const barHeight = Math.max((b.count / max) * plotHeight, 2);
          const x = PAD_X + i * step + (step - barWidth) / 2;
          const y = HEIGHT - PAD_BOTTOM - barHeight;
          const r = Math.min(4, barWidth / 2, barHeight);
          const showValue = i === maxIndex || i === lastIndex;
          const showAxisLabel = (bars.length - 1 - i) % labelEvery === 0;
          // React 19は<title>のchildrenが単一の文字列でないと中身を落とすため、先に組み立てる。
          const tooltip = `${b.tableLabel}: ${b.count.toLocaleString()}件${b.isPartial ? "（集計中）" : ""}`;
          return (
            <g key={b.key}>
              {/* 上端だけ角丸・ベースラインに接地する棒 */}
              <path
                d={`M${x},${y + r} a${r},${r} 0 0 1 ${r},-${r} h${barWidth - 2 * r} a${r},${r} 0 0 1 ${r},${r} v${barHeight - r} h${-barWidth} Z`}
                fill="var(--brand-navy)"
                opacity={b.isPartial ? 0.35 : 1}
              >
                <title>{tooltip}</title>
              </path>
              {showValue && (
                <text
                  x={x + barWidth / 2}
                  y={y - 6}
                  textAnchor="middle"
                  fontSize={11}
                  fill="var(--foreground)"
                  opacity={0.75}
                >
                  {b.count.toLocaleString()}
                </text>
              )}
              {showAxisLabel && (
                <text
                  x={x + barWidth / 2}
                  y={HEIGHT - PAD_BOTTOM + 16}
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
        {caption}
        {bars[lastIndex].isPartial && partialNote}
      </figcaption>
      <details className="mt-2 text-xs text-foreground/60">
        <summary className="cursor-pointer">数値を表で見る</summary>
        <table className="mt-2">
          <thead>
            <tr>
              <th className="pr-4 text-left font-normal text-foreground/50">{periodHeadLabel}</th>
              <th className="text-right font-normal text-foreground/50">開示件数</th>
            </tr>
          </thead>
          <tbody>
            {bars.map((b) => (
              <tr key={b.key}>
                <td className="pr-4">{b.tableLabel}</td>
                <td className="text-right">
                  {b.count.toLocaleString()}件{b.isPartial && "（集計中）"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
