import type { MonthlyDisclosureCount } from "@/lib/disclosures";
import { formatMonth } from "@/lib/format";

// /trendingの「月別の開示件数トレンド」棒グラフ。/stocks/[code]の株価グラフと同じく
// 外部チャートライブラリを使わずインラインSVGで自前描画するサーバーコンポーネント。
// 単一系列なので凡例は置かず、色はブランドネイビー1色（当月=集計中のみ薄色）。
// 数値ラベルは全部の棒には付けず、最大値と最新月だけ直接ラベルする。
// ホバー時は各棒のSVG <title>（ブラウザ標準ツールチップ）で月と件数を出し、
// 全数値はグラフ下の<details>内テーブル（クローラーも読める）で参照できるようにする。

const WIDTH = 640;
const HEIGHT = 200;
const PAD_TOP = 22; // 直接ラベルの分だけ上を空ける
const PAD_BOTTOM = 26; // 月ラベル分
const PAD_X = 8;

export default function MonthlyDisclosureTrend({
  counts,
}: {
  counts: MonthlyDisclosureCount[];
}) {
  if (counts.length < 2) return null;

  const max = Math.max(...counts.map((c) => c.count), 1);
  const plotHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;
  const step = (WIDTH - PAD_X * 2) / counts.length;
  const barWidth = Math.min(step - 6, 32);
  const maxIndex = counts.findIndex((c) => c.count === max);
  const lastIndex = counts.length - 1;
  // 月ラベルは重なりを避けて2ヶ月ごと（末尾の月は必ず出す）。
  const labelEvery = counts.length > 8 ? 2 : 1;

  return (
    <figure className="my-4">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="月別のEDINET大量保有報告書 開示件数の棒グラフ"
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
        {counts.map((c, i) => {
          const barHeight = Math.max((c.count / max) * plotHeight, 2);
          const x = PAD_X + i * step + (step - barWidth) / 2;
          const y = HEIGHT - PAD_BOTTOM - barHeight;
          const r = Math.min(4, barWidth / 2, barHeight);
          const showValue = i === maxIndex || i === lastIndex;
          const showMonth = (counts.length - 1 - i) % labelEvery === 0;
          // React 19は<title>のchildrenが単一の文字列でないと中身を落とすため、先に組み立てる。
          const tooltip = `${formatMonth(c.month)}: ${c.count.toLocaleString()}件${c.isPartial ? "（集計中）" : ""}`;
          return (
            <g key={c.month}>
              {/* 上端だけ角丸・ベースラインに接地する棒 */}
              <path
                d={`M${x},${y + r} a${r},${r} 0 0 1 ${r},-${r} h${barWidth - 2 * r} a${r},${r} 0 0 1 ${r},${r} v${barHeight - r} h${-barWidth} Z`}
                fill="var(--brand-navy)"
                opacity={c.isPartial ? 0.35 : 1}
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
                  {c.count.toLocaleString()}
                </text>
              )}
              {showMonth && (
                <text
                  x={x + barWidth / 2}
                  y={HEIGHT - PAD_BOTTOM + 16}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--foreground)"
                  opacity={0.5}
                >
                  {c.month.slice(2).replace("-", "/")}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-1 text-xs text-foreground/50">
        EDINETに提出された大量保有・変更・訂正報告書の月別件数。
        {counts[lastIndex].isPartial && "薄い棒は当月（集計中）です。"}
      </figcaption>
      <details className="mt-2 text-xs text-foreground/60">
        <summary className="cursor-pointer">数値を表で見る</summary>
        <table className="mt-2">
          <thead>
            <tr>
              <th className="pr-4 text-left font-normal text-foreground/50">月</th>
              <th className="text-right font-normal text-foreground/50">開示件数</th>
            </tr>
          </thead>
          <tbody>
            {counts.map((c) => (
              <tr key={c.month}>
                <td className="pr-4">{formatMonth(c.month)}</td>
                <td className="text-right">
                  {c.count.toLocaleString()}件{c.isPartial && "（集計中）"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
