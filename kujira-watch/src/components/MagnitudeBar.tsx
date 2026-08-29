// 一覧カードの数値（推定売買金額・買い入れpt・発行済比率・平均リターン）を、
// カード幅いっぱいの横バーとして数字の下に併記する。数字はどれも本文と同じ大きさで
// 並ぶため「349億円と0.5億円が同じ見え方」になっていた。バーを添えると読まなくても
// 順位と落差が目に入る＝表がそのまま棒グラフになる。
//
// SVGにせず素のspan2つ＋inline styleのwidthだけで描く。一覧は1ページに100件以上並ぶため、
// 1件あたりのHTMLを増やす実装は避ける（`/investors`をMUI行にしてHTML 1.5MB・TTFB 1.9秒まで
// 悪化させた前例がある。dev_log 2026-08-15）。
// カード本体が`<a>`や`<span>`の一覧があるので、要素はすべてspanでブロック化する
// （span内にdivを置くとHTMLとして不正になるため）。
//
// 数値そのものはバーの隣に必ずテキストで出ているので、バーは装飾として`aria-hidden`にする。

const TONE_CLASS = {
  navy: "bg-brand-navy/70",
  gold: "bg-brand-gold",
  gain: "bg-gain/70",
  loss: "bg-loss/70",
} as const;

export type MagnitudeTone = keyof typeof TONE_CLASS;

export default function MagnitudeBar({
  value,
  max,
  tone = "navy",
}: {
  value: number;
  max: number;
  tone?: MagnitudeTone;
}) {
  // 金額不明（0）・全件0のときはバーごと出さない。長さ0のバーを並べても意味が無く、
  // 「金額不明」という状態がかえって読み取りにくくなる。
  if (!(max > 0) || value <= 0) return null;
  // 最小2%。1位との差が3桁ある銘柄でも「バーがある＝集計できている」ことは分かるようにする。
  const pct = Math.max(2, Math.min(100, Math.round((value / max) * 100)));
  return (
    <span aria-hidden className="mt-1.5 block h-1.5 w-full overflow-hidden rounded-full bg-brand-navy/10">
      <span className={`block h-full rounded-full ${TONE_CLASS[tone]}`} style={{ width: `${pct}%` }} />
    </span>
  );
}
