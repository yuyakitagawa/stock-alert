// kujira-watch 本体（kujira-watch/src/app/globals.css）と同じブランド配色を使う。
// 動画とサイトで色が食い違うと、Shorts から流入した読者がサイトを別物と感じるため。
export const brand = {
  navy: '#16213a',
  navyDeep: '#0d1526',
  blue: '#0068b7',
  blueDark: '#004c87',
  gold: '#b8863a',
  goldBright: '#d9a44f',
  cream: '#faf7f0',
  buy: '#10b981',
  sell: '#f43f5e',
} as const;

// CI(ubuntu)では fonts-noto-cjk、macOS では Hiragino が使われる。
// Web フォントを取りに行かないので、レンダリングがネットワークに依存しない。
export const fontFamily =
  "'Noto Sans JP', 'Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', sans-serif";

/**
 * TikTok / Shorts のUIに隠れない領域。動画は1080x1920だが、実際に読ませてよいのは
 * この内側だけ。下部はキャプションとメニュー、右端はシェア・いいねのボタン列が重なる。
 *
 * 左右は必ず同じ値にする。以前は left:70 / right:190 だったため、中央寄せした要素が
 * すべて画面中心から60px左にずれていた（2026-08-19のインフルエンサーレビューで指摘）。
 * 対称にしたうえで、右端のボタン列（実測でおよそ x>940）に文字が入らないよう
 * コンテンツ幅を 760px に抑える。
 */
export const safeArea = {
  top: 200,
  bottom: 470,
  left: 160,
  right: 160,
} as const;

/** safeArea の内側の横幅。カード・チャートの最大幅はこれに揃える。 */
export const contentWidth = 1080 - safeArea.left - safeArea.right;

export const safeHeight = 1920 - safeArea.top - safeArea.bottom;

// 実写背景の明部でも文字が読めるようにする共通の影。
export const TEXT_SHADOW = '0 2px 14px rgba(0,0,0,0.75), 0 0 4px rgba(0,0,0,0.5)';

/** 数字・ラベルの下に敷く不透明の下地。背景に文字が溶けるのを構造的に防ぐ。 */
export const PLATE_BG = 'rgba(6,10,20,0.86)';

export const accentFor = (direction: 'buy' | 'sell'): string =>
  direction === 'buy' ? brand.goldBright : brand.sell;
