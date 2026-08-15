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
 * ここを守らないと、せっかくの数字がアプリのUIの裏に入って読めなくなる。
 */
export const safeArea = {
  top: 200,
  bottom: 470,
  left: 70,
  right: 190,
} as const;

export const safeHeight = 1920 - safeArea.top - safeArea.bottom;

export const accentFor = (direction: 'buy' | 'sell'): string =>
  direction === 'buy' ? brand.goldBright : brand.sell;
