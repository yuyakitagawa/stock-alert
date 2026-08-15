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
  buy: '#047857',
  sell: '#be123c',
} as const;

// CI(ubuntu)では fonts-noto-cjk、macOS では Hiragino が使われる。
// Web フォントを取りに行かないので、レンダリングがネットワークに依存しない。
export const fontFamily =
  "'Noto Sans JP', 'Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', sans-serif";
