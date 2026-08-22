/** 動画の1シーン。kind が見せ方（src/scenes/）の分岐になる。 */
export type SceneKind =
  | 'hook'
  | 'company'
  | 'deal'
  | 'filer'
  | 'change'
  | 'chart'
  | 'cta';

export type Scene = {
  kind: SceneKind;
  /** 画面に大きく出す字幕（narrationの要約）。無音視聴者にはこれが本文になる */
  caption: string;
  /** 読み上げ文。音声が無い場合の尺の見積もりにも使う */
  narration: string;
  /** public/ 配下のナレーション音声ファイル名。TTSが使えない場合は undefined（無音） */
  audio?: string;
  /** 音声の長さ（秒）。これがシーンの尺になる */
  durationSec?: number;
  /** kind="chart" のみ: 直近3ヶ月の終値（古い順） */
  closes?: number[];
  /** kind="chart" のみ: closes と同じ長さの日付（YYYY-MM-DD、古い順） */
  dates?: string[];
  /** kind="chart" のみ: 開示日に対応する closes のindex。範囲外なら undefined */
  discIndex?: number;
  /** このシーンの背景動画（public/配下）。company / filer にのみ割り当てる */
  backgroundVideo?: string;
  /** 背景動画の長さ（秒）。ループ再生の区切りに使う */
  backgroundVideoDurationSec?: number;
};

export type ShortProps = {
  /** 銘柄名（例: トヨタ自動車） */
  stockName: string;
  /** 証券コード（例: 7203） */
  stockCode: string;
  /** 大量保有報告書の提出者名 */
  filerName: string;
  /** 提出者の分類ラベル（例: 海外機関投資家） */
  dealTypeLabel: string;
  /** 保有比率が増えた開示か減った開示か */
  direction: 'buy' | 'sell';
  /** 推定取得/売却金額（億円） */
  dealAmountOku: number;
  /** 今回の保有比率（%） */
  holdingRatio: number;
  /** 前回開示の保有比率（%）。本文から拾えない回は undefined で change シーンを出さない */
  prevHoldingRatio?: number;
  /** 開示日 YYYY-MM-DD */
  discDate: string;
  /** hook → company → deal → filer → (change) → (chart) → cta の順 */
  scenes: Scene[];
  /** 効果音（public/se_*.wav）が用意できたか。render.py が生成可否を書き込む */
  sfx?: boolean;
  /** 締めシーンのエンドカード画像（public/配下、Canva製）。render.py が素材があるときだけ書き込む */
  endCard?: string;
};

/** 音声が無いときの尺の見積もり（日本語の読み上げ速度の実測概算・文字/秒）。 */
export const CHARS_PER_SECOND = 7.5;
/**
 * シーンの切り替わりに置く余白（秒）。読み上げ直後に切ると詰まって聞こえる。
 * 0.45秒だと完全無音がシーン境界ごとに入り「再生バグに聞こえる」との指摘が多数あったため
 * 0.18秒に詰めた（2026-08-19）。
 */
export const SCENE_PADDING_SEC = 0.18;

export const sceneDurationSec = (scene: Scene): number => {
  const base =
    scene.durationSec ?? Math.max(1.5, scene.narration.length / CHARS_PER_SECOND);
  return base + SCENE_PADDING_SEC;
};

export const defaultShortProps: ShortProps = {
  stockName: 'サンプル製作所',
  stockCode: '0000',
  filerName: 'サンプル・アセットマネジメント',
  dealTypeLabel: '海外機関投資家',
  direction: 'buy',
  dealAmountOku: 33.4,
  holdingRatio: 5.21,
  prevHoldingRatio: 3.4,
  discDate: '2026-08-15',
  scenes: [
    {
      kind: 'hook',
      caption: 'サンプル製作所に33億円',
      narration: '海外の運用会社がサンプル製作所を33億円買いました。',
    },
    {
      kind: 'company',
      caption: '電子計測機器の専業メーカー',
      narration: 'サンプル製作所は電子計測機器の開発と製造を手がける専業メーカーです。',
    },
    {
      kind: 'deal',
      caption: '推定33.4億円・保有比率5.21%',
      narration: '推定の取得金額は33.4億円。保有比率は5.21パーセントに達しました。',
    },
    {
      kind: 'filer',
      caption: '中長期で割安株を買う運用方針',
      narration: '提出したのは海外の運用会社で、割安株を中長期で持つ方針で知られます。',
    },
    {
      kind: 'change',
      caption: '3.40%から1.81ポイント積み増し',
      narration: '前回は3.40パーセントだったので、1.81ポイントの積み増しとなります。',
    },
    {
      kind: 'cta',
      caption: '続きはブログで公開中',
      narration: '続きは大口投資家の監視ブログで。',
    },
  ],
};
