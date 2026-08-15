/** 動画の1シーン。kind が見せ方（src/scenes/）の分岐になる。 */
export type SceneKind =
  | 'hook'
  | 'company'
  | 'deal'
  | 'filer'
  | 'change'
  | 'outlook'
  | 'cta';

export type Scene = {
  kind: SceneKind;
  /** 画面に大きく出す字幕（narrationの要約） */
  caption: string;
  /** 読み上げ文。音声が無い場合の尺の見積もりにも使う */
  narration: string;
  /** public/ 配下のナレーション音声ファイル名。TTSが使えない場合は undefined（無音） */
  audio?: string;
  /** 音声の長さ（秒）。これがシーンの尺になる */
  durationSec?: number;
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
  /** 開示日 YYYY-MM-DD */
  discDate: string;
  /** hook → company → deal → filer → change → outlook → cta の順 */
  scenes: Scene[];
};

/** 音声が無いときの尺の見積もり（日本語の読み上げ速度の実測概算・文字/秒）。 */
export const CHARS_PER_SECOND = 7.5;
/** シーンの切り替わりに置く余白（秒）。読み上げ直後に切ると詰まって聞こえる。 */
export const SCENE_PADDING_SEC = 0.45;

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
  discDate: '2026-08-15',
  scenes: [
    {
      kind: 'hook',
      caption: '海外ファンドが33億円を買い集めた',
      narration:
        '海外の資産運用会社が、サンプル製作所の株式をおよそ33億円分、買い集めていたことがわかりました。',
    },
    {
      kind: 'company',
      caption: '電子計測機器の専業メーカー',
      narration:
        'サンプル製作所は、電子計測機器の開発と製造を手がける専業メーカーです。研究開発向けの装置に強みを持っています。',
    },
    {
      kind: 'deal',
      caption: '推定33.4億円・保有比率5.21%',
      narration:
        '今回の大量保有報告書によると、推定の取得金額は33.4億円。保有比率は5.21パーセントに達しました。',
    },
    {
      kind: 'filer',
      caption: '中長期で割安株を買う運用方針',
      narration:
        '提出したのは海外の資産運用会社で、割安に放置された銘柄を中長期で保有する運用方針で知られています。',
    },
    {
      kind: 'change',
      caption: '前回から1.8ポイント積み増し',
      narration:
        '前回の開示では3.4パーセントだったため、1.8ポイントの積み増しとなります。継続的に買い増している形です。',
    },
    {
      kind: 'outlook',
      caption: '大株主として存在感が増す可能性',
      narration:
        '今後、大株主として経営に対する発言力が増していく可能性があります。追加の買い増しがあるかが焦点です。',
    },
    {
      kind: 'cta',
      caption: '続きはクジラウォッチで',
      narration: '詳しい分析はクジラウォッチで公開しています。',
    },
  ],
};
