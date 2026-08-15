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
  /** 冒頭のつかみ（1行・20字前後） */
  hook: string;
  /** 要約の箇条書き（3行・各30字前後） */
  bullets: string[];
  /** 締めの1行 */
  closing: string;
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
  hook: '海外ファンドが33億円を買い集めた',
  bullets: [
    '保有比率は5.21%に到達し、新たな大株主となった',
    '前回開示から1.8ポイントの積み増し',
    '開示日時点の下落リスク水準は「低」',
  ],
  closing: '続きはクジラウォッチで',
};
