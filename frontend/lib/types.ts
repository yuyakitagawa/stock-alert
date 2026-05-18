export type Recommend =
  | "S買い"
  | "A買い"
  | "高値警戒"
  | "方向感なし"
  | "弱気シグナル"
  | "下降シグナル"
  | "売り検討"
  | "買い継続"
  | "買い増し"
  | string;

export interface Ranking {
  date:       string;
  code:       string;
  rank:       number;
  name:       string;
  close:      number;
  rise_prob:  number;
  drop_prob:  number;
  net:        number;
  vol:        number;
  recommend:  Recommend;
  rel20:      number;
  stop_loss:  number | null;
  per:        number | null;
  pbr:        number | null;
}

export interface StockMeta {
  code:   string;
  name:   string;
  sector: string | null;
  market: string | null;
  per:    number | null;
  pbr:    number | null;
}

export interface Earnings {
  code:      string;
  next_date: string | null;
}

export interface AiAnalysis {
  code:          string;
  date:          string;
  summary:       string;
  bull_points:   string[];
  bear_points:   string[];
  model_version: string;
}
