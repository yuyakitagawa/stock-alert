export type Recommend =
  | "S買い"
  | "方向感なし"
  | "弱気シグナル"
  | "下降シグナル"
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

export interface CompanyProfile {
  description: string | null;
  website:     string | null;
  employees:   number | null;
}

export interface QuarterlyEarning {
  period:    string;
  revenue:   number | null;
  netIncome: number | null;
}

export interface DailyQuote {
  date:             string | null;
  price:            number | null;
  open:             number | null;
  high:             number | null;
  low:              number | null;
  close:            number | null;
  volume:           number | null;
  prevClose:        number | null;
  change:           number | null;
  changePct:        number | null;
  fiftyTwoWeekHigh: number | null;
  fiftyTwoWeekLow:  number | null;
}

export interface Article {
  id:               number;
  slug:             string;
  code:             string;
  name:             string;
  title:            string;
  body:             string;
  signal_date:      string;
  net_score:        number | null;
  published_at:     string;
  target_keyword?:  string | null;
  meta_description?: string | null;
  seo_score?:       number | null;
  rewrite_count?:   number | null;
}
