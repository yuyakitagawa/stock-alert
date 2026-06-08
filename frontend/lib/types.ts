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
  verdict:       string | null;
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

export interface Activity {
  id:         number;
  ts:         string;
  run_date:   string;
  role:       string;
  step:       string;
  status:     string;
  summary:    string;
  detail:     string;
  updated_at: string;
}

export interface WeeklyReview {
  week:             string;
  created_at:       string;
  avg_start:        number | null;
  avg_end:          number | null;
  win_start:        number | null;
  win_end:          number | null;
  big_start:        number | null;
  big_end:          number | null;
  adopted:          number;
  rejected:         number;
  skipped:          number;
  signals:          number;
  engineer_eval:    string;
  quant_eval:       string;
  securities_eval:  string;
  fm_eval:          string;
  human_feedback:   string;
  next_actions:     string;
}

export interface DividendCandidate {
  date:          string;
  code:          string;
  name:          string;
  yutai_month:   number;
  ex_date:       string;
  ex_month:      number;
  days_since_ex: number;
  div_yield:     number;
  close:         number;
  net:           number;
  drop_prob:     number | null;
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
