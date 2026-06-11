export type Recommend =
  | "S買い"
  | "方向感なし"
  | "弱気シグナル"
  | "下降シグナル"
  | "買い継続"
  | "買い増し"
  | string;

export interface Ranking {
  date:         string;
  code:         string;
  rank:         number;
  name:         string;
  close:        number;
  rise_prob:    number;
  drop_prob:    number;
  net:          number;
  vol:          number;
  recommend:    Recommend;
  rel20:        number;
  stop_loss:    number | null;
  per:          number | null;
  pbr:          number | null;
  piotroski:    number | null;
  bps_growth:   number | null;
  eps_surprise: number | null;
  pos52:        number | null;
}

export interface QvPosition {
  code:      string;
  name:      string;
  buyDate:   string;
  buyPrice:  number;
  buyNet:    number;
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

export interface WatchMetrics {
  price:       number | null;  // 現在値
  high52:      number | null;  // 52週高値
  drawdownPct: number | null;  // 52週高値からの下落率（負の値=下落）
  per:         number | null;  // 実績PER（Yahooフォールバック）
  pbr:         number | null;  // PBR（Yahooフォールバック）
  spark:       number[];       // 直近1ヶ月の終値（ミニチャート用）
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

export interface QvSimTrade {
  id:           number;
  code:         string;
  name:         string | null;
  entry_date:   string;
  exit_date:    string | null;
  entry_price:  number | null;
  exit_price:   number | null;
  return_pct:   number | null;
  reason:       string | null;
  held_days:    number | null;
  status:       "active" | "closed";
}

export interface RiskRegime {
  date:         string;
  regime:       "risk_on" | "caution" | "risk_off";
  score:        number;
  action:       string;
  label:        string;
  reasons:      string[];
  suppress_buy: boolean;
}
