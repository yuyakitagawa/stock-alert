export type Recommend =
  | "S買い"
  | "A買い"
  | "買い継続"
  | "買い増し"
  | "売り検討"
  | "様子見"
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

export interface AiAnalysis {
  code:          string;
  date:          string;
  summary:       string;
  bull_points:   string[];
  bear_points:   string[];
  model_version: string;
}
