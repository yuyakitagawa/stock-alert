import type { MicroCMSImage, MicroCMSListContent } from "microcms-js-sdk";

export type DealType =
  | "機関投資家買い" // レガシー値（新規投稿では下記に細分化）
  | "インサイダー買い"
  | "日系ファンド買い"
  | "外資系ファンド買い"
  | "ベンチャーキャピタル買い"
  | "財団買い"
  | "日系企業買い"
  | "外資系企業買い"
  | "自社株買い"
  | "ETFフロー"
  | "その他";

export type Category =
  | "決算前動向"
  | "インサイダー"
  | "日系ファンド"
  | "外資系ファンド"
  | "ベンチャーキャピタル"
  | "財団"
  | "日系企業"
  | "外資系企業"
  | "ETFフロー"
  | "その他";

export const CATEGORIES: Category[] = [
  "インサイダー",
  "日系ファンド",
  "外資系ファンド",
  "ベンチャーキャピタル",
  "財団",
  "日系企業",
  "外資系企業",
  "決算前動向",
  "ETFフロー",
  "その他",
];

export type Article = {
  title: string;
  body: string;
  stockName: string;
  stockCode: string;
  dealType: DealType;
  dealDate: string;
  dealAmount: number;
  sourceUrl?: string;
  category: Category;
  tags?: string;
  eyecatch?: MicroCMSImage;
};

export type ArticleContent = Article & MicroCMSListContent;
