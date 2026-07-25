import type { MicroCMSImage, MicroCMSListContent } from "microcms-js-sdk";

export type DealType =
  | "機関投資家買い"
  | "インサイダー買い"
  | "自社株買い"
  | "ETFフロー"
  | "その他";

export type Category = "決算前動向" | "インサイダー" | "ETFフロー" | "その他";

export const CATEGORIES: Category[] = [
  "決算前動向",
  "インサイダー",
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
