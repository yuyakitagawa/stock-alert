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

export const DEAL_TYPES: DealType[] = [
  "機関投資家買い",
  "インサイダー買い",
  "日系ファンド買い",
  "外資系ファンド買い",
  "ベンチャーキャピタル買い",
  "財団買い",
  "日系企業買い",
  "外資系企業買い",
  "自社株買い",
  "ETFフロー",
  "その他",
];

// サイト上部のカテゴリフィルターはmicroCMS側に別フィールドを持たず、dealTypeから
// 「買い」を除いた値をその場で導出する（web/publish_blog_articles.pyのdealType分類と
// 1対1で対応するため、CMS側の選択肢を別途同期させる必要が無い）。
export function categoryLabel(dealType: DealType): string {
  return dealType.endsWith("買い") ? dealType.slice(0, -2) : dealType;
}

export const CATEGORIES: string[] = DEAL_TYPES.map(categoryLabel);

export const DEAL_TYPE_BY_CATEGORY: Record<string, DealType> = Object.fromEntries(
  DEAL_TYPES.map((dealType) => [categoryLabel(dealType), dealType])
);

export type Article = {
  title: string;
  body: string;
  stockName: string;
  stockCode: string;
  dealType: DealType;
  dealDate: string;
  dealAmount: number;
  sourceUrl?: string;
  tags?: string;
  eyecatch?: MicroCMSImage;
};

export type ArticleContent = Article & MicroCMSListContent;
