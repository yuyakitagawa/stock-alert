import type { MicroCMSListContent } from "microcms-js-sdk";

export type NewsCategory = "プレスリリース" | "IR情報" | "サステナビリティ" | "お知らせ";

export const NEWS_CATEGORIES: NewsCategory[] = [
  "お知らせ",
  "プレスリリース",
  "IR情報",
  "サステナビリティ",
];

export type News = {
  title: string;
  body: string;
  summary: string;
  category: NewsCategory;
};

export type NewsContent = News & MicroCMSListContent;
