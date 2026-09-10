// サイトが提供する計算ツールの台帳。ヘッダー・フッター・ハブページ・サイトマップが
// ここだけを見るようにして、ツールを足したときに導線の追加漏れが起きないようにする。
//
// ツールはすべてブラウザ内で完結する（入力値をサーバーにもAPIにも送らない）。
// 記事は「起きたこと」を伝えるが、読者が自分の持ち株で同じ計算をやり直す手段が
// これまで無かった。教科書（/textbook）で読んだ数字の意味を、自分の数字で確かめる場所。

export type ToolMeta = {
  slug: string;
  title: string;
  /** 一覧・カードに出す1行説明。 */
  summary: string;
  /** メタディスクリプション。何を入れると何が出るかを書く。 */
  description: string;
  /** 何を入力するか（ハブのカードに出す）。 */
  inputs: string[];
  /** 関連する教科書の章。 */
  chapterId: string;
};

export const TOOLS: ToolMeta[] = [
  {
    slug: "filing-deadline",
    title: "大量保有報告書 提出期限・要否チェッカー",
    summary: "保有割合と取得日を入れると、報告書の提出義務と提出期限（休日を除いた5日）が出ます。",
    description:
      "株券等保有割合と取得日を入れるだけで、大量保有報告書・変更報告書の提出義務の有無と提出期限を計算します。金融商品取引法27条の23・27条の25の「五日以内」から、土日・国民の祝日・12/29〜1/3を除いて数えた期限日と、その数え方を1日ずつ表示します。",
    inputs: ["取得日（義務が生じた日）", "今回の保有割合", "前回報告時の保有割合（任意）"],
    chapterId: "rules",
  },
  {
    slug: "buyback-impact",
    title: "自社株買い インパクト計算機",
    summary: "取得枠・発行済株式数・株価から、発行済比率・EPS押し上げ率・出来高何日ぶんかを計算します。",
    description:
      "自社株買いの決定開示に載っている取得価額の上限・取得株数の上限と、発行済株式総数・株価を入れると、発行済株式に対する比率、時価総額に対する比率、枠を使い切った場合のEPS押し上げ率、1日平均出来高の何日ぶんに当たるかを計算します。",
    inputs: ["取得価額の総額の上限", "取得する株式の総数の上限", "発行済株式総数", "株価", "当期純利益・出来高（任意）"],
    chapterId: "buyback",
  },
];

export function toolBySlug(slug: string): ToolMeta | undefined {
  return TOOLS.find((tool) => tool.slug === slug);
}

export function toolPath(slug: string): string {
  return `/tools/${slug}`;
}
