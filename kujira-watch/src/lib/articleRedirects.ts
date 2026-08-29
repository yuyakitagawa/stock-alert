import { unstable_cache } from "next/cache";

import { getSupabaseServerClient } from "@/lib/supabase";
import { stockHref } from "@/lib/publishedPages";

// 削除した記事URLの引き継ぎ先（Supabase `deleted_article_redirects`。書き込みは
// lib/article_redirects.py＝記事を消す3つのツール）。
//
// なぜ必要か: 2026-08-29のGSC実測で、検索結果に出ているURL924件のうち194件が404を返し、
// そこに28日で25クリック（全クリックの18%）が着地していた。うち124件が削除済みの記事URL。
// 低価値・重複・誤報の記事を消す運用は続けるが、順位の付いたURLは捨てずに引き継ぐ。
//
// 404のときにしか引かないので、通常の記事表示にSupabaseの往復は増えない。
async function getArticleRedirectUncached(id: string): Promise<string | null> {
  try {
    const supabase = getSupabaseServerClient();
    const { data, error } = await supabase
      .from("deleted_article_redirects")
      .select("target_path")
      .eq("article_id", id)
      .maybeSingle();
    if (error) {
      console.error(`[articleRedirects] id=${id} 取得失敗`, error.message);
      return null;
    }
    return data?.target_path ?? null;
  } catch (error) {
    console.error(`[articleRedirects] id=${id} 例外`, error);
    return null;
  }
}

export const getArticleRedirect = unstable_cache(
  getArticleRedirectUncached,
  ["getArticleRedirect"],
  { revalidate: 3600 },
);

// 引き継ぎ先が「いま公開されているページ」かを確かめてから返す。
//
// なぜ要るか: 引き継ぎ先の既定は `/stocks/<code>` だが、銘柄ページは記事数などの条件を
// 満たすものだけを公開している（lib/pageIndexability.ts）。記事を消すと残り記事数が減るため、
// 「記事を消した結果その銘柄ページも非公開になる」組み合わせが起きる。そのまま308を返すと
// 308→404の二段になり、素の404より悪い（クロール枠を食い、リンク先も壊れる）。
// 行き先が無いときは null を返し、呼び出し側は素直に404にする。
export async function resolveArticleRedirect(id: string): Promise<string | null> {
  const target = await getArticleRedirect(id);
  if (!target) return null;
  const stockCode = /^\/stocks\/([^/?#]+)$/.exec(target)?.[1];
  if (stockCode) return await stockHref(stockCode);
  return target;
}
