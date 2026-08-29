import { unstable_cache } from "next/cache";

import { getSupabaseServerClient } from "@/lib/supabase";

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
