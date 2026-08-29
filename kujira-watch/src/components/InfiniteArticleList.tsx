"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import type { ArticleContent, DealType } from "@/types/article";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { UI } from "@/lib/i18n";
import ArticleCard from "./ArticleCard";
import DealDateHeading from "./DealDateHeading";
import DealDateSeeMoreLink from "./DealDateSeeMoreLink";
import AdUnit from "./AdUnit";

// 広告を挟む間隔（記事件数）。オートスクロールで一覧が際限なく伸びるため、
// これだけ記事を読み進めた次の取引日グループの区切りで広告を1枠差し込む。
const ARTICLES_PER_AD = 8;

export default function InfiniteArticleList({
  initialArticles,
  totalCount,
  dealType,
  excludeIds,
  publishedDates,
}: {
  initialArticles: ArticleContent[];
  totalCount: number;
  dealType?: DealType;
  // ページ上部の「注目」枠に既に表示済みの記事IDを一覧から除外する（重複表示防止）。
  excludeIds?: Set<string>;
  // 取引日ページを公開している日（YYYY-MM-DD）。ここに無い日は「この日の記事を見る」を出さない。
  // 追加読み込みで後から出てくる日も判定できるよう、全件をまとめて受け取る。
  publishedDates?: string[];
}) {
  const [articles, setArticles] = useState(initialArticles);
  const publishedDateSet = useMemo(() => new Set(publishedDates ?? []), [publishedDates]);
  const [loading, setLoading] = useState(false);
  const loadingRef = useRef(false);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const hasMore = articles.length < totalCount;
  const t = UI;

  const loadMore = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);

    const params = new URLSearchParams({ offset: String(articles.length) });
    if (dealType) params.set("dealType", dealType);
    const res = await fetch(`/api/articles?${params.toString()}`);
    const data: { contents: ArticleContent[] } = await res.json();

    setArticles((prev) => [...prev, ...data.contents]);
    loadingRef.current = false;
    setLoading(false);
  }, [articles.length, dealType]);

  // 画面下端のsentinelが見えたら次のページを自動取得する（ページネーションの代わり）。
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || !hasMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) loadMore();
      },
      { rootMargin: "600px" }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, loadMore]);

  const rest = excludeIds ? articles.filter((a) => !excludeIds.has(a.id)) : articles;
  const groups = groupArticlesByDealDate(rest);

  // 記事カードの間ではなく取引日グループの区切りに広告を置く（記事と広告が地続きに
  // 見えないようにするため）。最後のグループの後ろは読み込み中のsentinelが続くので置かない。
  const blocks: { group: (typeof groups)[number]; withAd: boolean }[] = [];
  let articlesSinceAd = 0;
  for (const [index, group] of groups.entries()) {
    articlesSinceAd += group.articles.length;
    const withAd = index < groups.length - 1 && articlesSinceAd >= ARTICLES_PER_AD;
    if (withAd) articlesSinceAd = 0;
    blocks.push({ group, withAd });
  }

  return (
    <>
      {blocks.map(({ group, withAd }) => (
        <Fragment key={group.date}>
          <div className="mb-8">
            <DealDateHeading label={group.label} />
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 sm:gap-6">
              {group.articles.map((article) => (
                <ArticleCard key={article.id} article={article} />
              ))}
            </div>
            <DealDateSeeMoreLink
              href={publishedDateSet.has(group.date) ? `/date/${group.date}` : null}
            />
          </div>
          {withAd && <AdUnit placement="infeed" />}
        </Fragment>
      ))}
      {hasMore && (
        <Box ref={sentinelRef} sx={{ display: "flex", justifyContent: "center", py: 4 }}>
          <Typography variant="body2" sx={{ color: "text.disabled" }}>
            {loading ? t.loadingMore : ""}
          </Typography>
        </Box>
      )}
    </>
  );
}
