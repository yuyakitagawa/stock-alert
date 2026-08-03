"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ArticleContent, DealType } from "@/types/article";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import ArticleCard from "./ArticleCard";
import DealDateHeading from "./DealDateHeading";
import FeaturedArticleCard from "./FeaturedArticleCard";

export default function InfiniteArticleList({
  initialArticles,
  totalCount,
  dealType,
  showFeatured = false,
}: {
  initialArticles: ArticleContent[];
  totalCount: number;
  dealType?: DealType;
  showFeatured?: boolean;
}) {
  const [articles, setArticles] = useState(initialArticles);
  const [loading, setLoading] = useState(false);
  const loadingRef = useRef(false);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const hasMore = articles.length < totalCount;

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

  const featured = showFeatured ? articles[0] : undefined;
  const rest = showFeatured ? articles.slice(1) : articles;
  const groups = groupArticlesByDealDate(rest);

  return (
    <>
      {featured && <FeaturedArticleCard article={featured} />}
      {groups.map((group) => (
        <div key={group.date} className="mb-8">
          <DealDateHeading label={group.label} />
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {group.articles.map((article) => (
              <ArticleCard key={article.id} article={article} />
            ))}
          </div>
        </div>
      ))}
      {hasMore && (
        <div ref={sentinelRef} className="flex justify-center py-8 text-sm text-foreground/40">
          {loading ? "読み込み中…" : ""}
        </div>
      )}
    </>
  );
}
