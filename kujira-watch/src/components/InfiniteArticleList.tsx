"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ArticleContent, DealType } from "@/types/article";
import { groupArticlesByDealDate } from "@/lib/groupByDealDate";
import { UI, type Locale } from "@/lib/i18n";
import ArticleCard from "./ArticleCard";
import DealDateHeading from "./DealDateHeading";

export default function InfiniteArticleList({
  initialArticles,
  totalCount,
  dealType,
  excludeIds,
  locale = "ja",
}: {
  initialArticles: ArticleContent[];
  totalCount: number;
  dealType?: DealType;
  // ページ上部の「注目」枠に既に表示済みの記事IDを一覧から除外する（重複表示防止）。
  excludeIds?: Set<string>;
  locale?: Locale;
}) {
  const [articles, setArticles] = useState(initialArticles);
  const [loading, setLoading] = useState(false);
  const loadingRef = useRef(false);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const hasMore = articles.length < totalCount;
  const t = UI[locale];

  const loadMore = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);

    const params = new URLSearchParams({ offset: String(articles.length) });
    if (dealType) params.set("dealType", dealType);
    if (locale === "en") params.set("translatedOnly", "1");
    const res = await fetch(`/api/articles?${params.toString()}`);
    const data: { contents: ArticleContent[] } = await res.json();

    setArticles((prev) => [...prev, ...data.contents]);
    loadingRef.current = false;
    setLoading(false);
  }, [articles.length, dealType, locale]);

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
  const groups = groupArticlesByDealDate(rest, locale);

  return (
    <>
      {groups.map((group) => (
        <div key={group.date} className="mb-8">
          <DealDateHeading date={group.date} label={group.label} locale={locale} />
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            {group.articles.map((article) => (
              <ArticleCard key={article.id} article={article} locale={locale} />
            ))}
          </div>
        </div>
      ))}
      {hasMore && (
        <div ref={sentinelRef} className="flex justify-center py-8 text-sm text-foreground/40">
          {loading ? t.loadingMore : ""}
        </div>
      )}
    </>
  );
}
