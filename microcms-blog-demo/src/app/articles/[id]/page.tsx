import { notFound } from "next/navigation";
import CategoryBadge from "@/components/CategoryBadge";
import DealTypeBadge from "@/components/DealTypeBadge";
import { formatDate, formatDealAmount } from "@/lib/format";
import { getArticleDetail } from "@/lib/microcms";
import { categoryLabel } from "@/types/article";

// Route segment config requires a literal value (cannot import from lib/microcms).
export const revalidate = 60;

export default async function ArticleDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  const article = await getArticleDetail(id).catch((error: unknown) => {
    if (error instanceof Error && error.message.includes("status: 404")) {
      notFound();
    }
    throw error;
  });

  const tags = article.tags
    ?.split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);

  return (
    <article className="overflow-hidden rounded-lg bg-white shadow-sm ring-1 ring-gray-200">
      <div className="p-6">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <DealTypeBadge dealType={article.dealType} />
          <CategoryBadge category={article.dealType ? categoryLabel(article.dealType) : undefined} />
        </div>
        <h1 className="mb-4 text-3xl font-bold text-brand-navy">{article.title}</h1>
        <dl className="mb-6 grid grid-cols-2 gap-x-4 gap-y-3 rounded-lg bg-section-tint p-4 text-sm sm:grid-cols-4">
          <div>
            <dt className="text-gray-500">銘柄</dt>
            <dd className="font-medium text-brand-navy">
              {article.stockName}（{article.stockCode}）
            </dd>
          </div>
          <div>
            <dt className="text-gray-500">取引日</dt>
            <dd className="font-medium text-brand-navy">{formatDate(article.dealDate)}</dd>
          </div>
          <div>
            <dt className="text-gray-500">金額規模</dt>
            <dd className="font-medium text-brand-navy">
              {formatDealAmount(article.dealAmount)}
            </dd>
          </div>
          {article.sourceUrl && (
            <div>
              <dt className="text-gray-500">出典</dt>
              <dd>
                <a
                  href={article.sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-brand-blue hover:underline"
                >
                  元記事を見る
                </a>
              </dd>
            </div>
          )}
        </dl>
        <div
          className="prose max-w-none prose-a:text-brand-blue"
          dangerouslySetInnerHTML={{ __html: article.body }}
        />
        {tags && tags.length > 0 && (
          <div className="mt-8 flex flex-wrap gap-2 border-t border-gray-200 pt-4">
            {tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-gray-300 px-2.5 py-0.5 text-xs text-gray-600"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}
