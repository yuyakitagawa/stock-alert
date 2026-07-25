import Image from "next/image";
import { notFound } from "next/navigation";
import CategoryBadge from "@/components/CategoryBadge";
import DealTypeBadge from "@/components/DealTypeBadge";
import { formatDate, formatDealAmount } from "@/lib/format";
import { getArticleDetail } from "@/lib/microcms";

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
    <article>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <DealTypeBadge dealType={article.dealType} />
        <CategoryBadge category={article.category} />
      </div>
      <h1 className="mb-4 text-3xl font-bold text-gray-900">{article.title}</h1>
      <dl className="mb-6 grid grid-cols-2 gap-x-4 gap-y-2 rounded-lg bg-gray-50 p-4 text-sm sm:grid-cols-4">
        <div>
          <dt className="text-gray-500">銘柄</dt>
          <dd className="font-medium text-gray-900">
            {article.stockName}（{article.stockCode}）
          </dd>
        </div>
        <div>
          <dt className="text-gray-500">取引日</dt>
          <dd className="font-medium text-gray-900">{formatDate(article.dealDate)}</dd>
        </div>
        <div>
          <dt className="text-gray-500">金額規模</dt>
          <dd className="font-medium text-gray-900">
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
                className="font-medium text-blue-600 hover:underline"
              >
                元記事を見る
              </a>
            </dd>
          </div>
        )}
      </dl>
      {article.eyecatch && (
        <div className="relative mb-6 aspect-[16/9] w-full overflow-hidden rounded-lg bg-gray-100">
          <Image
            src={article.eyecatch.url}
            alt={article.title}
            fill
            sizes="(min-width: 768px) 768px, 100vw"
            className="object-cover"
            priority
          />
        </div>
      )}
      <div
        className="prose max-w-none"
        dangerouslySetInnerHTML={{ __html: article.body }}
      />
      {tags && tags.length > 0 && (
        <div className="mt-8 flex flex-wrap gap-2 border-t border-gray-200 pt-4">
          {tags.map((tag) => (
            <span
              key={tag}
              className="rounded-full bg-gray-100 px-2.5 py-0.5 text-xs text-gray-600"
            >
              #{tag}
            </span>
          ))}
        </div>
      )}
    </article>
  );
}
