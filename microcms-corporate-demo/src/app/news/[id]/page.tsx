import Link from "next/link";
import CategoryBadge from "@/components/CategoryBadge";
import { formatDate } from "@/lib/format";
import { getNewsDetail } from "@/lib/microcms";

export const revalidate = 60;

export default async function NewsDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const news = await getNewsDetail(id);

  return (
    <article className="mx-auto max-w-3xl px-4 py-16">
      <Link href="/news" className="text-sm text-brand-blue hover:underline">
        &larr; ニュースリリース一覧
      </Link>
      <div className="mt-4 flex items-center gap-3">
        <CategoryBadge category={news.category} />
        <span className="text-sm text-gray-500">
          {formatDate(news.publishedAt ?? news.createdAt)}
        </span>
      </div>
      <h1 className="mt-3 text-2xl font-bold text-brand-navy">{news.title}</h1>
      <div
        className="prose prose-sm mt-8 max-w-none text-gray-700"
        dangerouslySetInnerHTML={{ __html: news.body }}
      />
    </article>
  );
}
