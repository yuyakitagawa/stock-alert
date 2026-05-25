import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import { fetchArticle, fetchArticles } from "@/lib/data";
import Navbar from "@/components/Navbar";

export const revalidate = 3600;

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  const articles = await fetchArticles(50);
  return articles.map((a) => ({ slug: a.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const article = await fetchArticle(slug);
  if (!article) return { title: "記事が見つかりません | StockSignal" };

  const description = article.meta_description
    ?? `${article.name}（${article.code}）にS買いシグナル。最新ニュースと投資ポイントを解説。ネットスコア ${article.net_score?.toFixed(1) ?? "—"}%。`;

  return {
    title: article.title,
    description,
    keywords: [article.name, article.code, "S買いシグナル", "日本株", "AI株スクリーニング"],
    openGraph: {
      title: article.title,
      description,
      type: "article",
      publishedTime: article.published_at,
    },
    alternates: {
      canonical: `/articles/${slug}`,
    },
  };
}

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString("ja-JP", {
    year: "numeric", month: "long", day: "numeric",
  });
}

function MarkdownBody({ body }: { body: string }) {
  const lines = body.split("\n");
  const elements: React.ReactNode[] = [];
  let key = 0;

  for (const line of lines) {
    if (line.startsWith("# ")) {
      elements.push(<h1 key={key++} className="text-2xl font-bold mt-6 mb-3">{line.slice(2)}</h1>);
    } else if (line.startsWith("## ")) {
      elements.push(<h2 key={key++} className="text-lg font-semibold mt-5 mb-2 text-gray-100">{line.slice(3)}</h2>);
    } else if (line.startsWith("- ")) {
      elements.push(<li key={key++} className="ml-4 list-disc text-gray-300">{line.slice(2)}</li>);
    } else if (line.startsWith("---")) {
      elements.push(<hr key={key++} className="border-gray-700 my-6" />);
    } else if (line.startsWith("*") && line.endsWith("*")) {
      elements.push(<p key={key++} className="text-xs text-gray-500 mt-2">{line.replace(/\*/g, "")}</p>);
    } else if (line.trim() === "") {
      elements.push(<div key={key++} className="h-2" />);
    } else {
      elements.push(<p key={key++} className="text-gray-300 leading-relaxed">{line}</p>);
    }
  }

  return <div className="space-y-1">{elements}</div>;
}

export default async function ArticlePage({ params }: Props) {
  const { slug } = await params;
  const article = await fetchArticle(slug);
  if (!article) notFound();

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "NewsArticle",
    headline: article.title,
    datePublished: article.published_at,
    author: { "@type": "Organization", name: "StockSignal" },
    publisher: { "@type": "Organization", name: "StockSignal" },
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <Navbar />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <main className="max-w-2xl mx-auto px-4 py-10">
        <Link href="/articles" className="text-xs text-gray-500 hover:text-gray-300 mb-6 inline-block">
          ← 記事一覧
        </Link>

        <div className="mb-6">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-xs bg-emerald-900/60 text-emerald-400 border border-emerald-700 rounded px-2 py-0.5">
              🥇 S買いシグナル
            </span>
            {article.net_score != null && (
              <span className="text-xs font-mono text-gray-400">
                net +{article.net_score.toFixed(1)}%
              </span>
            )}
          </div>
          <p className="text-xs text-gray-500">{formatDate(article.signal_date)}</p>
        </div>

        <article className="prose-invert">
          <MarkdownBody body={article.body} />
        </article>

        <div className="mt-10 pt-6 border-t border-gray-800 flex items-center justify-between">
          <Link
            href={`/stocks/${article.code}`}
            className="inline-block bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-sm hover:border-gray-500 transition-colors"
          >
            {article.name}（{article.code}）の詳細チャートを見る →
          </Link>
          <Link href="/articles" className="text-xs text-gray-600 hover:text-gray-400">
            ← 記事一覧
          </Link>
        </div>
      </main>
      <footer className="border-t border-gray-900 mt-16 py-6 text-center text-xs text-gray-700">
        © 2026 StockSignal
      </footer>
    </div>
  );
}
