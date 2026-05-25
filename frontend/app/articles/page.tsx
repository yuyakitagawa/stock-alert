import type { Metadata } from "next";
import Link from "next/link";
import { fetchArticles } from "@/lib/data";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "S買いシグナル銘柄 解説記事一覧",
  description: "AIが選んだS買いシグナル銘柄の最新ニュース・投資ポイントをわかりやすく解説。毎日更新。",
  keywords: ["S買いシグナル", "日本株", "AI株スクリーニング", "株式投資", "個人投資家"],
  openGraph: {
    title: "S買いシグナル銘柄 解説記事一覧 | StockSignal",
    description: "AIが選んだS買いシグナル銘柄の最新ニュース・投資ポイントを毎日解説。",
  },
};

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString("ja-JP", {
    year: "numeric", month: "long", day: "numeric",
  });
}

export default async function ArticlesPage() {
  const articles = await fetchArticles(30);

  return (
    <div className="min-h-screen bg-black text-white">
      <Navbar />
      <main className="max-w-3xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-bold mb-2">S買いシグナル 銘柄解説</h1>
        <p className="text-gray-400 text-sm mb-8">
          AIモデルがS買いシグナルを発令した銘柄を、最新ニュースとともに毎日解説します。
        </p>

        {articles.length === 0 ? (
          <p className="text-gray-500">記事はまだありません。</p>
        ) : (
          <ul className="space-y-4">
            {articles.map((a) => (
              <li key={a.slug}>
                <Link
                  href={`/articles/${a.slug}`}
                  className="block bg-gray-900 border border-gray-800 rounded-xl p-5 hover:border-gray-600 transition-colors"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <p className="text-xs text-gray-500 mb-1">{formatDate(a.signal_date)}</p>
                      <h2 className="text-base font-semibold text-white leading-snug line-clamp-2">
                        {a.title}
                      </h2>
                    </div>
                    {a.net_score != null && (
                      <span className="shrink-0 text-xs font-mono bg-emerald-900/60 text-emerald-400 border border-emerald-700 rounded px-2 py-1">
                        net +{a.net_score.toFixed(1)}%
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-2">{a.code} — {a.name}</p>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </main>
      <Footer />
    </div>
  );
}
