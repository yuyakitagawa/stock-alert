import Link from "next/link";
import Hero from "@/components/Hero";
import NoticeBar from "@/components/NoticeBar";
import ServiceCards from "@/components/ServiceCards";
import IRBanner from "@/components/IRBanner";
import NewsRow from "@/components/NewsRow";
import { getNewsList } from "@/lib/microcms";

export default async function HomePage() {
  const { contents } = await getNewsList({ limit: 5 });

  return (
    <div>
      <Hero />
      <NoticeBar />
      <ServiceCards />
      <IRBanner />
      <section className="mx-auto max-w-6xl px-4 py-16">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-brand-navy">ニュースリリース</h2>
          <Link href="/news" className="text-sm font-semibold text-brand-blue hover:underline">
            一覧を見る &rarr;
          </Link>
        </div>
        <div className="rounded-lg bg-white ring-1 ring-gray-200">
          {contents.map((news) => (
            <NewsRow key={news.id} news={news} />
          ))}
        </div>
      </section>
    </div>
  );
}
