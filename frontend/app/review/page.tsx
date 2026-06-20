import type { Metadata } from "next";
import type { ReactElement } from "react";
import { fetchWeeklyReviews, fetchActivity } from "@/lib/data";
import type { WeeklyReview, Activity } from "@/lib/types";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import ReviewActivityTabs from "@/components/ReviewActivityTabs";

export const revalidate = 30;

export const metadata: Metadata = {
  title: "チームレビュー & 活動ログ — StockSignal",
  description: "AIチームの週次相互評価レポートと活動記録",
};

export default async function ReviewPage() {
  const [reviews, activities] = await Promise.all([
    fetchWeeklyReviews(8),
    fetchActivity(80),
  ]);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 max-w-3xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white">チームレビュー & 活動ログ</h1>
          <p className="text-sm text-gray-600 mt-1">
            AIチームの週次評価レポートと、「今なにをしているか」の活動記録。
          </p>
        </div>
        <ReviewActivityTabs reviews={reviews} activities={activities} />
      </main>
      <Footer />
    </div>
  );
}
