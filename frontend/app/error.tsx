"use client";
import { useEffect } from "react";
import Link from "next/link";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-4 text-center">
      <div className="space-y-2">
        <p className="text-5xl">⚠️</p>
        <h1 className="text-xl font-bold text-white">エラーが発生しました</h1>
        <p className="text-sm text-gray-500">データの取得中に問題が発生しました。</p>
      </div>
      <div className="flex gap-3">
        <button
          onClick={reset}
          className="px-4 py-2 bg-green-700 hover:bg-green-600 text-white rounded-lg text-sm font-medium transition-colors"
        >
          再試行
        </button>
        <Link
          href="/"
          className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg text-sm font-medium transition-colors"
        >
          ホームへ
        </Link>
      </div>
    </div>
  );
}
