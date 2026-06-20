import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-6 px-4 text-center">
      <div className="space-y-2">
        <p className="text-6xl font-bold font-mono text-gray-800">404</p>
        <h1 className="text-xl font-bold text-white">ページが見つかりません</h1>
        <p className="text-sm text-gray-500">指定された銘柄またはページは存在しません。</p>
      </div>
      <Link
        href="/"
        className="px-5 py-2.5 bg-green-700 hover:bg-green-600 text-white rounded-lg text-sm font-medium transition-colors"
      >
        ホームに戻る
      </Link>
    </div>
  );
}
