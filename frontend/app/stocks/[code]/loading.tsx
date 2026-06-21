import { SkeletonSummary } from "@/components/Skeleton";

export default function Loading() {
  return (
    <div className="min-h-screen">
      <div className="h-14 bg-gray-950 border-b border-gray-800" />
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 space-y-6 animate-pulse">
        <div className="h-3 bg-gray-800 rounded w-48" />
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-3">
          <div className="h-6 bg-gray-800 rounded w-48" />
          <div className="h-4 bg-gray-800 rounded w-32" />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <SkeletonSummary key={i} />
          ))}
        </div>
        <div className="h-32 bg-gray-900 border border-gray-800 rounded-xl" />
      </div>
    </div>
  );
}
