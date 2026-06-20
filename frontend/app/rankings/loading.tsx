import { SkeletonRow } from "@/components/Skeleton";

export default function Loading() {
  return (
    <div className="min-h-screen">
      <div className="h-14 bg-gray-950 border-b border-gray-800" />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {/* Tab skeleton */}
        <div className="flex gap-2 mb-6">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-8 w-20 bg-gray-800 rounded-lg animate-pulse" />
          ))}
        </div>
        {/* Rows skeleton */}
        <div className="border border-gray-800 rounded-xl overflow-hidden divide-y divide-gray-800">
          {Array.from({ length: 12 }).map((_, i) => (
            <SkeletonRow key={i} />
          ))}
        </div>
      </div>
    </div>
  );
}
