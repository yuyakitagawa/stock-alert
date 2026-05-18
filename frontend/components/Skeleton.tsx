export function SkeletonCard() {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-4 space-y-3 animate-pulse">
      <div className="flex justify-between items-start">
        <div className="space-y-1.5">
          <div className="h-3.5 bg-gray-800 rounded w-28" />
          <div className="h-3 bg-gray-800 rounded w-12" />
        </div>
        <div className="h-5 bg-gray-800 rounded-full w-14" />
      </div>
      <div className="h-7 bg-gray-800 rounded w-24" />
      <div className="space-y-1.5">
        <div className="h-3 bg-gray-800 rounded w-full" />
        <div className="h-1.5 bg-gray-800 rounded-full w-full" />
      </div>
      <div className="h-3 bg-gray-800 rounded w-full pt-2" />
    </div>
  );
}

export function SkeletonRow() {
  return (
    <div className="flex items-center gap-3 px-4 py-3 animate-pulse">
      <div className="h-3 bg-gray-800 rounded w-5" />
      <div className="flex-1 space-y-1.5">
        <div className="h-3.5 bg-gray-800 rounded w-32" />
        <div className="h-3 bg-gray-800 rounded w-16" />
      </div>
      <div className="h-5 bg-gray-800 rounded-full w-14" />
      <div className="h-3.5 bg-gray-800 rounded w-16" />
    </div>
  );
}

export function SkeletonSummary() {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5 animate-pulse space-y-2">
      <div className="h-8 bg-gray-800 rounded w-12" />
      <div className="h-3 bg-gray-800 rounded w-16" />
    </div>
  );
}
