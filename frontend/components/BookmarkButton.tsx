"use client";
import { useBookmarks } from "@/lib/bookmarks";

interface Props {
  code: string;
  /** icon = アイコンのみ（一覧/カード用） / labeled = ラベル付きボタン（詳細ページ用） */
  variant?: "icon" | "labeled";
  className?: string;
}

function BookmarkIcon({ filled }: { filled: boolean }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
    </svg>
  );
}

export default function BookmarkButton({ code, variant = "icon", className = "" }: Props) {
  const { isBookmarked, toggle, mounted } = useBookmarks();
  const active = mounted && isBookmarked(code);

  // カード等の <Link> 内に置かれても遷移させない
  const onClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    toggle(code);
  };

  const label = active ? "ブックマーク済み" : "ブックマークに追加";

  if (variant === "labeled") {
    return (
      <button
        type="button"
        onClick={onClick}
        aria-pressed={active}
        title={label}
        className={`inline-flex items-center gap-1.5 text-sm font-medium px-3 py-1.5 rounded-lg border transition-colors ${
          active
            ? "bg-green-500/10 border-green-700 text-green-400 hover:bg-green-500/20"
            : "bg-gray-800 border-gray-700 text-gray-400 hover:text-white hover:border-gray-600"
        } ${className}`}
      >
        <BookmarkIcon filled={active} />
        {active ? "保存済み" : "保存"}
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      aria-label={label}
      title={label}
      className={`inline-flex items-center justify-center w-8 h-8 rounded-lg transition-colors ${
        active ? "text-green-400 hover:text-green-300" : "text-gray-600 hover:text-gray-300"
      } ${className}`}
    >
      <BookmarkIcon filled={active} />
    </button>
  );
}
