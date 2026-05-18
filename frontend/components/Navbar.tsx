import Link from "next/link";
import PushButton from "./PushButton";

interface Props {
  dateLabel?: string;
}

export default function Navbar({ dateLabel }: Props) {
  return (
    <header className="bg-gray-950/80 backdrop-blur-sm border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 shrink-0">
          <span className="text-green-400 text-xl leading-none">&#x1F4C8;</span>
          <span className="font-bold text-base tracking-tight text-white">
            Stock<span className="text-green-400">Signal</span>
          </span>
        </Link>

        {/* Nav links */}
        <nav className="hidden sm:flex items-center gap-6 text-sm font-medium">
          <Link
            href="/"
            className="text-gray-300 hover:text-white transition-colors"
          >
            ホーム
          </Link>
          <Link
            href="/rankings"
            className="text-gray-300 hover:text-white transition-colors"
          >
            ランキング
          </Link>
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-3">
          {dateLabel && (
            <span className="hidden md:block text-xs text-gray-500 tabular-nums">
              {dateLabel}
            </span>
          )}
          <PushButton />
        </div>
      </div>
    </header>
  );
}
