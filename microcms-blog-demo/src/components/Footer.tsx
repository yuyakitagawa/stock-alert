import Link from "next/link";
import { SITE_NAME } from "@/lib/site";

export default function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-gray-200 bg-brand-navy py-6 text-center text-xs text-white/70">
      <p>
        本サイトはEDINET大量保有報告書等の公開情報をもとにした解説であり、投資助言ではありません。
      </p>
      <p className="mt-2">
        <Link href="/about" className="underline hover:text-white">
          運営者情報・免責事項
        </Link>
      </p>
      <p className="mt-2">
        © {year} {SITE_NAME}
      </p>
    </footer>
  );
}
