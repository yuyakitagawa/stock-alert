import Link from "next/link";

export default function Pagination({
  currentPage,
  totalPages,
  basePath,
}: {
  currentPage: number;
  totalPages: number;
  basePath: string;
}) {
  if (totalPages <= 1) return null;

  const pages = Array.from({ length: totalPages }, (_, i) => i + 1);

  return (
    <nav className="mt-8 flex items-center justify-center gap-2" aria-label="ページネーション">
      <Link
        href={`${basePath}?page=${Math.max(1, currentPage - 1)}`}
        aria-disabled={currentPage === 1}
        className={`rounded-full border bg-white px-3 py-1.5 text-sm ${
          currentPage === 1
            ? "pointer-events-none border-gray-100 text-gray-300"
            : "border-brand-blue/40 text-brand-blue hover:bg-brand-blue hover:text-white"
        }`}
      >
        前へ
      </Link>
      {pages.map((page) => (
        <Link
          key={page}
          href={`${basePath}?page=${page}`}
          className={`rounded-full border px-3 py-1.5 text-sm ${
            page === currentPage
              ? "border-brand-blue bg-brand-blue text-white"
              : "border-brand-blue/40 bg-white text-brand-blue hover:bg-brand-blue hover:text-white"
          }`}
        >
          {page}
        </Link>
      ))}
      <Link
        href={`${basePath}?page=${Math.min(totalPages, currentPage + 1)}`}
        aria-disabled={currentPage === totalPages}
        className={`rounded-full border bg-white px-3 py-1.5 text-sm ${
          currentPage === totalPages
            ? "pointer-events-none border-gray-100 text-gray-300"
            : "border-brand-blue/40 text-brand-blue hover:bg-brand-blue hover:text-white"
        }`}
      >
        次へ
      </Link>
    </nav>
  );
}
