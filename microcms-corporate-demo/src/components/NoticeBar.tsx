export default function NoticeBar() {
  return (
    <div className="border-y border-brand-gold/30 bg-amber-50">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-2 px-4 py-2.5 text-xs text-amber-900 sm:gap-3">
        <span className="flex-shrink-0 rounded-full bg-brand-gold px-2.5 py-0.5 font-semibold text-white">
          お知らせ
        </span>
        <span>台風接近に伴うお客さまセンターの営業時間変更について</span>
      </div>
    </div>
  );
}
