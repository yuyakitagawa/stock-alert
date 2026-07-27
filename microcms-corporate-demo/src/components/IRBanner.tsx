export default function IRBanner() {
  return (
    <section className="bg-section-tint py-16">
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-6 px-4 sm:grid-cols-2">
        <a
          id="ir"
          href="#"
          className="flex flex-col justify-between rounded-lg bg-brand-navy p-8 text-white transition-opacity hover:opacity-90"
        >
          <div>
            <p className="text-xs font-semibold tracking-wide text-brand-gold">IR INFORMATION</p>
            <h3 className="mt-2 text-xl font-bold">株主・投資家の皆さまへ</h3>
            <p className="mt-3 text-sm leading-relaxed text-white/80">
              決算情報、有価証券報告書、株主総会に関する情報を掲載しています。
            </p>
          </div>
          <span className="mt-6 text-sm font-semibold underline">IR情報を見る &rarr;</span>
        </a>
        <a
          id="sustainability"
          href="#"
          className="flex flex-col justify-between rounded-lg bg-brand-green p-8 text-white transition-opacity hover:opacity-90"
        >
          <div>
            <p className="text-xs font-semibold tracking-wide text-white/80">SUSTAINABILITY</p>
            <h3 className="mt-2 text-xl font-bold">サステナビリティへの取り組み</h3>
            <p className="mt-3 text-sm leading-relaxed text-white/90">
              2050年カーボンニュートラルの実現に向けた環境・社会への取り組みをご紹介します。
            </p>
          </div>
          <span className="mt-6 text-sm font-semibold underline">詳しく見る &rarr;</span>
        </a>
      </div>
    </section>
  );
}
