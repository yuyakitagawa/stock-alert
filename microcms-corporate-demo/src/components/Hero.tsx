export default function Hero() {
  return (
    <section className="relative overflow-hidden bg-gradient-to-br from-brand-navy via-brand-blue-dark to-brand-blue">
      <div className="pointer-events-none absolute inset-0 opacity-20">
        <div className="absolute -left-16 -top-16 h-72 w-72 rounded-full bg-white blur-3xl" />
        <div className="absolute -bottom-24 right-0 h-96 w-96 rounded-full bg-brand-gold blur-3xl" />
      </div>
      <div className="relative mx-auto max-w-6xl px-4 py-20 sm:py-28">
        <p className="mb-3 text-sm font-semibold tracking-wide text-brand-gold">
          MIRAI ENERGY GROUP
        </p>
        <h1 className="max-w-2xl text-3xl font-bold leading-snug text-white sm:text-4xl">
          エネルギーで、くらしと産業の
          <br />
          あたらしい明日をつくる。
        </h1>
        <p className="mt-4 max-w-xl text-sm leading-relaxed text-white/80">
          ガス・電気の安定供給から、脱炭素社会に向けた次世代エネルギーの開発まで。
          みらいエネルギーグループは、お客さまと社会に寄り添うエネルギー企業を目指します。
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <a
            href="#service-gas"
            className="rounded-full bg-white px-6 py-3 text-sm font-semibold text-brand-navy transition-colors hover:bg-brand-gold hover:text-white"
          >
            料金プランを見る
          </a>
          <a
            href="#sustainability"
            className="rounded-full border border-white/60 px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-white/10"
          >
            サステナビリティの取り組み
          </a>
        </div>
      </div>
    </section>
  );
}
