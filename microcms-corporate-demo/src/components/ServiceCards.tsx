const SERVICES = [
  {
    id: "service-gas",
    title: "ガス",
    description: "都市ガスの料金プラン・お申し込み・ガス機器の点検について。",
    accent: "bg-brand-blue",
  },
  {
    id: "service-electric",
    title: "電気",
    description: "電気とガスのセット割でおトクに。電力使用量の見える化サービスも。",
    accent: "bg-brand-navy",
  },
  {
    id: "service-life",
    title: "暮らしのサービス",
    description: "住まいの修理・リフォーム、家事代行など、くらしをまるごとサポート。",
    accent: "bg-brand-gold",
  },
  {
    id: "service-business",
    title: "法人のお客さま",
    description: "省エネ・脱炭素を支援するエネルギーマネジメントサービスをご提供。",
    accent: "bg-brand-green",
  },
];

export default function ServiceCards() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-16">
      <h2 className="mb-8 text-center text-2xl font-bold text-brand-navy">サービス一覧</h2>
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {SERVICES.map((service) => (
          <a
            key={service.id}
            id={service.id}
            href="#"
            className="group flex flex-col overflow-hidden rounded-lg bg-white shadow-sm ring-1 ring-gray-200 transition-shadow hover:shadow-md"
          >
            <div className={`h-1.5 w-full ${service.accent}`} />
            <div className="flex flex-1 flex-col gap-2 p-6">
              <h3 className="text-lg font-semibold text-brand-navy">{service.title}</h3>
              <p className="text-sm leading-relaxed text-gray-600">{service.description}</p>
              <span className="mt-auto pt-2 text-sm font-semibold text-brand-blue group-hover:underline">
                詳しく見る &rarr;
              </span>
            </div>
          </a>
        ))}
      </div>
    </section>
  );
}
