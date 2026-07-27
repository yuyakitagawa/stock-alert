import Link from "next/link";

const COLUMNS: { heading: string; links: string[] }[] = [
  { heading: "ガス・電気", links: ["ガス料金プラン", "電気料金プラン", "引っ越し手続き", "料金シミュレーション"] },
  { heading: "暮らしのサービス", links: ["住まいの工事・修理", "ガス機器のご紹介", "会員サービス", "よくあるご質問"] },
  { heading: "企業情報", links: ["会社概要", "事業内容", "採用情報", "プレスリリース"] },
  { heading: "IR・サステナビリティ", links: ["IR情報", "株主・投資家の皆さま", "サステナビリティ", "環境への取り組み"] },
];

export default function Footer() {
  return (
    <footer id="contact" className="bg-brand-navy text-white">
      <div className="mx-auto max-w-6xl px-4 py-10">
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-4">
          {COLUMNS.map((column) => (
            <div key={column.heading}>
              <h3 className="mb-3 text-sm font-semibold text-white/90">{column.heading}</h3>
              <ul className="space-y-2 text-xs text-white/70">
                {column.links.map((link) => (
                  <li key={link}>
                    <Link href="#" className="hover:text-white hover:underline">
                      {link}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="mt-10 flex flex-col items-start justify-between gap-4 border-t border-white/10 pt-6 text-xs text-white/60 sm:flex-row sm:items-center">
          <p>お問い合わせ：0120-XXX-XXX（受付時間 9:00〜17:00）</p>
          <p>&copy; みらいエネルギー株式会社（本サイトはデモです）</p>
        </div>
      </div>
    </footer>
  );
}
