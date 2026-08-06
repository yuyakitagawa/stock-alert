import type { CompanyInfo } from "@/lib/companyInfo";
import { formatDate } from "@/lib/format";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="kicker text-foreground/40">{label}</dt>
      <dd className="mt-0.5 text-sm font-bold text-brand-navy">{value}</dd>
    </div>
  );
}

export default function CompanyInfoCard({ info }: { info: CompanyInfo }) {
  const stats: { label: string; value: string }[] = [];

  if (info.sector) stats.push({ label: "業種", value: info.sector });
  if (info.close !== null) {
    stats.push({ label: "終値", value: `${info.close.toLocaleString("ja-JP")}円` });
  }
  if (info.per !== null) stats.push({ label: "PER", value: `${info.per}倍` });
  if (info.pbr !== null) stats.push({ label: "PBR", value: `${info.pbr}倍` });
  if (info.pos52 !== null) {
    stats.push({ label: "52週レンジ位置", value: `${Math.round(info.pos52 * 100)}%` });
  }
  stats.push({
    label: "株主優待",
    value: info.hasYutai ? `あり${info.yutaiMonth ? `（${info.yutaiMonth}月権利）` : ""}` : "なし",
  });

  if (stats.length === 0) return null;

  return (
    <div className="mb-6 border border-rule bg-paper p-4 sm:p-5">
      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        {stats.map((stat) => (
          <Stat key={stat.label} {...stat} />
        ))}
      </dl>
      {info.closeDate && (
        <p className="mt-3 text-xs text-foreground/40">
          株価・指標は{formatDate(info.closeDate)}時点
        </p>
      )}
    </div>
  );
}
