export default function DealDateHeading({ label }: { label: string }) {
  return (
    <h2 className="mb-5 flex items-center gap-3 font-serif text-base font-bold text-brand-navy">
      {label}
      <span aria-hidden className="h-px flex-1 bg-rule" />
    </h2>
  );
}
