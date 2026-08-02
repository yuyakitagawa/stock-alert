export default function DealDateHeading({ label }: { label: string }) {
  return (
    <h2 className="mb-4 flex items-center gap-2 text-sm font-bold text-gray-500">
      <span aria-hidden className="h-3 w-1 rounded-full bg-brand-blue" />
      {label}
    </h2>
  );
}
