import { NextResponse } from "next/server";
import { fetchDailyQuote } from "@/lib/data";

export const revalidate = 3600;

export async function GET(_: Request, { params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  const data = await fetchDailyQuote(code);
  return NextResponse.json(data);
}
