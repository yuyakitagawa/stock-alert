import { NextResponse } from "next/server";
import { fetchRecentEarnings } from "@/lib/data";

export const revalidate = 86400;

export async function GET(_: Request, { params }: { params: Promise<{ code: string }> }) {
  const { code } = await params;
  const data = await fetchRecentEarnings(code);
  return NextResponse.json(data);
}
