import { revalidatePath } from "next/cache";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const secret = req.nextUrl.searchParams.get("secret");
  if (secret !== process.env.INTERNAL_SEND_SECRET) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  revalidatePath("/");
  revalidatePath("/rankings");
  return NextResponse.json({ revalidated: true, now: Date.now() });
}
