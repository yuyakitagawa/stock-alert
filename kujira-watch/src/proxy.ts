import { NextResponse } from "next/server";
import type { NextFetchEvent, NextRequest } from "next/server";
import { classifyVisitor } from "@/lib/crawlers";
import { getSupabaseServerClient } from "@/lib/supabase";

export function proxy(request: NextRequest, event: NextFetchEvent) {
  const userAgent = request.headers.get("user-agent") ?? "";
  const botName = classifyVisitor(userAgent);

  if (botName) {
    event.waitUntil(
      (async () => {
        try {
          await getSupabaseServerClient().from("blog_crawler_log").insert({
            path: request.nextUrl.pathname,
            bot_name: botName,
            user_agent: userAgent,
          });
        } catch {
          // ログ記録の失敗でサイト表示に影響を出さないよう握りつぶす
        }
      })()
    );
  }

  return NextResponse.next();
}

export const config = {
  matcher: "/((?!_next/static|_next/image|favicon.ico).*)",
};
