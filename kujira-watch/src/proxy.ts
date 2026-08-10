import { NextResponse } from "next/server";
import type { NextFetchEvent, NextRequest } from "next/server";
import { classifyVisitor } from "@/lib/crawlers";
import { getSupabaseServerClient } from "@/lib/supabase";

const VISITOR_COOKIE = "kw_vid";

export function proxy(request: NextRequest, event: NextFetchEvent) {
  const userAgent = request.headers.get("user-agent") ?? "";
  const botName = classifyVisitor(userAgent);
  // NextRequest.ip はv15で削除されたため、Vercelが付与するヘッダーから取得する。
  const ipAddress =
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ??
    request.headers.get("x-real-ip") ??
    null;

  const response = NextResponse.next();
  let visitorId = request.cookies.get(VISITOR_COOKIE)?.value;
  if (!visitorId) {
    visitorId = crypto.randomUUID();
    response.cookies.set(VISITOR_COOKIE, visitorId, {
      httpOnly: true,
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 365,
    });
  }

  if (botName) {
    event.waitUntil(
      (async () => {
        try {
          await getSupabaseServerClient().from("blog_crawler_log").insert({
            path: request.nextUrl.pathname,
            bot_name: botName,
            user_agent: userAgent,
            visitor_id: botName === "Browser" ? visitorId : null,
            ip_address: ipAddress,
          });
        } catch {
          // ログ記録の失敗でサイト表示に影響を出さないよう握りつぶす
        }
      })()
    );
  }

  return response;
}

export const config = {
  matcher: "/((?!_next/static|_next/image|favicon.ico).*)",
};
