import { NextResponse } from "next/server";
import type { NextFetchEvent, NextRequest } from "next/server";
import { classifyVisitor } from "@/lib/crawlers";
import { EN_HOST } from "@/lib/en";
import { getSupabaseServerClient } from "@/lib/supabase";

const VISITOR_COOKIE = "kw_vid";

// 英語版ホストでもルート直下から共通で配信するもの（rewriteの対象外）。
// _next/static・_next/image・favicon.ico は config.matcher 側で proxy 自体を通していない。
const SHARED_ROOT_PATHS = /^\/(?:api\/|icon(?:$|\/)|apple-icon(?:$|\/)|logo(?:$|\/)|ads\.txt$|manifest\.webmanifest$)/;

// 英語版（en.kujira-watch.com）は app/(en)/en 配下のルートを、ホスト名を見て
// パス先頭の /en を隠す形で配信する（公開URLは en.kujira-watch.com/articles/<id>、
// 実体は /en/articles/<id>）。next.config.ts の redirects() は rewrite より先に評価される
// ので、英語版ホストで /en/... を直接叩かれても既存の「/en/* → /」301で / に寄る
// （相対先なので同じホストに留まる）＝同じ内容が2つのURLに出ることはない。
function enRewritePath(pathname: string): string | null {
  if (pathname === "/en" || pathname.startsWith("/en/")) return null;
  if (SHARED_ROOT_PATHS.test(pathname)) return null;
  return pathname === "/" ? "/en" : `/en${pathname}`;
}

export function proxy(request: NextRequest, event: NextFetchEvent) {
  const userAgent = request.headers.get("user-agent") ?? "";
  const botName = classifyVisitor(userAgent);
  // NextRequest.ip はv15で削除されたため、Vercelが付与するヘッダーから取得する。
  const ipAddress =
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ??
    request.headers.get("x-real-ip") ??
    null;
  // Vercelでは host ヘッダーに公開ホスト名（kujira-watch.com / en.kujira-watch.com）が入る。
  // ポート付き（ローカルの localhost:3000）は落として比較する。
  const host = (request.headers.get("host") ?? "").split(":")[0].toLowerCase();

  const rewriteTo = host === EN_HOST ? enRewritePath(request.nextUrl.pathname) : null;
  let response: NextResponse;
  if (rewriteTo) {
    const url = request.nextUrl.clone();
    url.pathname = rewriteTo;
    response = NextResponse.rewrite(url);
  } else {
    response = NextResponse.next();
  }

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
            // path は公開URLのパス（英語版でも /en を付けない）。host 列で日英を区別する。
            path: request.nextUrl.pathname,
            host: host || null,
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
