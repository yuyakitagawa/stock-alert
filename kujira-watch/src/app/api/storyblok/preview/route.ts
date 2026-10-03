import { draftMode } from "next/headers";
import { redirect } from "next/navigation";

// Storyblokのビジュアルエディタが iframe で開くプレビューURL。
// Storyblok側の Settings → Visual Editor に
//   https://kujira-watch.com/api/storyblok/preview?secret=<STORYBLOK_PREVIEW_SECRET>&slug=
// を登録する（末尾にストーリーのslugが付く）。Draft Modeを有効にして該当ページへ飛ばす。
const SLUG_TO_PATH: Record<string, string> = {
  home: "/",
  "site-settings": "/",
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const secret = process.env.STORYBLOK_PREVIEW_SECRET;
  if (!secret || url.searchParams.get("secret") !== secret) {
    return new Response("Invalid token", { status: 401 });
  }
  (await draftMode()).enable();
  const slug = (url.searchParams.get("slug") ?? "").replace(/^\/+|\/+$/g, "");
  redirect(SLUG_TO_PATH[slug] ?? "/");
}
