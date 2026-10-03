import { revalidateTag } from "next/cache";
import { STORYBLOK_CACHE_TAG } from "@/lib/storyblok";

// Storyblokの公開Webhook。Settings → Webhooks に
//   https://kujira-watch.com/api/storyblok/revalidate?secret=<STORYBLOK_REVALIDATE_SECRET>
// を「Story published / unpublished」で登録する。公開版のキャッシュだけを捨てるので、
// Vercelのビルドは走らない（課金ゼロ構成を崩さない）。
export async function POST(request: Request) {
  const secret = process.env.STORYBLOK_REVALIDATE_SECRET;
  if (!secret || new URL(request.url).searchParams.get("secret") !== secret) {
    return Response.json({ ok: false }, { status: 401 });
  }
  revalidateTag(STORYBLOK_CACHE_TAG, "max");
  return Response.json({ ok: true });
}
