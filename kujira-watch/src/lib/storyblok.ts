import { draftMode } from "next/headers";

// Storyblok（ビジュアル編集CMS）からTOP・サイト設定を読む。記事はmicroCMSのまま。
// SDK(@storyblok/react)は入れず、CDN APIを直接叩く。使うのは数ストーリーの取得と
// ビジュアルエディタ用の属性付けだけで、SDKのクライアント描画一式は要らないため。
//
// 無料のStarterプランはAPIが月10万リクエストまで（超過分の購入不可）。公開版は
// Next.jsのデータキャッシュに載せ、Storyblokの公開Webhook（/api/storyblok/revalidate）で
// 破棄されるまで再取得しない。下書き（ビジュアルエディタ内のプレビュー）だけ毎回取得する。

export const STORYBLOK_CACHE_TAG = "storyblok";
const API_BASE = "https://api.storyblok.com/v2/cdn";
// Webhookが届かなかった場合の保険。1日1回は公開版を取り直す。
const FALLBACK_REVALIDATE_SECONDS = 60 * 60 * 24;

export type StoryblokBlok = {
  _uid: string;
  component: string;
  _editable?: string;
  [key: string]: unknown;
};

export async function isStoryblokDraft(): Promise<boolean> {
  return (await draftMode()).isEnabled;
}

// ストーリーが無い・トークン未設定・APIエラーのときは null を返す。呼び出し側は
// コード内の既定値で描画するので、Storyblokが落ちてもサイトは現行の見た目で表示される。
export async function getStoryContent(slug: string): Promise<StoryblokBlok | null> {
  const draft = await isStoryblokDraft();
  const token = draft ? process.env.STORYBLOK_PREVIEW_TOKEN : process.env.STORYBLOK_PUBLIC_TOKEN;
  if (!token) return null;

  const params = new URLSearchParams({ token, version: draft ? "draft" : "published" });
  const url = `${API_BASE}/stories/${encodeURIComponent(slug)}?${params}`;
  try {
    const res = await fetch(
      url,
      draft
        ? { cache: "no-store" }
        : { next: { tags: [STORYBLOK_CACHE_TAG], revalidate: FALLBACK_REVALIDATE_SECONDS } }
    );
    if (!res.ok) return null;
    const json = (await res.json()) as { story?: { content?: StoryblokBlok } };
    return json.story?.content ?? null;
  } catch {
    return null;
  }
}

// ビジュアルエディタでブロックをクリック選択できるようにする属性。
// Storyblokは下書きの各ブロックに `<!--#storyblok#{...}-->` 形式の _editable を付けて返す。
// 公開版のレスポンスにも付いてくるため、Draft Mode時以外は何も返さない（本番HTMLを汚さない）。
export function editableAttrs(
  blok: StoryblokBlok | null | undefined,
  draft: boolean
): Record<string, string> {
  const raw = blok?._editable;
  if (!draft || !raw) return {};
  const json = raw.replace(/^<!--#storyblok#/, "").replace(/-->$/, "");
  try {
    const opts = JSON.parse(json) as { id: string; uid: string };
    return { "data-blok-c": json, "data-blok-uid": `${opts.id}-${opts.uid}` };
  } catch {
    return {};
  }
}

export function textField(blok: StoryblokBlok | null | undefined, key: string): string {
  const v = blok?.[key];
  return typeof v === "string" ? v.trim() : "";
}

// 編集者が入力したリンク先。サイト内パス（/始まり）か https のみ通す
// （javascript: などのスキームを描画しないため）。
export function safeHref(value: string): string | null {
  if (value.startsWith("/") && !value.startsWith("//")) return value;
  if (/^https:\/\//.test(value)) return value;
  return null;
}
