// Storyblokのブロック定義（コンポーネント）と初期ストーリーを作る。何度実行しても同じ状態になる。
// 使うのはManagement API（STORYBLOK_MANAGEMENT_TOKEN = Storyblokの My account → Personal access tokens）。
//   node --env-file=.env.local scripts/storyblok-setup.mjs
// ブロックの種類・選択肢はコード側（src/lib/siteTheme.ts / src/app/(ja)/page.tsx）と一致させること。
// 初期ストーリーは下書きとして作るだけで公開はしない（公開はStoryblokの画面で行う）。

const API = "https://mapi.storyblok.com/v1";
const token = process.env.STORYBLOK_MANAGEMENT_TOKEN;
const previewSecret = process.env.STORYBLOK_PREVIEW_SECRET;
const revalidateSecret = process.env.STORYBLOK_REVALIDATE_SECRET;
const SITE = process.env.STORYBLOK_SITE_URL ?? "https://kujira-watch.com";
if (!token || !previewSecret || !revalidateSecret) {
  console.error("STORYBLOK_MANAGEMENT_TOKEN / STORYBLOK_PREVIEW_SECRET / STORYBLOK_REVALIDATE_SECRET が必要です");
  process.exit(1);
}

async function api(method, path, body) {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: { Authorization: token, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${method} ${path} -> ${res.status} ${text.slice(0, 300)}`);
  return text ? JSON.parse(text) : {};
}

const option = (display_name, options, default_value) => ({
  type: "option",
  display_name,
  use_uuid: false,
  options: options.map(([value, name]) => ({ value, name })),
  default_value,
  exclude_empty_option: true,
});

const TOP_SECTIONS = ["top_features", "top_trending", "top_featured", "top_latest", "notice"];

const COMPONENTS = [
  {
    name: "site_settings",
    display_name: "サイト設定",
    is_root: true,
    is_nestable: false,
    schema: {
      palette: option(
        "配色",
        [
          ["classic", "クラシック（現行：クリーム×紺×金）"],
          ["slate", "スレート（白×クールグレー）"],
          ["forest", "フォレスト（生成り×深緑）"],
          ["wine", "ワイン（温白×ワインレッド）"],
          ["ink", "インク（白黒の新聞調）"],
        ],
        "classic"
      ),
      heading_font: option("見出しの書体", [["sans", "ゴシック（現行）"], ["serif", "明朝"]], "sans"),
      card_style: option("カードの質感", [["flat", "フラット（罫線のみ・現行）"], ["raised", "浮き上がり（薄い影）"]], "flat"),
    },
  },
  {
    name: "home",
    display_name: "TOPページ",
    is_root: true,
    is_nestable: false,
    schema: {
      headline: { type: "text", display_name: "見出し（h1）", description: "空欄なら既定の見出し" },
      lead: { type: "textarea", display_name: "リード文", description: "空欄なら既定の文章" },
      body: {
        type: "bloks",
        display_name: "セクション（ドラッグで並べ替え）",
        restrict_components: true,
        component_whitelist: TOP_SECTIONS,
      },
    },
  },
  {
    name: "top_features",
    display_name: "わかること（3枚のカード）",
    is_nestable: true,
    schema: { heading: { type: "text", display_name: "見出し", description: "空欄なら「このデータベースでわかること」" } },
  },
  { name: "top_trending", display_name: "取引急増ランキング", is_nestable: true, schema: {} },
  { name: "top_featured", display_name: "注目記事", is_nestable: true, schema: {} },
  {
    name: "top_latest",
    display_name: "新着開示の一覧",
    is_nestable: true,
    schema: { heading: { type: "text", display_name: "見出し", description: "空欄なら「大量保有・売買の新着開示」" } },
  },
  {
    name: "notice",
    display_name: "お知らせ枠",
    is_nestable: true,
    schema: {
      title: { type: "text", display_name: "タイトル" },
      text: { type: "textarea", display_name: "本文" },
      link_label: { type: "text", display_name: "リンクの文言" },
      link_url: { type: "text", display_name: "リンク先", description: "/stocks のようなサイト内パス、または https:// から始まるURL" },
      tone: option("色", [["info", "青"], ["gold", "金"]], "info"),
    },
  },
];

const { space } = await api("GET", "/spaces/me");
const spaceId = space.id;
console.log(`space ${space.name} (${spaceId})`);

const { components: existing } = await api("GET", `/spaces/${spaceId}/components`);
for (const c of COMPONENTS) {
  const found = existing.find((e) => e.name === c.name);
  if (found) await api("PUT", `/spaces/${spaceId}/components/${found.id}`, { component: { ...found, ...c } });
  else await api("POST", `/spaces/${spaceId}/components`, { component: c });
  console.log(`component ${found ? "updated" : "created"}: ${c.name}`);
}

async function upsertStory(slug, name, content, keepBody) {
  const { stories } = await api("GET", `/spaces/${spaceId}/stories?with_slug=${slug}`);
  if (stories.length > 0) {
    const { story } = await api("GET", `/spaces/${spaceId}/stories/${stories[0].id}`);
    // 既に同じ種類で中身があるものは上書きしない（編集済みの内容を消さないため）。
    if (keepBody && story.content?.component === content.component) {
      console.log(`story kept: ${slug}`);
      return;
    }
    await api("PUT", `/spaces/${spaceId}/stories/${story.id}`, { story: { name, slug, content } });
    console.log(`story replaced: ${slug}`);
  } else {
    await api("POST", `/spaces/${spaceId}/stories`, { story: { name, slug, content } });
    console.log(`story created: ${slug}`);
  }
}

const uid = () => crypto.randomUUID();
await upsertStory(
  "home",
  "TOPページ",
  {
    component: "home",
    headline: "",
    lead: "",
    body: ["top_features", "top_trending", "top_featured", "top_latest"].map((component) => ({ _uid: uid(), component })),
  },
  true
);
await upsertStory(
  "site-settings",
  "サイト設定",
  { component: "site_settings", palette: "classic", heading_font: "sans", card_style: "flat" },
  true
);

// ビジュアルエディタのプレビューURL（末尾にストーリーのslugが付く）。
await api("PUT", `/spaces/${spaceId}`, {
  space: { domain: `${SITE}/api/storyblok/preview?secret=${previewSecret}&slug=` },
});
console.log("preview url set");

// 公開時にサイトのキャッシュを捨てるWebhook。
const endpoint = `${SITE}/api/storyblok/revalidate?secret=${revalidateSecret}`;
const { webhook_endpoints: hooks = [] } = await api("GET", `/spaces/${spaceId}/webhook_endpoints`);
const hook = { name: "kujira-watch revalidate", endpoint, actions: ["story.published", "story.unpublished", "story.deleted"] };
const sameName = hooks.find((h) => h.name === hook.name);
if (sameName) await api("PUT", `/spaces/${spaceId}/webhook_endpoints/${sameName.id}`, { webhook_endpoint: hook });
else await api("POST", `/spaces/${spaceId}/webhook_endpoints`, { webhook_endpoint: hook });
console.log(`webhook ${sameName ? "updated" : "created"}`);
