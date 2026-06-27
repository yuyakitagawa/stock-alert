import Anthropic from "https://esm.sh/@anthropic-ai/sdk@0.39.0";

const LINE_CHANNEL_SECRET = Deno.env.get("LINE_CHANNEL_SECRET") ?? "";
const LINE_CHANNEL_ACCESS_TOKEN = Deno.env.get("LINE_CHANNEL_ACCESS_TOKEN") ?? "";
const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY") ?? "";

const SB_URL = (Deno.env.get("SUPABASE_URL") ?? "").trim();
const SB_KEY = (Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "").trim();

function sbHeaders(extra: Record<string, string> = {}) {
  return {
    apikey: SB_KEY,
    Authorization: `Bearer ${SB_KEY}`,
    "Content-Type": "application/json",
    ...extra,
  };
}

async function verifySignature(body: string, signature: string): Promise<boolean> {
  if (!LINE_CHANNEL_SECRET) return false;
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(LINE_CHANNEL_SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(body));
  const hash = btoa(String.fromCharCode(...new Uint8Array(sig)));
  return hash === signature;
}

async function replyToLine(replyToken: string, text: string): Promise<void> {
  const maxLen = 5000;
  const truncated = text.length > maxLen ? text.slice(0, maxLen - 3) + "..." : text;
  await fetch("https://api.line.me/v2/bot/message/reply", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${LINE_CHANNEL_ACCESS_TOKEN}`,
    },
    body: JSON.stringify({
      replyToken,
      messages: [{ type: "text", text: truncated }],
    }),
  });
}

// ── ウォッチリスト管理 ────────────────────────

interface WatchItem {
  code: string;
  name: string;
  dp_threshold: number;
  dp_sell_threshold: number;
  line_user_id: string;
}

async function getWatchlist(userId: string): Promise<WatchItem[]> {
  const res = await fetch(
    `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&select=code,name,dp_threshold,dp_sell_threshold&order=created_at`,
    { headers: sbHeaders() },
  );
  return res.ok ? await res.json() : [];
}

async function addToWatchlist(
  userId: string,
  code: string,
  name: string,
  threshold: number,
  sellThreshold: number = 20.0,
): Promise<boolean> {
  const res = await fetch(`${SB_URL}/rest/v1/dp_watchlist`, {
    method: "POST",
    headers: sbHeaders({ Prefer: "resolution=merge-duplicates" }),
    body: JSON.stringify({
      line_user_id: userId,
      code,
      name,
      dp_threshold: threshold,
      dp_sell_threshold: sellThreshold,
    }),
  });
  return res.ok;
}

async function removeFromWatchlist(userId: string, code: string): Promise<boolean> {
  const res = await fetch(
    `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&code=eq.${code}`,
    { method: "DELETE", headers: sbHeaders() },
  );
  return res.ok;
}

async function lookupStock(
  code: string,
): Promise<{ code: string; name: string; close: number; drop_prob: number } | null> {
  const res = await fetch(
    `${SB_URL}/rest/v1/gen_rankings?code=eq.${code}&select=code,name,close,drop_prob&order=date.desc&limit=1`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return null;
  const rows = await res.json();
  return rows.length > 0 ? rows[0] : null;
}

function toFullWidth(s: string): string {
  return s.replace(/[A-Za-z0-9]/g, (c) => String.fromCharCode(c.charCodeAt(0) + 0xfee0));
}

async function searchStockByName(keyword: string): Promise<{ code: string; name: string }[]> {
  const queries = [keyword];
  const fw = toFullWidth(keyword);
  if (fw !== keyword) queries.push(fw);

  for (const q of queries) {
    const res = await fetch(
      `${SB_URL}/rest/v1/jpx_stock_list?name=ilike.*${encodeURIComponent(q)}*&select=code,name&limit=5`,
      { headers: sbHeaders() },
    );
    const rows = res.ok ? await res.json() : [];
    if (rows.length > 0) return rows;
  }
  return [];
}

// ── コマンド処理 ─────────────────────────────

type CommandResult = { handled: true; reply: string } | { handled: false };

async function handleCommand(text: string, userId: string): Promise<CommandResult> {
  const trimmed = text.trim();

  if (/^(ウォッチ|リスト|一覧)$/i.test(trimmed)) {
    const list = await getWatchlist(userId);
    if (list.length === 0) {
      return { handled: true, reply: "ウォッチリストは空です。\n「ウォッチ 8473」で追加できます。" };
    }
    const lines = ["📋 ウォッチリスト:"];
    for (const item of list) {
      const stock = await lookupStock(item.code);
      const dpStr = stock ? `下落確率${stock.drop_prob}%` : "";
      const priceStr = stock ? `${stock.close.toLocaleString()}円` : "";
      let status = "";
      if (stock) {
        if (stock.drop_prob < item.dp_threshold) status = "🔔買い時！";
        else if (stock.drop_prob >= (item.dp_sell_threshold ?? 20)) status = "⚠️売り検討";
      }
      lines.push(`  ${item.code} ${item.name} ${priceStr} ${dpStr} ${status}`);
    }
    lines.push(`\n※ 下落確率が買い閾値未満で買い通知、売り閾値以上で売り通知`);
    return { handled: true, reply: lines.join("\n") };
  }

  const addMatch = trimmed.match(
    /^ウォッチ\s+(.+?)(?:\s+(\d+(?:\.\d+)?))?(?:\s+(\d+(?:\.\d+)?))?$/i,
  );
  if (addMatch) {
    const target = addMatch[1].trim();
    const threshold = addMatch[2] ? parseFloat(addMatch[2]) : 8.0;
    const sellThreshold = addMatch[3] ? parseFloat(addMatch[3]) : 20.0;

    if (/^\d{4}$/.test(target)) {
      const stock = await lookupStock(target);
      if (!stock) {
        return { handled: true, reply: `${target} はランキングに見つかりません。銘柄コード4桁で指定してください。` };
      }
      await addToWatchlist(userId, stock.code, stock.name, threshold, sellThreshold);
      return {
        handled: true,
        reply:
          `✅ ${stock.name}(${stock.code}) をウォッチリストに追加\n` +
          `株価: ${stock.close.toLocaleString()}円\n` +
          `下落確率: ${stock.drop_prob}%\n` +
          `\n※ 下落確率 < ${threshold}%で買い通知 / ≥ ${sellThreshold}%で売り通知`,
      };
    }

    const matches = await searchStockByName(target);
    if (matches.length === 0) {
      return { handled: true, reply: `「${target}」に該当する銘柄が見つかりません。コード4桁で指定してください。` };
    }
    if (matches.length === 1) {
      const stock = await lookupStock(matches[0].code);
      const name = stock?.name ?? matches[0].name;
      await addToWatchlist(userId, matches[0].code, name, threshold, sellThreshold);
      return {
        handled: true,
        reply:
          `✅ ${name}(${matches[0].code}) をウォッチリストに追加\n` +
          (stock ? `株価: ${stock.close.toLocaleString()}円\n下落確率: ${stock.drop_prob}%\n` : "") +
          `\n※ 下落確率 < ${threshold}%で買い通知 / ≥ ${sellThreshold}%で売り通知`,
      };
    }
    const options = matches.map((m) => `  ${m.code} ${m.name}`).join("\n");
    return {
      handled: true,
      reply: `「${target}」に複数該当:\n${options}\n\nコードで指定してください。\n例: ウォッチ ${matches[0].code}`,
    };
  }

  const removeMatch = trimmed.match(/^(解除|削除|外す)\s+(\d{4})$/i);
  if (removeMatch) {
    const code = removeMatch[2];
    await removeFromWatchlist(userId, code);
    return { handled: true, reply: `🗑 ${code} をウォッチリストから削除しました。` };
  }

  if (/^\d{4}$/.test(trimmed)) {
    const stock = await lookupStock(trimmed);
    if (!stock) {
      return { handled: true, reply: `${trimmed} のデータが見つかりません。` };
    }
    const dpStatus =
      stock.drop_prob < 8 ? "🟢安全圏" : stock.drop_prob >= 15 ? "🔴危険" : "🟡通常";
    return {
      handled: true,
      reply:
        `📊 ${stock.name}(${stock.code})\n` +
        `株価: ${stock.close.toLocaleString()}円\n` +
        `下落確率: ${stock.drop_prob}% ${dpStatus}\n\n` +
        `「ウォッチ ${stock.code}」でウォッチに追加`,
    };
  }

  return { handled: false };
}

// ── 会話履歴 ─────────────────────────────────

async function saveMessage(userId: string, role: "user" | "assistant", content: string): Promise<void> {
  if (!SB_URL || !SB_KEY) return;
  await fetch(`${SB_URL}/rest/v1/line_chat_history`, {
    method: "POST",
    headers: sbHeaders({ Prefer: "return=minimal" }),
    body: JSON.stringify({ line_user_id: userId, role, content: content.slice(0, 2000) }),
  });
  const res = await fetch(
    `${SB_URL}/rest/v1/line_chat_history?line_user_id=eq.${userId}&order=created_at.desc&select=id&offset=6&limit=100`,
    { headers: sbHeaders() },
  );
  if (res.ok) {
    const old = await res.json();
    if (old.length > 0) {
      const ids = old.map((r: { id: string }) => r.id);
      await fetch(`${SB_URL}/rest/v1/line_chat_history?id=in.(${ids.join(",")})`, {
        method: "DELETE",
        headers: sbHeaders(),
      });
    }
  }
}

async function getRecentMessages(userId: string): Promise<{ role: string; content: string }[]> {
  if (!SB_URL || !SB_KEY) return [];
  const res = await fetch(
    `${SB_URL}/rest/v1/line_chat_history?line_user_id=eq.${userId}&order=created_at.desc&limit=6&select=role,content`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return [];
  const rows = await res.json();
  return rows.reverse();
}

// ── 企業情報データ取得 ────────────────────────

async function fetchTdnetDisclosures(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";
  const res = await fetch(
    `${SB_URL}/rest/v1/ext_tdnet_disclosures?code=eq.${code}&order=disclosed_at.desc&select=disclosed_at,title,category&limit=5`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return "";
  const rows = await res.json();
  if (rows.length === 0) return "";
  const lines = ["【適時開示（TDnet）】"];
  for (const r of rows) {
    const d = r.disclosed_at?.slice(0, 10) ?? "";
    const cat = r.category ? `[${r.category}]` : "";
    lines.push(`  ${d} ${cat} ${r.title}`);
  }
  return lines.join("\n");
}

async function fetchJpxShortSelling(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";
  const res = await fetch(
    `${SB_URL}/rest/v1/jpx_short_selling?code=eq.${code}&order=calc_date.desc&select=calc_date,short_seller,short_ratio&limit=5`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return "";
  const rows = await res.json();
  if (rows.length === 0) return "";
  const lines = ["【空売り残高（JPX）】"];
  for (const r of rows) {
    lines.push(`  ${r.calc_date} ${r.short_seller}: ${r.short_ratio}%`);
  }
  return lines.join("\n");
}

async function fetchJpxMarginBalance(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";
  const res = await fetch(
    `${SB_URL}/rest/v1/jpx_margin_balance?code=eq.${code}&order=record_date.desc&select=record_date,margin_buy,margin_sell&limit=3`,
    { headers: sbHeaders() },
  );
  if (!res.ok) return "";
  const rows = await res.json();
  if (rows.length === 0) return "";
  const lines = ["【信用残高（JPX）】"];
  for (const r of rows) {
    lines.push(`  ${r.record_date} 買残: ${Number(r.margin_buy).toLocaleString()} / 売残: ${Number(r.margin_sell).toLocaleString()}`);
  }
  return lines.join("\n");
}

// ── 市場データ取得 ───────────────────────────

async function fetchStockDetail(code: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";

  const [rankRes, finRes, tdnetStr, shortStr, marginStr] = await Promise.all([
    fetch(
      `${SB_URL}/rest/v1/gen_rankings?code=eq.${code}&select=code,name,close,drop_prob,per,pbr,piotroski,bps_growth,eps_surprise,vol,rel20&order=date.desc&limit=5`,
      { headers: sbHeaders() },
    ),
    fetch(
      `${SB_URL}/rest/v1/jquants_fin_summary?code=eq.${code}&select=code,disc_date,doc_type,eps,bps,div_ann,payout_ratio,np,cfo,ta,equity,op,sales,fnp,fop,fsales&order=disc_date.desc&limit=4`,
      { headers: sbHeaders() },
    ),
    fetchTdnetDisclosures(code),
    fetchJpxShortSelling(code),
    fetchJpxMarginBalance(code),
  ]);

  const ranks = rankRes.ok ? await rankRes.json() : [];
  const fins = finRes.ok ? await finRes.json() : [];
  const lines: string[] = [];

  if (ranks.length > 0) {
    const r = ranks[0];
    lines.push(`【${r.name}(${r.code})の詳細データ】`);
    lines.push(`株価: ${r.close}円, 下落確率: ${r.drop_prob}%`);
    lines.push(`PER: ${r.per}, PBR: ${r.pbr}, Piotroski: ${r.piotroski}`);
    lines.push(`BPS成長: ${r.bps_growth}, EPSサプライズ: ${r.eps_surprise}`);
    lines.push(`出来高: ${r.vol}, 20日相対強度: ${r.rel20}`);
    if (ranks.length > 1) {
      lines.push(`下落確率の直近5日推移: ${ranks.map((x: { drop_prob: number }) => x.drop_prob + "%").join(" → ")}`);
    }
  }
  if (fins.length > 0) {
    const f = fins[0];
    const close = ranks.length > 0 ? Number(ranks[0].close) : 0;
    const divYield = close > 0 && f.div_ann ? ((f.div_ann / close) * 100).toFixed(2) : null;
    const roe = f.equity && f.equity > 0 && f.np ? ((f.np / f.equity) * 100).toFixed(1) : null;
    const equityRatio = f.ta && f.ta > 0 && f.equity ? ((f.equity / f.ta) * 100).toFixed(1) : null;
    const opMargin = f.sales && f.sales > 0 && f.op ? ((f.op / f.sales) * 100).toFixed(1) : null;

    lines.push(`\n【決算データ(${f.disc_date} ${f.doc_type})】`);
    lines.push(`EPS: ${f.eps}, BPS: ${f.bps}`);
    if (divYield)
      lines.push(
        `配当利回り: ${divYield}%, 年間配当: ${f.div_ann}円, 配当性向: ${f.payout_ratio ? (f.payout_ratio * 100).toFixed(1) + "%" : "N/A"}`,
      );
    if (roe) lines.push(`ROE: ${roe}%`);
    if (equityRatio) lines.push(`自己資本比率: ${equityRatio}%`);
    if (opMargin) lines.push(`営業利益率: ${opMargin}%`);
    if (f.cfo != null) lines.push(`営業CF: ${(f.cfo / 1e6).toFixed(0)}百万円`);
    if (f.fnp != null && f.np != null && f.fnp > 0) {
      const progress = ((f.np / f.fnp) * 100).toFixed(1);
      lines.push(
        `通期予想進捗率: ${progress}% (実績NP: ${(f.np / 1e6).toFixed(0)}百万 / 予想: ${(f.fnp / 1e6).toFixed(0)}百万)`,
      );
    }

    const fyRecords = fins.filter((x: { doc_type: string }) => x.doc_type === "FY");
    if (fyRecords.length >= 2) {
      const cur = fyRecords[0];
      const prev = fyRecords[1];
      if (cur.np != null && prev.np != null && prev.np !== 0) {
        const npGrowth = (((cur.np - prev.np) / Math.abs(prev.np)) * 100).toFixed(1);
        lines.push(`純利益成長率(前期比): ${npGrowth}%`);
      }
      if (cur.sales != null && prev.sales != null && prev.sales !== 0) {
        const salesGrowth = (((cur.sales - prev.sales) / Math.abs(prev.sales)) * 100).toFixed(1);
        lines.push(`売上成長率(前期比): ${salesGrowth}%`);
      }
    }
  }

  if (tdnetStr) lines.push("\n" + tdnetStr);
  if (shortStr) lines.push("\n" + shortStr);
  if (marginStr) lines.push("\n" + marginStr);

  return lines.join("\n");
}

async function fetchMarketContext(userId: string, userMessage: string): Promise<string> {
  if (!SB_URL || !SB_KEY) return "";

  const today = new Date().toISOString().slice(0, 10);

  const [rankRes, n225Res, watchRes] = await Promise.all([
    fetch(
      `${SB_URL}/rest/v1/gen_rankings?date=eq.${today}&select=code,name,close,drop_prob,per,pbr&order=drop_prob.asc&limit=20`,
      { headers: sbHeaders() },
    ),
    fetch(
      `${SB_URL}/rest/v1/yahoo_market_index?ticker=eq.N225&order=date.desc&limit=1&select=date,close`,
      { headers: sbHeaders() },
    ),
    fetch(
      `${SB_URL}/rest/v1/dp_watchlist?line_user_id=eq.${userId}&select=code,name,dp_threshold,dp_sell_threshold`,
      { headers: sbHeaders() },
    ),
  ]);

  const rankings = rankRes.ok ? await rankRes.json() : [];
  const n225 = n225Res.ok ? await n225Res.json() : [];
  const watchlist: WatchItem[] = watchRes.ok ? await watchRes.json() : [];

  const lines: string[] = [];

  if (n225.length > 0) {
    lines.push(`N225: ${n225[0].close}円 (${n225[0].date})`);
  }

  if (rankings.length > 0) {
    const dps = rankings
      .filter((r: { drop_prob: number | null }) => r.drop_prob != null)
      .map((r: { drop_prob: number }) => r.drop_prob);
    const avgDp = dps.length > 0 ? dps.reduce((a: number, b: number) => a + b, 0) / dps.length : null;

    if (avgDp != null) {
      const signal = avgDp >= 15 ? "🔴キャッシュ推奨" : "🟢投資継続OK";
      lines.push(`市場平均下落確率: ${avgDp.toFixed(1)}% → ${signal}`);
    }

    lines.push(`\n本日のトップ10:`);
    for (const r of rankings.slice(0, 10)) {
      lines.push(`  ${r.code} ${r.name}: 下落確率=${r.drop_prob}% PER=${r.per} PBR=${r.pbr}`);
    }
  }

  if (watchlist.length > 0) {
    lines.push(`\nこのユーザーのウォッチリスト:`);
    for (const w of watchlist) {
      const stock = rankings.find((r: { code: string }) => r.code === w.code);
      const dpStr = stock ? `下落確率=${stock.drop_prob}%` : "下落確率=--";
      lines.push(`  ${w.code} ${w.name}: ${dpStr} (買<${w.dp_threshold} 売≥${w.dp_sell_threshold ?? 20})`);
    }
  }

  const codeMatch = userMessage.match(/(\d{4})/);
  if (codeMatch) {
    const detail = await fetchStockDetail(codeMatch[1]);
    if (detail) lines.push("\n" + detail);
  }

  return lines.join("\n");
}

// ── Claude API (tool use) ─────────────────────

const TOOLS: Anthropic.Tool[] = [
  {
    name: "add_watchlist",
    description:
      "ウォッチリストに銘柄を追加する。ユーザーが「〇〇をウォッチして」「〇〇を監視したい」「〇〇が下がったら教えて」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "銘柄コード(4桁)または銘柄名の一部。例: '8473', 'SBI', '三菱UFJ'" },
        buy_threshold: { type: "number", description: "買い閾値(dp%)。指定なければ8.0" },
        sell_threshold: { type: "number", description: "売り閾値(dp%)。指定なければ20.0" },
      },
      required: ["query"],
    },
  },
  {
    name: "remove_watchlist",
    description: "ウォッチリストから銘柄を削除する。「〇〇を外して」「〇〇の監視やめて」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "銘柄コード(4桁)または銘柄名の一部" },
      },
      required: ["query"],
    },
  },
  {
    name: "show_watchlist",
    description: "ウォッチリストの一覧を表示する。「ウォッチリスト見せて」「今何を監視してる？」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {},
    },
  },
  {
    name: "lookup_stock",
    description: "特定銘柄の現在の詳細情報を調べる。「〇〇の状況は？」「〇〇って今どう？」等と言ったとき使う。",
    input_schema: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "銘柄コード(4桁)または銘柄名の一部" },
      },
      required: ["query"],
    },
  },
];

async function resolveCode(
  query: string,
): Promise<{ code: string; name: string } | { candidates: { code: string; name: string }[] } | null> {
  if (/^\d{4}$/.test(query)) {
    const stock = await lookupStock(query);
    return stock ? { code: stock.code, name: stock.name } : null;
  }
  const matches = await searchStockByName(query);
  if (matches.length === 0) return null;
  if (matches.length === 1) return matches[0];
  return { candidates: matches };
}

async function executeToolCall(
  toolName: string,
  input: Record<string, unknown>,
  userId: string,
): Promise<string> {
  if (toolName === "show_watchlist") {
    const list = await getWatchlist(userId);
    if (list.length === 0) return "ウォッチリストは空です。銘柄名やコードを言ってもらえれば追加します。";
    const lines = ["📋 ウォッチリスト:"];
    for (const item of list) {
      const stock = await lookupStock(item.code);
      const dpStr = stock ? `下落確率${stock.drop_prob}%` : "";
      const priceStr = stock ? `${stock.close.toLocaleString()}円` : "";
      let status = "";
      if (stock) {
        if (stock.drop_prob < item.dp_threshold) status = "🔔買い時！";
        else if (stock.drop_prob >= (item.dp_sell_threshold ?? 20)) status = "⚠️売り検討";
      }
      lines.push(`  ${item.code} ${item.name} ${priceStr} ${dpStr} ${status}`);
    }
    lines.push(`\n※ 下落確率が買い閾値未満で買い通知、売り閾値以上で売り通知`);
    return lines.join("\n");
  }

  if (toolName === "add_watchlist") {
    const resolved = await resolveCode(input.query as string);
    if (!resolved) return `「${input.query}」に該当する銘柄が見つかりません。`;
    if ("candidates" in resolved) {
      return `「${input.query}」に複数該当:\n${resolved.candidates.map((m) => `  ${m.code} ${m.name}`).join("\n")}\nどの銘柄か教えてください。`;
    }
    const buyTh = (input.buy_threshold as number) ?? 8.0;
    const sellTh = (input.sell_threshold as number) ?? 20.0;
    await addToWatchlist(userId, resolved.code, resolved.name, buyTh, sellTh);
    const stock = await lookupStock(resolved.code);
    return (
      `✅ ${resolved.name}(${resolved.code}) をウォッチリストに追加` +
      (stock ? `\n株価: ${stock.close.toLocaleString()}円\n下落確率: ${stock.drop_prob}%` : "") +
      `\n\n※ 下落確率 < ${buyTh}%で買い通知 / ≥ ${sellTh}%で売り通知`
    );
  }

  if (toolName === "remove_watchlist") {
    const resolved = await resolveCode(input.query as string);
    if (!resolved) return `「${input.query}」に該当する銘柄が見つかりません。`;
    if ("candidates" in resolved) {
      return `「${input.query}」に複数該当:\n${resolved.candidates.map((m) => `  ${m.code} ${m.name}`).join("\n")}\nどの銘柄か教えてください。`;
    }
    await removeFromWatchlist(userId, resolved.code);
    return `🗑 ${resolved.name}(${resolved.code}) をウォッチリストから削除しました。`;
  }

  if (toolName === "lookup_stock") {
    const resolved = await resolveCode(input.query as string);
    if (!resolved) return `「${input.query}」に該当する銘柄が見つかりません。`;
    if ("candidates" in resolved) {
      return `「${input.query}」に複数該当:\n${resolved.candidates.map((m) => `  ${m.code} ${m.name}`).join("\n")}\nどの銘柄か教えてください。`;
    }
    const detail = await fetchStockDetail(resolved.code);
    return detail || `${resolved.code} のデータが見つかりません。`;
  }

  return "不明な操作です。";
}

async function askClaude(userMessage: string, context: string, userId: string): Promise<string> {
  const client = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

  const systemPrompt = `あなたは日本株投資の専門アシスタントBot「stock-alert」です。
ユーザーの株に関する相談に、実際のデータに基づいて日本語で答えてください。

以下はリアルタイムの市場データです:
${context || "（本日のデータはまだありません）"}

あなたの分析フレームワーク:
- XGBoostで63日先の±15%変動を予測。下落モデル(AUC 0.771)の精度が高い
- 下落確率(dp): dp<8は安全圏、dp≥15は危険水準
- 上昇モデルは精度が低いため、netスコアは回答に一切含めないこと。「net」「上昇期待」「上昇バイアス」等の表現も禁止。下落確率のみで判断する
- 数値の推移を示すときは必ず何の数値かを明記すること。「直近5日推移: 2.5→3.2」ではなく「下落確率の直近5日推移: 2.5%→3.2%」のように指標名と単位を必ず付ける
- 全銘柄の平均dp≥15なら日経ETFをキャッシュに退避すべき（マーケットタイミング）

ファンダメンタル指標の見方:
- PER<15: 割安圏、PBR<1: 純資産割れ（割安候補）
- ROE>10%: 資本効率良好、ROE>15%: 優秀
- 配当利回り>3%: 高配当、>5%: 超高配当（持続性要確認）
- 配当性向>80%: 配当維持リスクあり
- Piotroski≥7: 財務健全（9点満点）
- 自己資本比率>40%: 安定、<20%: 財務リスク
- 営業利益率>10%: 高収益体質
- 営業CF黒字: キャッシュ創出力あり
- 通期予想進捗率: 1Q>25%, 2Q>50%, 3Q>75%で順調

銘柄の表記揺れ対応:
- DBの銘柄名は全角英数字（例: ＳＢＩホールディングス、ＫＤＤＩ）
- ユーザーは半角・略称・通称で入力する（SBI, SBIホールディングス, KDDI, トヨタ, 三菱UFJ等）
- toolのqueryには銘柄コード4桁を使うのが最も確実。わかる銘柄はコードで指定せよ
- 主要銘柄コード: トヨタ=7203, ソニー=6758, 任天堂=7974, キーエンス=6861, 三菱UFJ=8306, 三井住友FG=8316, みずほ=8411, SBI HD=8473, ソフトバンクG=9984, NTT=9432, KDDI=9433, ファーストリテイリング=9983, 東京エレクトロン=8035, 信越化学=4063
- 上記にない銘柄は名前の一部（漢字部分やカタカナ部分）をqueryに渡す

追加情報ソース:
- 適時開示（TDnet）: 業績修正・増配・自社株買い・M&A等のカタリスト情報
- 空売り残高（JPX）: 0.5%以上の大口空売りポジション
- 信用残高（JPX）: 信用買残・売残の推移と増減
これらはlookup_stockツールで銘柄を調べた際に自動取得される。

株の相談の答え方:
- ユーザーの質問意図に合わせて関連する指標を選んで回答する
  - 「買いか？」→ 下落確率 + PER/PBR + ROE + 配当利回り + 市場環境
  - 「配当は？」→ 配当利回り + 配当性向 + 純利益成長率 + 営業CF
  - 「安全？」→ 下落確率 + 自己資本比率 + Piotroski + 営業CF
  - 「割安？」→ PER + PBR + ROE + BPS成長 + EPSサプライズ
  - 「成長性は？」→ 売上/純利益成長率 + 営業利益率 + 通期進捗率
  - 「ニュースは？」→ 適時開示（TDnet）のカタリスト情報
  - 「需給は？」→ 空売り残高 + 信用残高 + 出来高
- 全指標を列挙せず、質問に最も関連する3-5指標を選んで分析する
- ユーザーが銘柄の監視や通知を求めたら、toolを使ってウォッチリストを操作する
- 銘柄の詳細を聞かれたら lookup_stock ツールで最新データを取得してから回答する

このBotの機能（ユーザーに「何ができるの？」「機能は？」と聞かれたら、以下を正確に全部案内すること。省略しない）:
1. 📊 毎日の日経225シグナル通知: 毎日20時に全銘柄の平均下落確率からN225を持ち続けてよいか（投資継続OK/キャッシュ推奨）を自動でLINE push通知する
2. 📋 ウォッチリスト銘柄の定期通知: 同じ日次通知で、ウォッチリストに登録した全銘柄の株価・下落確率・買い時/売り時も一緒に届く
3. 💬 株の相談: LINEで話しかければ、実際のデータに基づいた銘柄分析・相場相談ができる（直近の会話を記憶しているので、文脈に沿った会話が可能）
4. 📝 ウォッチリスト管理: 「SBIをウォッチして」「三菱UFJを外して」「リスト」など自然な日本語でウォッチリストを操作できる
5. 📰 適時開示・需給情報: 銘柄を調べると、TDnet適時開示（業績修正/増配/自社株買い等）、JPX空売り残高、信用残高も表示

ルール:
- 投資は自己責任である旨を必要に応じて添える
- 800文字以内で回答する
- 「下落確率」と表記する（dpとは言わない）
- データがない質問には正直に「わからない」と答える`;

  const history = await getRecentMessages(userId);
  const messages: Anthropic.MessageParam[] = [];

  let lastRole = "";
  for (const h of history) {
    if (h.role === lastRole) continue;
    messages.push({ role: h.role as "user" | "assistant", content: h.content });
    lastRole = h.role;
  }
  if (messages.length > 0 && messages[messages.length - 1].role === "user") {
    messages[messages.length - 1] = { role: "user", content: userMessage };
  } else {
    messages.push({ role: "user", content: userMessage });
  }

  const MAX_TURNS = 5;
  let lastText = "";

  for (let turn = 0; turn < MAX_TURNS; turn++) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 1024,
      system: systemPrompt,
      tools: TOOLS,
      messages,
    });

    const textBlocks = response.content.filter(
      (b): b is Anthropic.TextBlock => b.type === "text",
    );
    const toolBlocks = response.content.filter(
      (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
    );

    if (textBlocks.length > 0) {
      lastText = textBlocks.map((b) => b.text).join("\n");
    }

    if (toolBlocks.length === 0) {
      return lastText || "回答を生成できませんでした。";
    }

    const toolResults: Anthropic.ToolResultBlockParam[] = [];
    for (const tool of toolBlocks) {
      const result = await executeToolCall(tool.name, tool.input as Record<string, unknown>, userId);
      toolResults.push({ type: "tool_result", tool_use_id: tool.id, content: result });
    }

    messages.push({ role: "assistant", content: response.content });
    messages.push({ role: "user", content: toolResults });
  }

  if (lastText) return lastText;

  const finalResponse = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    system: systemPrompt,
    messages: [
      ...messages,
      { role: "user", content: "ここまでのツール結果をもとに、ユーザーの質問に日本語で簡潔に答えてください。" },
    ],
  });
  const finalText = finalResponse.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("\n");

  return finalText || "うまく回答できませんでした。もう一度お試しください。";
}

// ── Webhook エントリポイント ─────────────────

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { status: 200 });
  }

  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const rawBody = await req.text();
  const signature = req.headers.get("x-line-signature") ?? "";

  if (LINE_CHANNEL_SECRET && !(await verifySignature(rawBody, signature))) {
    return Response.json({ error: "Invalid signature" }, { status: 403 });
  }

  const body = JSON.parse(rawBody);
  const events = body.events ?? [];

  for (const event of events) {
    if (event.type !== "message" || event.message?.type !== "text") continue;

    const userMessage = event.message.text as string;
    const replyToken = event.replyToken as string;
    const userId = event.source?.userId as string;

    if (!userId) continue;

    try {
      await saveMessage(userId, "user", userMessage);

      const cmd = await handleCommand(userMessage, userId);
      if (cmd.handled) {
        await saveMessage(userId, "assistant", cmd.reply);
        await replyToLine(replyToken, cmd.reply);
        continue;
      }

      const context = await fetchMarketContext(userId, userMessage);
      const reply = await askClaude(userMessage, context, userId);
      await saveMessage(userId, "assistant", reply);
      await replyToLine(replyToken, reply);
    } catch (e) {
      console.error("[LINE webhook] Error:", e);
      await replyToLine(replyToken, "エラーが発生しました。しばらくしてからお試しください。");
    }
  }

  return Response.json({ ok: true });
});
