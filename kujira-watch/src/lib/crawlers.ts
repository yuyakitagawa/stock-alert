// 主要な検索エンジン・SNS・AIクローラーのUser-Agent判定パターン。
// 新しいクローラーに気づいたら随時追加する。
const BOT_PATTERNS: [RegExp, string][] = [
  [/Googlebot/i, "Googlebot"],
  [/bingbot/i, "Bingbot"],
  [/Applebot/i, "Applebot"],
  [/DuckDuckBot/i, "DuckDuckBot"],
  [/Baiduspider/i, "Baiduspider"],
  [/YandexBot/i, "YandexBot"],
  [/GPTBot/i, "GPTBot"],
  [/ChatGPT-User/i, "ChatGPT-User"],
  [/OAI-SearchBot/i, "OAI-SearchBot"],
  [/ClaudeBot/i, "ClaudeBot"],
  [/Claude-Web/i, "Claude-Web"],
  [/anthropic-ai/i, "anthropic-ai"],
  [/PerplexityBot/i, "PerplexityBot"],
  [/facebookexternalhit/i, "facebookexternalhit"],
  [/Twitterbot/i, "Twitterbot"],
  [/LinkedInBot/i, "LinkedInBot"],
  [/Slackbot/i, "Slackbot"],
  [/Discordbot/i, "Discordbot"],
  [/AhrefsBot/i, "AhrefsBot"],
  [/SemrushBot/i, "SemrushBot"],
  [/MJ12bot/i, "MJ12bot"],
  [/GoogleOther/i, "GoogleOther"],
];

export function detectBot(userAgent: string): string | null {
  for (const [pattern, name] of BOT_PATTERNS) {
    if (pattern.test(userAgent)) return name;
  }
  return null;
}

// 主要ブラウザのUAパターン。クローラーでもこれらでもないUA（curl/スクリプト等の
// ノイズ）はclassifyVisitorの対象外（ログしない）とする。
const BROWSER_PATTERNS: RegExp[] = [/Chrome\//, /Safari\//, /Firefox\//, /Edg\//, /OPR\//];

// bot_nameに記録する値を決める。既知クローラーはその名前、主要ブラウザは"Browser"、
// どちらでもなければnull（記録対象外）。
export function classifyVisitor(userAgent: string): string | null {
  const bot = detectBot(userAgent);
  if (bot) return bot;
  if (BROWSER_PATTERNS.some((pattern) => pattern.test(userAgent))) return "Browser";
  return null;
}
