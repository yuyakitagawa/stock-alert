/**
 * 旧記事のタイトル生成処理が60文字を超えた提出者名を「…」で省略していたため、
 * microCMSに保存済みの正式な提出者名で復元する。
 *
 * 対象は決定的テンプレート（「銘柄（コード）、提出者が…」）だけに限定し、
 * 自社株買いなど別形式のタイトルは変更しない。
 */
export function fullArticleTitle(title: string, filerName?: string): string {
  if (!filerName || !title.includes("…")) return title;

  const separator = title.indexOf("、");
  const action = title.indexOf("が", separator + 1);
  if (separator < 0 || action < 0) return title;

  const abbreviatedFiler = title.slice(separator + 1, action);
  if (!abbreviatedFiler.endsWith("…")) return title;

  return `${title.slice(0, separator + 1)}${filerName}${title.slice(action)}`;
}
