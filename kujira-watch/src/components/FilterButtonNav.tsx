import Stack from "@mui/material/Stack";
import ActionButton from "./ActionButton";

// 一覧ページ（ランキング・銘柄・投資家）の絞り込みナビ。以前はテキストリンクの
// 太字/薄字で選択状態を表していたが、押せる要素だと分かりづらかったため
// ボタン化し、選択中のみ塗りつぶし(contained)にする。
// 選択肢が多いページ（投資家分類は13種類）は`collapsible`で<details>に畳み、
// 一覧本体が絞り込みボタンに押し下げられないようにする。`collapsible`を使う場合、
// items[0]は「すべて」（=絞り込みなし）であることを前提にしている。
export default function FilterButtonNav({
  ariaLabel,
  items,
  collapsible = false,
}: {
  ariaLabel: string;
  items: { href: string; label: string; selected: boolean }[];
  collapsible?: boolean;
}) {
  const nav = (
    <Stack
      component="nav"
      aria-label={ariaLabel}
      direction="row"
      className="no-scrollbar"
      sx={{
        gap: 1,
        mb: collapsible ? 0 : 3,
        mt: collapsible ? 1.5 : 0,
        // スマホでは折り返すと絞り込みだけで画面が埋まってしまうため横スクロールにする。
        flexWrap: { xs: "nowrap", sm: "wrap" },
        overflowX: { xs: "auto", sm: "visible" },
        "& .MuiButton-root": { flexShrink: 0 },
      }}
    >
      {items.map((item) => (
        <ActionButton key={item.href} href={item.href} selected={item.selected}>
          {item.label}
        </ActionButton>
      ))}
    </Stack>
  );

  if (!collapsible) return nav;

  // items[0]（すべて）以外を選んでいる＝絞り込み中のときだけ開いた状態で描画する。
  // 何で絞っているかが一目で分かり、他の分類への切り替えも1タップ省ける。
  const selectedIndex = items.findIndex((item) => item.selected);
  const filtering = selectedIndex > 0;

  return (
    <details className="mb-4" open={filtering}>
      <summary className="cursor-pointer text-sm text-brand-blue hover:underline">
        {ariaLabel}
        {filtering && `（現在: ${items[selectedIndex].label}）`}
      </summary>
      {nav}
    </details>
  );
}
