import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import { X_FOLLOW_URL, X_SCREEN_NAME, YOUTUBE_CHANNEL_URL } from "@/lib/site";

// 記事末尾のXフォロー導線。検索から一度来た読者に、再訪のきっかけとして
// Xタイムラインを持ってもらうのが目的。ShareButtonsと同じ理由で、SDKや
// 埋め込みウィジェットは使わず素のリンク（フォローintent）だけで実装する。
// 本文と同系色の枠線ボックスでは読み飛ばされるため、サイトで他に使っていない
// ネイビー地＋白ボタンの反転配色で意図的に目立たせている。
export default function FollowCta() {
  return (
    <Box
      sx={{
        mt: 2,
        p: { xs: 2.5, sm: 3 },
        borderRadius: 2,
        bgcolor: "primary.main",
        color: "#fff",
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        gap: 1.5,
      }}
    >
      <Typography sx={{ fontSize: { xs: "1.125rem", sm: "1.25rem" }, fontWeight: 800, lineHeight: 1.4 }}>
        🐋 次のクジラの売買を見逃さない
      </Typography>
      <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.85)" }}>
        新着記事や大口投資家の動きは、公式X（@{X_SCREEN_NAME}）で毎日お知らせしています。
        その日いちばん大きい開示は、YouTubeの1分ショート動画でも解説しています。
      </Typography>
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1.5 }}>
        <Button
          component="a"
          href={X_FOLLOW_URL}
          target="_blank"
          rel="noopener noreferrer"
          variant="contained"
          size="medium"
          sx={{
            bgcolor: "#fff",
            color: "primary.main",
            px: 3,
            "&:hover": { bgcolor: "brand.goldBright", color: "primary.main" },
          }}
        >
          Xでフォローする
        </Button>
        <Button
          component="a"
          href={YOUTUBE_CHANNEL_URL}
          target="_blank"
          rel="noopener noreferrer"
          variant="outlined"
          size="medium"
          sx={{
            color: "#fff",
            borderColor: "rgba(255,255,255,0.6)",
            px: 3,
            "&:hover": { borderColor: "#fff", bgcolor: "rgba(255,255,255,0.08)" },
          }}
        >
          YouTubeで1分解説を見る
        </Button>
      </Box>
    </Box>
  );
}
