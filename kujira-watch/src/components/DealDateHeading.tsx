import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export default function DealDateHeading({ label }: { label: string }) {
  return (
    <Typography variant="subtitle1" component="h2" sx={{ mb: 2.5, display: "flex", alignItems: "center", gap: 1.5, fontWeight: 700, color: "primary.main" }}>
      {label}
      <Box component="span" aria-hidden sx={{ height: "1px", flex: 1, bgcolor: "divider" }} />
    </Typography>
  );
}
