"use client";

import { useState, type ReactNode } from "react";
import Tooltip from "@mui/material/Tooltip";
import ClickAwayListener from "@mui/material/ClickAwayListener";
import IconButton from "@mui/material/IconButton";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

// ℹ️アイコン。ホバー中または押下時だけ補足説明を表示する（スマホはタップでトグル、外側タップで閉じる）。
export default function InfoTip({ content, label = "補足説明" }: { content: ReactNode; label?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <ClickAwayListener onClickAway={() => setOpen(false)}>
      <span className="inline-flex align-middle">
        <Tooltip
          title={<span className="block text-xs leading-relaxed">{content}</span>}
          open={open}
          onOpen={() => setOpen(true)}
          onClose={() => setOpen(false)}
          disableTouchListener
          arrow
          placement="bottom-start"
        >
          <IconButton
            size="small"
            aria-label={label}
            onClick={() => setOpen((v) => !v)}
            sx={{ p: 0.25, ml: 0.5, color: "text.secondary" }}
          >
            <InfoOutlinedIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Tooltip>
      </span>
    </ClickAwayListener>
  );
}
