import React from "react";
import {
  AbsoluteFill,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Markdown } from "./markdown";

// ---- Types ----
type Tool = {
  name: "search" | "visit" | string;
  args: Record<string, any>;
  args_str: string;
  queries?: string[];
  urls?: string[];
  goal?: string;
  result?: string;
};
type Turn = { think: string; tool: Tool | null };
export type CaseData = {
  benchmark: string;
  id: number;
  question: string;
  reference_answer: string;
  final_answer: string;
  judge_correct: boolean;
  stats: Record<string, any>;
  turns: Turn[];
};

// ---- Tokens ----
const BG = "#fafaf9";
const PANEL = "#ffffff";
const INK = "#0a0a0a";
const TEXT = "#3f3f46";
const MUTED = "#737373";
const FAINT = "#e7e5e4";
const ACCENT = "#f59e0b";
const ACCENT_INK = "#854d0e";
const OK = "#16a34a";
const FONT_SANS =
  '"Inter", "Helvetica Neue", "PingFang SC", Helvetica, Arial, sans-serif';
const FONT_MONO =
  '"JetBrains Mono", "SF Mono", Menlo, Consolas, monospace';

// ---- Layout constants ----
const QUESTION_HEIGHT = 92;
const VIEW_TOP = 64;
const VIEW_BOTTOM = 38;
const VIEW_H = 720 - VIEW_TOP - VIEW_BOTTOM;
const ANCHOR_Y = 32;
const PADDING_TOP = 18;

// Height estimation constants — calibrated against actual chrome render at 1168px content width
const CHARS_PER_LINE_THINK = 110;  // italic, 13.5px
const CHARS_PER_LINE_RESULT = 130; // sans, 12.5px
const LINE_H_THINK = 22;
const LINE_H_RESULT = 20;
const TOOL_LINE_H = 26;

// ---- Per-turn timing (content-aware) ----
const computeTurnFrames = (t: Turn): number => {
  const thinkChars = Math.min(t.think.length, 8 * 110);
  const toolChars = t.tool
    ? (t.tool.name === "search"
        ? (t.tool.queries || []).join("").length
        : t.tool.name === "visit"
          ? (t.tool.urls || []).join("").length + (t.tool.goal?.length || 0)
          : 0)
    : 0;
  const thinkF = Math.ceil(thinkChars / 80) + 4;
  const toolF = 4 + Math.ceil(toolChars / 36) + 3;
  const resultF = t.tool?.result ? 6 + 20 : 0;
  const pauseF = 4;
  return thinkF + toolF + resultF + pauseF;
};

const HEAD_FRAMES = 60;
const TAIL_FRAMES = 360;

// ---- Height estimation ----
const estimateRowHeight = (t: Turn): number => {
  const PADDING = 14 + 18; // top + bottom padding inside row
  const LABEL_H = 22;
  const GAP = 8;

  // Think is hard-capped to 8 lines max via CSS maxHeight
  const thinkH = 8 * LINE_H_THINK + 8;

  let toolH = 0;
  if (t.tool) {
    const items =
      t.tool.name === "search"
        ? (t.tool.queries || []).length
        : t.tool.name === "visit"
          ? (t.tool.urls || []).length + (t.tool.goal ? 1 : 0)
          : 1;
    toolH = 28 + items * TOOL_LINE_H + 16;
  }

  let resultH = 0;
  if (t.tool?.result) {
    // Result hard-capped to 8 lines via CSS maxHeight
    resultH = 8 * LINE_H_RESULT + 32;
  }

  return PADDING + LABEL_H + thinkH + GAP + toolH + (resultH ? GAP + resultH : 0) + 24;
};

// ---- Streaming primitive ----
const useTyped = (text: string, startFrame: number, charsPerFrame: number) => {
  const frame = useCurrentFrame();
  const local = Math.max(0, frame - startFrame);
  const target = Math.min(text.length, Math.round(local * charsPerFrame));
  return {
    text: text.slice(0, target),
    done: target >= text.length,
    started: frame >= startFrame,
  };
};

const Caret: React.FC<{ color?: string; size?: number }> = ({
  color = INK,
  size = 13,
}) => {
  const frame = useCurrentFrame();
  const blink = Math.floor(frame / 8) % 2 === 0 ? 1 : 0;
  return (
    <span
      style={{
        display: "inline-block",
        width: 2,
        height: size,
        background: color,
        verticalAlign: -2,
        marginLeft: 3,
        opacity: blink,
      }}
    />
  );
};

// ---- Chrome ----
const Chrome: React.FC<{
  benchmark: string;
  id: number;
  turnIdx: number;
  turnTotal: number;
}> = ({ benchmark, id, turnIdx, turnTotal }) => (
  <>
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        padding: "14px 28px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        fontFamily: FONT_SANS,
        borderBottom: `1px solid ${FAINT}`,
        background: "rgba(255,255,255,0.94)",
        zIndex: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <svg width="20" height="20" viewBox="0 0 32 32">
          <path
            d="M22 4a8 8 0 0 1 0 16h-6a2 2 0 0 1-2-2v-6a8 8 0 0 1 8-8Z"
            fill="#0f172a"
          />
          <circle cx="9" cy="23" r="5" fill={ACCENT} />
        </svg>
        <div style={{ fontSize: 14, fontWeight: 600, color: INK, letterSpacing: -0.2 }}>
          LiteResearcher-4B
        </div>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          fontSize: 12,
          color: MUTED,
          letterSpacing: 0.4,
        }}
      >
        <span
          style={{
            width: 6, height: 6, borderRadius: 999,
            background: ACCENT, display: "inline-block",
          }}
        />
        <span style={{ fontFamily: FONT_MONO }}>
          turn {Math.min(turnIdx + 1, turnTotal)}/{turnTotal}
        </span>
      </div>
    </div>
    <div
      style={{
        position: "absolute", left: 0, right: 0, bottom: 0,
        height: 4, background: ACCENT, zIndex: 10,
      }}
    />
  </>
);

// ---- Question ----
const QuestionBlock: React.FC<{ question: string }> = ({ question }) => (
  <div
    style={{
      background: PANEL,
      border: `1px solid ${FAINT}`,
      borderRadius: 12,
      padding: "12px 16px",
      marginBottom: 18,
      fontFamily: FONT_SANS,
    }}
  >
    <div
      style={{
        fontSize: 10.5, color: MUTED,
        textTransform: "uppercase", letterSpacing: 1.2,
        marginBottom: 6, fontWeight: 700,
      }}
    >
      Question
    </div>
    <div
      style={{
        fontSize: 14.5, lineHeight: 1.5,
        color: INK, fontWeight: 500,
        display: "-webkit-box",
        WebkitLineClamp: 2,
        WebkitBoxOrient: "vertical" as any,
        overflow: "hidden",
      }}
    >
      {question}
    </div>
  </div>
);

// ---- Tool call (full list) ----
const ToolCallBlock: React.FC<{ tool: Tool; startFrame: number }> = ({
  tool,
  startFrame,
}) => {
  const frame = useCurrentFrame();
  // For visit: list URL then goal as same-weight items
  const items: { kind: "search" | "visit-url" | "visit-goal"; text: string }[] =
    tool.name === "search"
      ? (tool.queries || []).map((q) => ({ kind: "search" as const, text: q }))
      : tool.name === "visit"
        ? [
            ...(tool.urls || []).map((u) => ({ kind: "visit-url" as const, text: u })),
            ...(tool.goal ? [{ kind: "visit-goal" as const, text: tool.goal }] : []),
          ]
        : [];
  const fullText = items.map((it) => it.text).join("\n");
  const { text: shown, done } = useTyped(fullText, startFrame, 36);
  if (frame < startFrame) return null;

  let remaining = shown.length;
  const visible = items.map((it) => {
    if (remaining <= 0) return { text: "", done: false };
    if (remaining >= it.text.length) {
      remaining -= it.text.length + 1;
      return { text: it.text, done: true };
    }
    const v = { text: it.text.slice(0, remaining), done: false };
    remaining = 0;
    return v;
  });
  const firstStreaming = visible.findIndex((v) => !v.done);

  // Search keeps dark theme; Visit uses light theme
  const isSearch = tool.name === "search";
  const bg = isSearch ? INK : "#f5f5f4";
  const fg = isSearch ? "#fafafa" : INK;
  const itemBg = isSearch ? "rgba(255,255,255,0.05)" : "#ffffff";
  const itemBorder = isSearch ? "rgba(255,255,255,0.08)" : FAINT;
  const iconColor = isSearch ? "#a3a3a3" : MUTED;
  const caretColor = isSearch ? ACCENT : ACCENT_INK;

  return (
    <div
      style={{
        background: bg,
        color: fg,
        borderRadius: 10,
        padding: "10px 14px",
        fontFamily: FONT_MONO,
        fontSize: 12,
        lineHeight: 1.5,
        marginTop: 8,
        border: isSearch ? `1px solid ${INK}` : `1px solid ${FAINT}`,
      }}
    >
      <div
        style={{
          color: ACCENT_INK,
          fontWeight: 700,
          fontSize: 10.5,
          letterSpacing: 1.2,
          textTransform: "uppercase",
          marginBottom: 8,
          fontFamily: FONT_SANS,
        }}
      >
        {isSearch ? "🔍 search" : "🌐 visit"}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {visible.map((vi, i) => {
          if (!vi.text && !vi.done) return null;
          const kind = items[i].kind;
          const icon =
            kind === "search" ? "🔍" : kind === "visit-url" ? "🌐" : "🎯";
          return (
            <div
              key={i}
              style={{
                padding: "5px 9px",
                background: itemBg,
                border: `1px solid ${itemBorder}`,
                borderRadius: 5,
                color: fg,
                wordBreak: "break-all",
              }}
            >
              <span style={{ color: iconColor, marginRight: 6 }}>{icon}</span>
              {vi.text}
              {!vi.done && i === firstStreaming && (
                <Caret color={caretColor} size={11} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ---- Result block (full markdown) ----
const ResultBlock: React.FC<{ result: string; startFrame: number }> = ({
  result,
  startFrame,
}) => {
  const frame = useCurrentFrame();
  if (frame < startFrame || !result) return null;
  // Tool response arrives instantly (it's data from outside, not the model typing).
  // Small fade-in so it doesn't pop.
  const fadeFrames = 6;
  const fade = Math.min(1, Math.max(0, (frame - startFrame) / fadeFrames));
  return (
    <div
      style={{
        background: "#f5f5f4",
        borderRadius: 10,
        padding: "10px 14px",
        fontFamily: FONT_SANS,
        fontSize: 12.5,
        lineHeight: 1.55,
        color: TEXT,
        marginTop: 8,
        border: `1px solid ${FAINT}`,
        opacity: fade,
        transform: `translateY(${(1 - fade) * 4}px)`,
      }}
    >
      <div
        style={{
          color: MUTED,
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: 1.2,
          textTransform: "uppercase",
          marginBottom: 6,
        }}
      >
        ↳ tool_response
      </div>
      <div style={{ maxHeight: 8 * 20, overflow: "hidden" }}>
        <Markdown text={result} />
      </div>
    </div>
  );
};

// ---- Think block (streaming text) ----
const ThinkBlock: React.FC<{ text: string; startFrame: number }> = ({
  text,
  startFrame,
}) => {
  const frame = useCurrentFrame();
  const { text: shown, done } = useTyped(text || "", startFrame, 80);
  if (frame < startFrame) return null;
  return (
    <div
      style={{
        fontFamily: FONT_SANS,
        fontSize: 13.5,
        lineHeight: 1.6,
        color: TEXT,
        fontStyle: "italic",
        marginBottom: 8,
        whiteSpace: "pre-wrap",
        maxHeight: 8 * 22,
        overflow: "hidden",
      }}
    >
      {shown}
      {!done && <Caret color={MUTED} size={12} />}
    </div>
  );
};

// ---- Turn row (content-driven height) ----
const TurnRow: React.FC<{
  turn: Turn;
  index: number;
  startFrameAbs: number;
  isActive: boolean;
  rowHeight: number;
}> = ({ turn, index, startFrameAbs, isActive, rowHeight }) => {
  const frame = useCurrentFrame();
  const thinkChars = Math.min(turn.think.length, 8 * 110); // cap to visible
  const thinkTypingF = Math.ceil(thinkChars / 80);
  const thinkAt = startFrameAbs;
  const toolAt = startFrameAbs + thinkTypingF + 4;
  const toolChars = turn.tool
    ? (turn.tool.name === "search"
        ? (turn.tool.queries || []).join("").length
        : turn.tool.name === "visit"
          ? (turn.tool.urls || []).join("").length + (turn.tool.goal?.length || 0)
          : 0)
    : 0;
  const resultAt = toolAt + 4 + Math.ceil(toolChars / 36) + 3;

  const enterT =
    frame < thinkAt ? 0 : Math.min(1, (frame - thinkAt) / 12);

  return (
    <div
      style={{
        padding: "14px 0 18px",
        position: "relative",
        opacity: enterT * (isActive ? 1 : 0.55),
        transform: `translateY(${(1 - enterT) * 10}px)`,
        marginBottom: 12,
        borderBottom: `1px dashed ${FAINT}`,
        height: rowHeight,
        overflow: "hidden",
        boxSizing: "border-box",
      }}
    >
      {isActive && (
        <div
          style={{
            position: "absolute",
            left: -18,
            top: 18,
            bottom: 22,
            width: 3,
            borderRadius: 2,
            background: ACCENT,
          }}
        />
      )}
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 10,
          marginBottom: 6,
        }}
      >
        <div
          style={{
            fontFamily: FONT_SANS,
            fontSize: 10.5,
            fontWeight: 700,
            letterSpacing: 1.2,
            textTransform: "uppercase",
            color: isActive ? ACCENT_INK : MUTED,
          }}
        >
          ● Turn {index + 1}
        </div>
        <div
          style={{
            fontFamily: FONT_SANS,
            fontSize: 10.5,
            fontWeight: 700,
            letterSpacing: 1.2,
            textTransform: "uppercase",
            color: MUTED,
          }}
        >
          💭 thinking
        </div>
      </div>
      <ThinkBlock text={turn.think} startFrame={thinkAt} />
      {turn.tool && (
        <ToolCallBlock tool={turn.tool} startFrame={toolAt} />
      )}
      {turn.tool?.result && (
        <ResultBlock result={turn.tool.result} startFrame={resultAt} />
      )}
    </div>
  );
};

// ---- Final answer ----
const FinalAnswerBlock: React.FC<{
  final: string;
  reference: string;
  correct: boolean;
  startFrame: number;
}> = ({ final, reference, correct, startFrame }) => {
  const frame = useCurrentFrame();
  const { text: shown, done } = useTyped(final, startFrame + 8, 22);
  if (frame < startFrame) return null;
  const local = frame - startFrame;
  const opacity = Math.min(1, local / 8);
  const y = (1 - Math.min(1, local / 10)) * 14;
  return (
    <div style={{ opacity, transform: `translateY(${y}px)`, fontFamily: FONT_SANS }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <div
          style={{
            width: 24, height: 24, borderRadius: 999,
            background: correct ? OK : "#dc2626",
            color: "white", fontSize: 14, fontWeight: 800,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          {correct ? "✓" : "✗"}
        </div>
        <div
          style={{
            fontSize: 12, fontWeight: 700, color: INK,
            textTransform: "uppercase", letterSpacing: 1.2,
          }}
        >
          Final Answer
        </div>
        {reference && (
          <div style={{ marginLeft: "auto", fontSize: 12, color: MUTED, fontFamily: FONT_MONO }}>
            ref: <span style={{ color: INK, fontWeight: 600 }}>{reference}</span>
          </div>
        )}
      </div>
      <div
        style={{
          background: PANEL,
          border: `1px solid ${FAINT}`,
          borderLeft: `3px solid ${ACCENT}`,
          borderRadius: 10,
          padding: "16px 20px",
          fontSize: 14,
          color: INK,
          minHeight: 120,
        }}
      >
        <Markdown text={shown} />
        {!done && <Caret color={ACCENT_INK} size={13} />}
      </div>
    </div>
  );
};

// ---- Main composition ----
export const TrajectoryReplay: React.FC<{ caseData: CaseData }> = ({
  caseData,
}) => {
  const { fps, durationInFrames } = useVideoConfig();
  const frame = useCurrentFrame();

  // Per-turn timing (content-aware)
  const turnFrames = caseData.turns.map(computeTurnFrames);
  const turnStartFrames: number[] = [];
  let acc = HEAD_FRAMES;
  for (let i = 0; i < caseData.turns.length; i++) {
    turnStartFrames.push(acc);
    acc += turnFrames[i];
  }
  const finalAt = acc;

  // Active turn
  let activeIdx = -1;
  for (let i = 0; i < turnStartFrames.length; i++) {
    if (frame >= turnStartFrames[i]) activeIdx = i;
  }

  // Per-turn estimated heights
  const rowHeights = caseData.turns.map(estimateRowHeight);

  // Scroll target: top of active turn at ANCHOR_Y
  const scrollTargetForIdx = (i: number) => {
    if (i < 0) return 0;
    let top = PADDING_TOP + QUESTION_HEIGHT;
    for (let k = 0; k < i; k++) top += rowHeights[k] + 12; // +marginBottom
    return Math.max(0, top - ANCHOR_Y);
  };
  const currTarget = scrollTargetForIdx(activeIdx);
  const prevTarget = scrollTargetForIdx(activeIdx - 1);
  // Spring on turn change
  const t = activeIdx >= 0
    ? spring({
        frame: frame - turnStartFrames[activeIdx],
        fps,
        config: { damping: 28, stiffness: 80, mass: 1.0 },
        durationInFrames: 40,
      })
    : 0;
  const scrollY = prevTarget + (currTarget - prevTarget) * t;

  const finalActive = frame >= finalAt;
  const finalFade = Math.min(1, Math.max(0, (frame - finalAt) / 10));

  return (
    <AbsoluteFill
      style={{
        background: BG,
        fontFamily: FONT_SANS,
        color: INK,
      }}
    >
      <Chrome
        benchmark={caseData.benchmark}
        id={caseData.id}
        turnIdx={Math.max(0, activeIdx)}
        turnTotal={caseData.turns.length}
      />
      <div
        style={{
          position: "absolute",
          top: VIEW_TOP,
          bottom: VIEW_BOTTOM,
          left: 0,
          right: 0,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 56,
            right: 56,
            top: PADDING_TOP,
            transform: `translateY(${-scrollY}px)`,
            opacity: 1 - finalFade,
            display: finalFade >= 1 ? "none" : "block",
          }}
        >
          <QuestionBlock question={caseData.question} />
          {caseData.turns.map((t, i) => (
            <TurnRow
              key={i}
              turn={t}
              index={i}
              startFrameAbs={turnStartFrames[i]}
              isActive={i === activeIdx}
              rowHeight={rowHeights[i]}
            />
          ))}
        </div>
        {finalActive && (
          <div
            style={{
              position: "absolute",
              left: 56, right: 56, top: 24,
              opacity: finalFade,
            }}
          >
            <div
              style={{
                borderLeft: `3px solid ${FAINT}`,
                paddingLeft: 14,
                marginBottom: 12,
                fontSize: 12, color: MUTED,
                fontFamily: FONT_SANS, lineHeight: 1.4,
              }}
            >
              <span
                style={{
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: 1.1,
                  marginRight: 8,
                }}
              >
                Q
              </span>
              {caseData.question.length > 130
                ? caseData.question.slice(0, 128) + "…"
                : caseData.question}
            </div>
            <FinalAnswerBlock
              final={caseData.final_answer}
              reference={caseData.reference_answer}
              correct={caseData.judge_correct}
              startFrame={finalAt}
            />
          </div>
        )}
      </div>
    </AbsoluteFill>
  );
};
