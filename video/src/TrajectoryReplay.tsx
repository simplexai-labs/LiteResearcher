import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Markdown } from "./markdown";

// ---- Types -----------------------------------------------------------------

type Tool = {
  name: "search" | "visit" | string;
  args: Record<string, any>;
  args_str: string;
  result?: string;
};

type Turn = {
  think: string;
  tool: Tool | null;
};

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

// ---- Tokens (Lev8 palette: black/grey/white + amber accent matching logo) --

const BG = "#fafaf9";          // off-white
const PANEL = "#ffffff";
const INK = "#0a0a0a";
const TEXT = "#3f3f46";
const MUTED = "#737373";
const FAINT = "#e7e5e4";
const ACCENT = "#facc15";      // amber, matches Simplex yellow drop
const ACCENT_INK = "#854d0e";
const CODE_BG = "#0a0a0a";
const CODE_INK = "#fafaf9";
const OK = "#16a34a";

const FONT_SANS =
  '"Inter", "Helvetica Neue", "PingFang SC", Helvetica, Arial, sans-serif';
const FONT_MONO =
  '"JetBrains Mono", "SF Mono", Menlo, Consolas, monospace';

// ---- Streaming primitives --------------------------------------------------

/** Type-stream a string char-by-char. Returns visible substring + a flag. */
const useTyped = (text: string, startFrame: number, charsPerFrame = 2) => {
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
  size = 16,
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

// ---- Building blocks --------------------------------------------------------

const Chrome: React.FC<{ benchmark: string; id: number }> = ({
  benchmark,
  id,
}) => {
  return (
    <>
      {/* Top bar */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          padding: "16px 28px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontFamily: FONT_SANS,
          borderBottom: `1px solid ${FAINT}`,
          background: "rgba(255,255,255,0.7)",
          backdropFilter: "blur(8px)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {/* logo glyph: blue-grey drop + yellow dot */}
          <svg width="20" height="20" viewBox="0 0 32 32">
            <path
              d="M22 4a8 8 0 0 1 0 16h-6a2 2 0 0 1-2-2v-6a8 8 0 0 1 8-8Z"
              fill="#0f172a"
            />
            <circle cx="9" cy="23" r="5" fill={ACCENT} />
          </svg>
          <div
            style={{
              fontSize: 14,
              fontWeight: 600,
              color: INK,
              letterSpacing: -0.2,
            }}
          >
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
          <span style={{ textTransform: "uppercase" }}>
            {benchmark} · #{id}
          </span>
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: 999,
              background: ACCENT,
              display: "inline-block",
            }}
          />
          <span>live trajectory</span>
        </div>
      </div>
      {/* Bottom URL */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: 0,
          padding: "10px 28px",
          fontFamily: FONT_MONO,
          fontSize: 11,
          color: MUTED,
          letterSpacing: 0.3,
          borderTop: `1px solid ${FAINT}`,
          background: "rgba(255,255,255,0.7)",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>simplex-ai-inc.github.io/LiteResearcher</span>
        <span>🤗 LiteResearcher-4B</span>
      </div>
    </>
  );
};

const QuestionBlock: React.FC<{ question: string }> = ({ question }) => {
  // Question is shown statically (it's user-supplied input, not streamed)
  return (
    <div
      style={{
        background: PANEL,
        border: `1px solid ${FAINT}`,
        borderRadius: 12,
        padding: "14px 18px",
        fontFamily: FONT_SANS,
      }}
    >
      <div
        style={{
          fontSize: 11,
          color: MUTED,
          textTransform: "uppercase",
          letterSpacing: 1,
          marginBottom: 6,
          fontWeight: 600,
        }}
      >
        Question
      </div>
      <div
        style={{
          fontSize: 17,
          lineHeight: 1.5,
          color: INK,
          fontWeight: 500,
        }}
      >
        {question}
      </div>
    </div>
  );
};

const ThinkingBlock: React.FC<{
  text: string;
  startFrame: number;
}> = ({ text, startFrame }) => {
  const { text: shown, done, started } = useTyped(text, startFrame, 1.8);
  if (!started) return null;
  return (
    <div
      style={{
        fontFamily: FONT_SANS,
        padding: "10px 0 4px",
      }}
    >
      <div
        style={{
          fontSize: 11,
          textTransform: "uppercase",
          letterSpacing: 1.2,
          color: MUTED,
          fontWeight: 700,
          marginBottom: 4,
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        <span>💭</span> Thinking
      </div>
      <div
        style={{
          fontSize: 15,
          lineHeight: 1.55,
          color: TEXT,
          fontStyle: "italic",
        }}
      >
        {shown}
        {!done && <Caret color={MUTED} size={14} />}
      </div>
    </div>
  );
};

const ToolCallBlock: React.FC<{
  tool: Tool;
  startFrame: number;
}> = ({ tool, startFrame }) => {
  const frame = useCurrentFrame();
  const args = tool.args || {};
  const argsLine = tool.args_str || "";
  const { text: argsShown, done } = useTyped(argsLine, startFrame, 4);
  if (frame < startFrame) return null;

  const goal = (args.goal as string) || "";

  return (
    <div
      style={{
        background: CODE_BG,
        color: CODE_INK,
        borderRadius: 10,
        padding: "12px 14px",
        fontFamily: FONT_MONO,
        fontSize: 13,
        lineHeight: 1.55,
        boxShadow: "0 4px 14px rgba(10,10,10,0.18)",
        border: `1px solid ${INK}`,
        marginTop: 6,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          color: ACCENT,
          fontWeight: 700,
          fontSize: 11,
          textTransform: "uppercase",
          letterSpacing: 1.1,
          marginBottom: 6,
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 999,
            background: ACCENT,
            display: "inline-block",
          }}
        />
        tool_call · {tool.name}
      </div>
      <div>
        <span style={{ color: "#9ca3af" }}>{tool.name}(</span>
        <span style={{ color: CODE_INK }}>{argsShown}</span>
        {!done && <Caret color={ACCENT} size={12} />}
        {done && <span style={{ color: "#9ca3af" }}>)</span>}
      </div>
      {goal && done && (
        <div style={{ marginTop: 6, color: "#a3a3a3", fontSize: 12 }}>
          // {goal}
        </div>
      )}
    </div>
  );
};

const ToolResultBlock: React.FC<{
  result: string;
  startFrame: number;
}> = ({ result, startFrame }) => {
  const frame = useCurrentFrame();
  const { text: shown, done } = useTyped(result || "", startFrame, 5);
  if (frame < startFrame || !result) return null;

  return (
    <div
      style={{
        background: "#f5f5f4",
        borderRadius: 10,
        padding: "10px 14px",
        fontFamily: FONT_MONO,
        fontSize: 12,
        lineHeight: 1.55,
        color: TEXT,
        marginTop: 6,
        border: `1px solid ${FAINT}`,
      }}
    >
      <div
        style={{
          color: MUTED,
          fontSize: 10,
          textTransform: "uppercase",
          letterSpacing: 1.1,
          fontWeight: 700,
          marginBottom: 4,
        }}
      >
        ↳ tool_response
      </div>
      <div>
        {shown}
        {!done && <Caret color={MUTED} size={12} />}
      </div>
    </div>
  );
};

const FinalAnswerBlock: React.FC<{
  final: string;
  reference: string;
  correct: boolean;
  startFrame: number;
}> = ({ final, reference, correct, startFrame }) => {
  const frame = useCurrentFrame();
  const { text: shown, done } = useTyped(final, startFrame + 8, 18);
  if (frame < startFrame) return null;
  const local = frame - startFrame;
  const headerOpacity = Math.min(1, local / 6);
  const headerY = (1 - Math.min(1, local / 6)) * 8;

  return (
    <div
      style={{
        opacity: headerOpacity,
        transform: `translateY(${headerY}px)`,
        fontFamily: FONT_SANS,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 10,
        }}
      >
        <div
          style={{
            width: 22,
            height: 22,
            borderRadius: 999,
            background: correct ? OK : "#dc2626",
            color: "white",
            fontSize: 13,
            fontWeight: 800,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {correct ? "✓" : "✗"}
        </div>
        <div
          style={{
            fontSize: 13,
            fontWeight: 700,
            color: INK,
            textTransform: "uppercase",
            letterSpacing: 1.2,
          }}
        >
          Final Answer
        </div>
        {reference && (
          <div
            style={{
              marginLeft: "auto",
              fontSize: 12,
              color: MUTED,
              fontFamily: FONT_MONO,
            }}
          >
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
          fontSize: 14.5,
          color: INK,
          minHeight: 120,
        }}
      >
        <Markdown text={shown} />
        {!done && <Caret color={ACCENT_INK} size={14} />}
      </div>
    </div>
  );
};

// ---- Main composition ------------------------------------------------------

export const TrajectoryReplay: React.FC<{ caseData: CaseData }> = ({
  caseData,
}) => {
  const { fps, durationInFrames } = useVideoConfig();
  const frame = useCurrentFrame();

  // Build a timeline. We always render every prior block (no scroll jump);
  // overflow is clipped by a translateY auto-scroll once content exceeds viewport.

  // Time budgets (frames)
  const Q_AT = 6;                          // question fades in
  const PER_TURN_BASE = Math.round(fps * 4.0); // 4s per turn baseline
  const FINAL_GAP = Math.round(fps * 0.6);
  const FINAL_DURATION = Math.round(fps * 12);  // tail for streaming + linger
  // distribute turns to fill (durationInFrames - FINAL_DURATION - Q_AT - small pad)
  const turnArea = durationInFrames - FINAL_DURATION - Q_AT - 12;
  const perTurn = Math.max(
    PER_TURN_BASE,
    Math.floor(turnArea / Math.max(1, caseData.turns.length))
  );

  // Compute per-turn start frames + sub-phase offsets
  const turnTimings = caseData.turns.map((t, i) => {
    const start = Q_AT + 4 + i * perTurn;
    const thinkAt = start;
    const thinkTypingFrames = Math.ceil(t.think.length / 1.8);
    const toolAt = thinkAt + thinkTypingFrames + 8;
    const toolArgsFrames = Math.ceil((t.tool?.args_str?.length || 0) / 4);
    const resultAt = toolAt + toolArgsFrames + 10;
    return { thinkAt, toolAt, resultAt };
  });

  const finalAt =
    Q_AT +
    4 +
    caseData.turns.length * perTurn +
    FINAL_GAP;

  const finalActive = frame >= finalAt;
  const finalFade = Math.min(1, Math.max(0, (frame - finalAt) / 12));

  return (
    <AbsoluteFill
      style={{
        background: BG,
        fontFamily: FONT_SANS,
        color: INK,
      }}
    >
      <Chrome benchmark={caseData.benchmark} id={caseData.id} />
      {/* Conversation viewport */}
      <div
        style={{
          position: "absolute",
          top: 64,
          bottom: 38,
          left: 0,
          right: 0,
          overflow: "hidden",
          padding: "20px 64px",
        }}
      >
        {/* Phase 1 — streaming chat (turns) */}
        <div
          style={{
            opacity: 1 - finalFade,
            transform: `translateY(${finalFade * -40}px)`,
            display: finalFade >= 1 ? "none" : "flex",
            flexDirection: "column",
            gap: 10,
            height: "100%",
          }}
        >
          {/* Sliding window: show the last 2 turns to keep content in view */}
          {(() => {
            // Determine how many turns are currently active (think has started)
            const activeIdxList = turnTimings
              .map((tm, i) => ({ i, tm }))
              .filter((x) => frame >= x.tm.thinkAt)
              .map((x) => x.i);
            const lastActive = activeIdxList[activeIdxList.length - 1] ?? -1;
            // window: show [lastActive-1, lastActive]
            const winStart = Math.max(0, lastActive - 1);
            // Always show question on first turn(s), then hide it once content fills
            const showQuestion = lastActive < 1;

            return (
              <>
                {showQuestion && (
                  <div
                    style={{
                      opacity: Math.min(1, Math.max(0, (frame - Q_AT) / 8)),
                    }}
                  >
                    <QuestionBlock question={caseData.question} />
                  </div>
                )}
                {!showQuestion && (
                  <div
                    style={{
                      background: "transparent",
                      borderLeft: `3px solid ${FAINT}`,
                      paddingLeft: 14,
                      marginBottom: 4,
                      fontSize: 12,
                      color: MUTED,
                      fontFamily: FONT_SANS,
                      lineHeight: 1.4,
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
                    {caseData.question.length > 110
                      ? caseData.question.slice(0, 108) + "…"
                      : caseData.question}
                  </div>
                )}
                {caseData.turns.map((t, i) => {
                  if (i < winStart) return null;
                  const tm = turnTimings[i];
                  return (
                    <div
                      key={i}
                      style={{ display: "flex", flexDirection: "column" }}
                    >
                      <ThinkingBlock text={t.think} startFrame={tm.thinkAt} />
                      {t.tool && (
                        <ToolCallBlock tool={t.tool} startFrame={tm.toolAt} />
                      )}
                      {t.tool?.result && (
                        <ToolResultBlock
                          result={t.tool.result}
                          startFrame={tm.resultAt}
                        />
                      )}
                    </div>
                  );
                })}
              </>
            );
          })()}
        </div>

        {/* Phase 2 — final answer full-screen */}
        {finalActive && (
          <div
            style={{
              position: "absolute",
              inset: "20px 64px",
              opacity: finalFade,
              transform: `translateY(${(1 - finalFade) * 20}px)`,
              display: "flex",
              flexDirection: "column",
              gap: 10,
            }}
          >
            <div
              style={{
                background: "transparent",
                borderLeft: `3px solid ${FAINT}`,
                paddingLeft: 14,
                marginBottom: 4,
                fontSize: 12,
                color: MUTED,
                fontFamily: FONT_SANS,
                lineHeight: 1.4,
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
              {caseData.question.length > 110
                ? caseData.question.slice(0, 108) + "…"
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
