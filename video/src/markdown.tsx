// Tiny markdown renderer — returns ReactNode for inline + block markdown.
// Supports: **bold**, *italic*, `code`, paragraphs, "- " bullets, numbered "1." lists.
// No HTML escaping needed since text comes from our own JSON; we never render raw HTML.

import React from "react";

const BOLD = /\*\*([^*]+)\*\*/g;
const ITALIC = /(^|[^*])\*([^*\n]+)\*/g;
const CODE = /`([^`]+)`/g;

const renderInline = (s: string): React.ReactNode[] => {
  // Replace **bold** first via sentinels, then *italic*, then `code`.
  type Tok = { type: "text" | "bold" | "italic" | "code"; v: string };
  const tokens: Tok[] = [{ type: "text", v: s }];

  const expand = (regex: RegExp, kind: Tok["type"], group: number) => {
    const out: Tok[] = [];
    for (const t of tokens) {
      if (t.type !== "text") {
        out.push(t);
        continue;
      }
      let last = 0;
      const str = t.v;
      const re = new RegExp(regex.source, "g");
      let m: RegExpExecArray | null;
      while ((m = re.exec(str)) !== null) {
        if (m.index > last) out.push({ type: "text", v: str.slice(last, m.index) });
        if (group === 0) out.push({ type: kind, v: m[1] });
        else out.push({ type: "text", v: m[1] }, { type: kind, v: m[2] });
        last = m.index + m[0].length;
      }
      if (last < str.length) out.push({ type: "text", v: str.slice(last) });
    }
    tokens.length = 0;
    tokens.push(...out);
  };

  expand(BOLD, "bold", 0);
  expand(ITALIC, "italic", 1);
  expand(CODE, "code", 0);

  return tokens.map((t, i) => {
    if (t.type === "bold")
      return (
        <strong key={i} style={{ fontWeight: 700, color: "#000" }}>
          {t.v}
        </strong>
      );
    if (t.type === "italic") return <em key={i}>{t.v}</em>;
    if (t.type === "code")
      return (
        <code
          key={i}
          style={{
            fontFamily:
              '"JetBrains Mono", "SF Mono", Menlo, Consolas, monospace',
            background: "#f1f1ef",
            padding: "1px 6px",
            borderRadius: 4,
            fontSize: "0.92em",
          }}
        >
          {t.v}
        </code>
      );
    return <React.Fragment key={i}>{t.v}</React.Fragment>;
  });
};

export const Markdown: React.FC<{
  text: string;
  style?: React.CSSProperties;
}> = ({ text, style }) => {
  if (!text) return null;
  const lines = text.split(/\n/);
  // Group into blocks
  const blocks: Array<
    | { kind: "p"; lines: string[] }
    | { kind: "ul"; items: string[] }
    | { kind: "ol"; items: string[] }
  > = [];
  let cur: (typeof blocks)[number] | null = null;
  const flush = () => {
    if (cur) {
      blocks.push(cur);
      cur = null;
    }
  };
  for (const raw of lines) {
    const ln = raw.trim();
    if (!ln) {
      flush();
      continue;
    }
    const ulm = /^[-*]\s+(.*)$/.exec(ln);
    const olm = /^\d+[.)]\s+(.*)$/.exec(ln);
    if (ulm) {
      if (!cur || cur.kind !== "ul") {
        flush();
        cur = { kind: "ul", items: [] };
      }
      cur.items.push(ulm[1]);
    } else if (olm) {
      if (!cur || cur.kind !== "ol") {
        flush();
        cur = { kind: "ol", items: [] };
      }
      cur.items.push(olm[1]);
    } else {
      if (!cur || cur.kind !== "p") {
        flush();
        cur = { kind: "p", lines: [] };
      }
      cur.lines.push(ln);
    }
  }
  flush();

  return (
    <div style={style}>
      {blocks.map((b, i) => {
        if (b.kind === "p")
          return (
            <p
              key={i}
              style={{ margin: i === 0 ? "0 0 8px" : "8px 0", lineHeight: 1.55 }}
            >
              {renderInline(b.lines.join(" "))}
            </p>
          );
        if (b.kind === "ul")
          return (
            <ul
              key={i}
              style={{
                margin: "6px 0",
                paddingLeft: 22,
                lineHeight: 1.55,
              }}
            >
              {b.items.map((it, j) => (
                <li key={j} style={{ marginBottom: 4 }}>
                  {renderInline(it)}
                </li>
              ))}
            </ul>
          );
        return (
          <ol
            key={i}
            style={{
              margin: "6px 0",
              paddingLeft: 22,
              lineHeight: 1.55,
            }}
          >
            {b.items.map((it, j) => (
              <li key={j} style={{ marginBottom: 4 }}>
                {renderInline(it)}
              </li>
            ))}
          </ol>
        );
      })}
    </div>
  );
};
