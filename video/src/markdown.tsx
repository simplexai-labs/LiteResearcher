// Tiny markdown renderer — returns ReactNode for inline + block markdown.
// Supports: ## headings, **bold**, *italic*, `code`, [text](url), paragraphs, "- " bullets, numbered "1." lists.

import React from "react";

const BOLD = /\*\*([^*]+)\*\*/g;
const ITALIC = /(^|[^*])\*([^*\n]+)\*/g;
const CODE = /`([^`]+)`/g;
const LINK = /\[([^\]]+)\]\(([^)]+)\)/g;

const renderInline = (s: string): React.ReactNode[] => {
  type Tok =
    | { type: "text"; v: string }
    | { type: "bold" | "italic" | "code"; v: string }
    | { type: "link"; v: string; href: string };
  const tokens: Tok[] = [{ type: "text", v: s }];

  // Apply link first so its inner text isn't mis-tokenized as italic, etc.
  const expandLink = () => {
    const out: Tok[] = [];
    for (const t of tokens) {
      if (t.type !== "text") { out.push(t); continue; }
      let last = 0;
      const str = t.v;
      const re = new RegExp(LINK.source, "g");
      let m: RegExpExecArray | null;
      while ((m = re.exec(str)) !== null) {
        if (m.index > last) out.push({ type: "text", v: str.slice(last, m.index) });
        out.push({ type: "link", v: m[1], href: m[2] });
        last = m.index + m[0].length;
      }
      if (last < str.length) out.push({ type: "text", v: str.slice(last) });
    }
    tokens.length = 0;
    tokens.push(...out);
  };

  const expand = (regex: RegExp, kind: "bold" | "italic" | "code", group: number) => {
    const out: Tok[] = [];
    for (const t of tokens) {
      if (t.type !== "text") { out.push(t); continue; }
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

  expandLink();
  expand(BOLD, "bold", 0);
  expand(ITALIC, "italic", 1);
  expand(CODE, "code", 0);

  return tokens.map((t, i) => {
    if (t.type === "bold")
      return <strong key={i} style={{ fontWeight: 700, color: "#000" }}>{t.v}</strong>;
    if (t.type === "italic") return <em key={i}>{t.v}</em>;
    if (t.type === "code")
      return (
        <code key={i} style={{
          fontFamily: '"JetBrains Mono", "SF Mono", Menlo, Consolas, monospace',
          background: "#f1f1ef", padding: "1px 6px", borderRadius: 4, fontSize: "0.92em",
        }}>{t.v}</code>
      );
    if (t.type === "link")
      return (
        <span key={i} style={{ color: "#854d0e", borderBottom: "1px solid #fde68a" }}>
          {t.v}
        </span>
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
  type Block =
    | { kind: "p"; lines: string[] }
    | { kind: "ul"; items: string[] }
    | { kind: "ol"; items: string[] }
    | { kind: "h"; level: number; text: string };
  const blocks: Block[] = [];
  let cur: Block | null = null;
  const flush = () => { if (cur) { blocks.push(cur); cur = null; } };
  for (const raw of lines) {
    const ln = raw.trim();
    if (!ln) { flush(); continue; }
    const hm = /^(#{1,6})\s+(.*)$/.exec(ln);
    const ulm = /^[-*]\s+(.*)$/.exec(ln);
    const olm = /^\d+[.)]\s+(.*)$/.exec(ln);
    if (hm) {
      flush();
      blocks.push({ kind: "h", level: hm[1].length, text: hm[2] });
    } else if (ulm) {
      if (!cur || cur.kind !== "ul") { flush(); cur = { kind: "ul", items: [] }; }
      cur.items.push(ulm[1]);
    } else if (olm) {
      if (!cur || cur.kind !== "ol") { flush(); cur = { kind: "ol", items: [] }; }
      cur.items.push(olm[1]);
    } else {
      if (!cur || cur.kind !== "p") { flush(); cur = { kind: "p", lines: [] }; }
      cur.lines.push(ln);
    }
  }
  flush();

  return (
    <div style={style}>
      {blocks.map((b, i) => {
        if (b.kind === "h") {
          const sz = b.level <= 2 ? "1rem" : b.level === 3 ? "0.96rem" : "0.92rem";
          return (
            <div
              key={i}
              style={{
                fontWeight: 700,
                fontSize: sz,
                color: "#0a0a0a",
                margin: i === 0 ? "0 0 6px" : "10px 0 6px",
                lineHeight: 1.3,
              }}
            >
              {renderInline(b.text)}
            </div>
          );
        }
        if (b.kind === "p")
          return (
            <p key={i} style={{ margin: i === 0 ? "0 0 8px" : "8px 0", lineHeight: 1.55 }}>
              {renderInline(b.lines.join(" "))}
            </p>
          );
        if (b.kind === "ul")
          return (
            <ul key={i} style={{ margin: "6px 0", paddingLeft: 22, lineHeight: 1.55 }}>
              {b.items.map((it, j) => (
                <li key={j} style={{ marginBottom: 4 }}>{renderInline(it)}</li>
              ))}
            </ul>
          );
        return (
          <ol key={i} style={{ margin: "6px 0", paddingLeft: 22, lineHeight: 1.55 }}>
            {b.items.map((it, j) => (
              <li key={j} style={{ marginBottom: 4 }}>{renderInline(it)}</li>
            ))}
          </ol>
        );
      })}
    </div>
  );
};
