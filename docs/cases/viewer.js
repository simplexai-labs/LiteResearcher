// viewer.js – render selected qwen3-4B-RL cases.
const BENCH_LIST_EL = document.getElementById("bench-list");
const MAIN_EL = document.getElementById("main");

const STATE = {
  index: null,
  benchData: {},     // benchmark -> {benchmark, cases: [...]}
  current: null,     // {benchmark, caseId}
};

async function loadJSON(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`fetch ${path} failed (${resp.status})`);
  return resp.json();
}

function escapeHTML(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderMarkdown(md) {
  if (md === null || md === undefined) return "";
  let text = String(md).replace(/\r\n/g, "\n");

  // Protect fenced code blocks first, then render other markdown tokens.
  const codeBlocks = [];
  text = text.replace(/```([^\n`]*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.push({ lang: (lang || "").trim(), code: escapeHTML(code) }) - 1;
    return `@@CODEBLOCK_${idx}@@`;
  });

  let html = escapeHTML(text);

  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
  // Strip any unmatched bold markers so they don't show up as literal **.
  html = html.replace(/\*\*+/g, "");

  const lines = html.split("\n");
  const out = [];
  let inUl = false;
  let inOl = false;

  function closeLists() {
    if (inUl) {
      out.push("</ul>");
      inUl = false;
    }
    if (inOl) {
      out.push("</ol>");
      inOl = false;
    }
  }

  for (const line of lines) {
    const t = line.trim();
    if (!t) {
      closeLists();
      continue;
    }

    const h = t.match(/^(#{1,3})\s+(.+)$/);
    if (h) {
      closeLists();
      const level = h[1].length;
      out.push(`<h${level}>${h[2]}</h${level}>`);
      continue;
    }

    const ul = t.match(/^[-*]\s+(.+)$/);
    if (ul) {
      if (inOl) {
        out.push("</ol>");
        inOl = false;
      }
      if (!inUl) {
        out.push("<ul>");
        inUl = true;
      }
      out.push(`<li>${ul[1]}</li>`);
      continue;
    }

    const ol = t.match(/^\d+\.\s+(.+)$/);
    if (ol) {
      if (inUl) {
        out.push("</ul>");
        inUl = false;
      }
      if (!inOl) {
        out.push("<ol>");
        inOl = true;
      }
      out.push(`<li>${ol[1]}</li>`);
      continue;
    }

    closeLists();
    out.push(`<p>${t}</p>`);
  }
  closeLists();

  let rendered = out.join("\n");
  rendered = rendered.replace(/@@CODEBLOCK_(\d+)@@/g, (_, i) => {
    const b = codeBlocks[Number(i)];
    if (!b) return "";
    return `<pre><code>${b.code}</code></pre>`;
  });
  return rendered;
}

function shortHost(url) {
  try {
    const u = new URL(url);
    return u.host.replace(/^www\./, "");
  } catch (e) {
    return url;
  }
}

function tagFor(meta) {
  if (meta.tongyi_correct === false) {
    return `<span class="tag warn" title="Tongyi 没做对">通义错</span>`;
  }
  if (meta.tongyi_correct === true) {
    return `<span class="tag muted" title="Tongyi 也对">通义对</span>`;
  }
  return "";
}


function renderSidebar() {
  if (!STATE.index) return;
  const html = STATE.index.benchmarks.map(b => {
    const items = b.cases.map(c => {
      const isActive = STATE.current && STATE.current.benchmark === b.benchmark && STATE.current.caseId === c.id;
      return `
        <div class="case-item ${isActive ? "active" : ""}" data-bench="${escapeHTML(b.benchmark)}" data-id="${escapeHTML(String(c.id))}">
          <div style="color:var(--text);font-size:12.5px;line-height:1.4">${escapeHTML(c.question.slice(0, 90))}${c.question.length > 90 ? "…" : ""}</div>
        </div>`;
    }).join("");
    const open = STATE.current && STATE.current.benchmark === b.benchmark ? "open" : "";
    return `
      <div class="bench ${open}" data-bench="${escapeHTML(b.benchmark)}">
        <div class="bench-head">
          <span><strong>${escapeHTML(b.benchmark)}</strong></span>
        </div>
        <div class="cases">${items}</div>
      </div>`;
  }).join("");
  BENCH_LIST_EL.innerHTML = html;

  BENCH_LIST_EL.querySelectorAll(".bench-head").forEach(el => {
    el.addEventListener("click", () => {
      el.parentElement.classList.toggle("open");
    });
  });
  BENCH_LIST_EL.querySelectorAll(".case-item").forEach(el => {
    el.addEventListener("click", () => {
      const bench = el.getAttribute("data-bench");
      const id = el.getAttribute("data-id");
      selectCase(bench, id);
    });
  });
}


async function selectCase(bench, idRaw) {
  if (!STATE.benchData[bench]) {
    try {
      STATE.benchData[bench] = await loadJSON(`cases/${bench}.json`);
    } catch (e) {
      MAIN_EL.innerHTML = `<div class="empty">加载失败: ${escapeHTML(e.message)}</div>`;
      return;
    }
  }
  const data = STATE.benchData[bench];
  const caseObj = data.cases.find(c => String(c.id) === String(idRaw));
  if (!caseObj) {
    MAIN_EL.innerHTML = `<div class="empty">找不到 case id=${escapeHTML(idRaw)}</div>`;
    return;
  }
  STATE.current = { benchmark: bench, caseId: caseObj.id };
  const newHash = `#bench=${encodeURIComponent(bench)}&id=${encodeURIComponent(caseObj.id)}`;
  if (location.hash !== newHash) {
    history.replaceState(null, "", newHash);
  }
  renderSidebar();
  renderCase(caseObj);
}

function renderCase(c) {
  const stats = c.stats || {};
  const statTags = [
    `<span class="tag muted">🔍 ${stats.n_search} 次搜索</span>`,
    `<span class="tag muted">🌐 ${stats.n_visit} 次访问</span>`,
    `<span class="tag muted">🪐 ${stats.n_domains} 个域名</span>`,
    `<span class="tag muted">🧠 ${stats.n_think_turns} 轮思考</span>`,
    c.judge_correct ? `<span class="tag ok">✓ correct</span>` : `<span class="tag bad">✗ wrong</span>`,
  ].filter(Boolean).join(" ");

  // If the last assistant step's answer matches c.final_answer, don't render
  // the standalone final-answer block (avoid showing the same text twice).
  let lastAssistantAnswer = "";
  for (let i = c.steps.length - 1; i >= 0; i--) {
    const st = c.steps[i];
    if (st && st.type === "assistant" && st.answer) {
      lastAssistantAnswer = st.answer;
      break;
    }
  }
  const normalize = (s) => String(s || "").replace(/\s+/g, " ").trim();
  const stepsHTML = c.steps.map((s, idx) => renderStep(s, idx)).join("");

  const finalAnswer =
    c.final_answer && normalize(c.final_answer) !== normalize(lastAssistantAnswer)
      ? `<div class="answer-box"><div class="label">最终答案 (final_answer)</div><div class="md">${renderMarkdown(c.final_answer)}</div></div>`
      : "";

  MAIN_EL.innerHTML = `
    <div class="case-header">
      <div class="breadcrumb">${escapeHTML(c.benchmark)}</div>
      <h2>${escapeHTML(c.question)}</h2>
      <div class="stats">${statTags}</div>
      <div class="ans-row">
        <div class="ans-col"><div class="label">参考答案 (reference)</div><div class="val">${escapeHTML(c.reference_answer || "—")}</div></div>
      </div>
    </div>
    ${stepsHTML}
    ${finalAnswer}
  `;
  attachToggles();
  MAIN_EL.scrollTo({ top: 0, behavior: "smooth" });
}


function renderStep(s, idx) {
  if (s.type === "question") {
    return `
      <div class="step question">
        <div class="role"><span class="badge" style="color:var(--accent)">❓ Question</span><span class="turn">step ${idx}</span></div>
        <div class="think">${escapeHTML((s.content || "").replace(/^Question:\s*/i, ""))}</div>
      </div>`;
  }
  if (s.type === "tool_response") {
    const content = s.content || "";
    return `
      <div class="step tool_response">
        <div class="role"><span class="badge" style="color:var(--tool)">🛰 Tool Response</span><span class="turn">step ${idx}</span></div>
        <div class="tool-resp-body"><pre>${escapeHTML(content)}</pre></div>
        <button class="toggle">展开 / 收起 (${content.length} chars)</button>
      </div>`;
  }
  if (s.type === "assistant") {
    const think = s.think
      ? `<div class="think">${escapeHTML(s.think)}</div>`
      : "";
    const calls = (s.tool_calls || []).map(tc => renderToolCall(tc)).join("");
    const answer = s.answer
      ? `<div class="answer-box"><div class="label">&lt;answer&gt;</div><div class="md">${renderMarkdown(s.answer)}</div></div>`
      : "";
    return `
      <div class="step assistant">
        <div class="role"><span class="badge" style="color:var(--think)">🧠 Assistant</span><span class="turn">step ${idx}</span></div>
        ${think}
        ${calls}
        ${answer}
      </div>`;
  }
  return "";
}

function renderToolCall(tc) {
  if (tc.name === "search") {
    const items = (tc.queries || []).map(q => `<div class="q">🔍 ${escapeHTML(q)}</div>`).join("");
    return `
      <div class="tool-call search">
        <div class="tc-head"><span class="badge">SEARCH</span></div>
        <div class="queries">${items}</div>
      </div>`;
  }
  if (tc.name === "visit") {
    const items = (tc.urls || []).map(u => `<div class="u">🌐 <a href="${escapeHTML(u)}" target="_blank" rel="noopener">${escapeHTML(shortHost(u))}</a> <span style="color:var(--muted);font-size:11px"> ${escapeHTML(u)}</span></div>`).join("");
    const goal = tc.goal ? `<div class="goal">目标: ${escapeHTML(tc.goal)}</div>` : "";
    return `
      <div class="tool-call visit">
        <div class="tc-head"><span class="badge">VISIT</span></div>
        <div class="urls">${items}</div>
        ${goal}
      </div>`;
  }
  return `
    <div class="tool-call">
      <div class="tc-head"><span class="badge">${escapeHTML(tc.name || "TOOL")}</span></div>
      <pre>${escapeHTML(JSON.stringify(tc.arguments || tc, null, 2))}</pre>
    </div>`;
}

function attachToggles() {
  MAIN_EL.querySelectorAll(".toggle").forEach(btn => {
    btn.addEventListener("click", () => {
      const body = btn.previousElementSibling;
      body.classList.toggle("expanded");
    });
  });
}


function parseHash() {
  const h = (location.hash || "").replace(/^#/, "");
  if (!h) return null;
  const params = {};
  for (const part of h.split("&")) {
    const [k, v] = part.split("=");
    if (k) params[decodeURIComponent(k)] = decodeURIComponent(v || "");
  }
  if (params.bench && params.id) return { bench: params.bench, id: params.id };
  return null;
}

async function boot() {
  try {
    STATE.index = await loadJSON("cases/index.json");
  } catch (e) {
    BENCH_LIST_EL.textContent = "无法加载 cases/index.json,请用 HTTP 服务器打开 (例如 python -m http.server)。";
    return;
  }
  const hash = parseHash();
  if (hash) {
    STATE.current = { benchmark: hash.bench, caseId: hash.id };
  } else if (STATE.index.benchmarks.length) {
    const first = STATE.index.benchmarks[0];
    STATE.current = { benchmark: first.benchmark, caseId: first.cases[0].id };
  }
  renderSidebar();
  if (STATE.current) {
    selectCase(STATE.current.benchmark, STATE.current.caseId);
  }
  window.addEventListener("hashchange", () => {
    const h = parseHash();
    if (h) selectCase(h.bench, h.id);
  });
}

boot();
