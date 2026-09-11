/* 工具活动只展示服务端事件；文本摘要不参与模型上下文。 */
const state = {
  mode: "traditional",
  running: false,
  results: { traditional: null, agentic: null },
  turn: null,
  timer: null,
};
const elements = typeof document === "undefined" ? {} : {
  workspace: document.querySelector("#workspace"),
  modes: [...document.querySelectorAll(".mode")],
  question: document.querySelector("#question-text"),
  connection: document.querySelector("#connection-status"),
  description: document.querySelector("#mode-description"),
  run: document.querySelector("#run-button"),
  runLabel: document.querySelector("#run-label"),
  resultArea: document.querySelector("#result-area"),
  resultMode: document.querySelector("#result-mode"),
  resultTime: document.querySelector("#result-time"),
  notice: document.querySelector("#run-notice"),
  activity: document.querySelector("#activity"),
  indicator: document.querySelector("#activity-indicator"),
  activityCount: document.querySelector("#activity-count"),
  process: document.querySelector("#process"),
  processTitle: document.querySelector("#process-title"),
  processNote: document.querySelector("#process-note"),
  liveStatus: document.querySelector("#live-status"),
  liveStatusText: document.querySelector("#live-status-text"),
  answerSection: document.querySelector("#answer-section"),
  answer: document.querySelector("#answer"),
  evidenceSection: document.querySelector("#evidence-section"),
  evidence: document.querySelector("#evidence"),
  evidenceCount: document.querySelector("#evidence-count"),
  footer: document.querySelector("#result-footer"),
  metricStatus: document.querySelector("#metric-status"),
  metricLatency: document.querySelector("#metric-latency"),
  metricTokens: document.querySelector("#metric-tokens"),
  metricCount: document.querySelector("#metric-count"),
};
const modeCopy = {
  traditional: { title: "传统 RAG", description: "单次检索，使用 Top-5 候选回答" },
  agentic: { title: "Mini-Agent", description: "逐步定位、读取与补证 · 最多 6 次工具调用" },
};
const statusLabels = {
  running: "调用中", success: "完成", error: "失败",
  missing_result: "未收到结果", interrupted: "连接中断",
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInline(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}

function tableCells(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function isTableDivider(line) {
  return /^\s*\|?\s*:?-{3,}/.test(line) && line.includes("|");
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").replaceAll("\r", "").split("\n");
  const output = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    if (line.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      const headers = tableCells(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      output.push(
        `<div class="table-scroll"><table><thead><tr>${headers.map((cell) => `<th>${renderInline(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell).replaceAll("&lt;br&gt;", "<br>")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`,
      );
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = Math.min(heading[1].length + 2, 6);
      output.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^[-*]\s+/, ""));
        index += 1;
      }
      output.push(`<ul>${items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</ul>`);
      continue;
    }
    if (/^\d+[.)]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\d+[.)]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\d+[.)]\s+/, ""));
        index += 1;
      }
      output.push(`<ol>${items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</ol>`);
      continue;
    }
    const paragraph = [line];
    index += 1;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,4})\s+/.test(lines[index]) &&
      !/^[-*]\s+/.test(lines[index]) &&
      !/^\d+[.)]\s+/.test(lines[index])
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    output.push(`<p>${renderInline(paragraph.join("\n")).replaceAll("\n", "<br>")}</p>`);
  }
  return output.join("");
}


function basename(path) {
  return String(path || "").split(/[\\/]/).filter(Boolean).pop() || "/";
}

function toolPresentation(trace) {
  const args = trace.args || {};
  const summary = trace.summary || {};
  const phase = trace.status;
  const action = trace.name || "工具调用";
  const target = trace.name === "search" ? args.query
    : trace.name === "glob" ? args.target
      : trace.name === "grep" ? (Array.isArray(args.patterns) ? args.patterns.join(" / ") : String(args.patterns || ""))
      : basename(summary.path || args.path);
  const first = summary.start_line ?? args.start_line;
  const last = summary.end_line;
  let location = "";
  if (trace.name === "read") {
    if (first != null && last != null) location = "L" + first + "–" + last;
    else if (first != null) location = "从 L" + first + " 开始";
  }
  const notes = [];
  if (Number.isInteger(summary.hit_count)) notes.push(summary.hit_count + " 个候选");
  if (Number.isInteger(summary.match_count)) notes.push(summary.match_count + (trace.name === "grep" ? " 个命中行" : " 个文件"));
  if (trace.name === "grep" && summary.truncated === true) notes.push("结果已截断");
  if (Number.isInteger(summary.entry_count)) notes.push(summary.entry_count + " 个子项");
  if (Number.isInteger(summary.next_line)) notes.push("可续读 L" + summary.next_line);
  if (phase === "error") notes.push("调用失败");
  if (phase === "missing_result" || phase === "interrupted") notes.push(statusLabels[phase]);
  return { action, target: target || "", location, note: notes.join(" · ") };
}

/* 同一个 call id 永远对应同一条记录；迟到的开始事件不能覆盖完成状态。 */
function upsertTool(traces, incoming) {
  const existing = traces.find((item) => item.tool_call_id === incoming.tool_call_id);
  if (existing) {
    const status = existing.status !== "running" && incoming.status === "running"
      ? existing.status : incoming.status ?? existing.status;
    const updated = { ...existing, ...incoming, status,
      summary: { ...(existing.summary || {}), ...(incoming.summary || {}) } };
    Object.assign(existing, updated);
    return existing;
  }
  const inserted = { ...incoming, summary: incoming.summary || {} };
  traces.push(inserted);
  return inserted;
}

function reconcileTools(live, finalTraces) {
  const merged = live.map((trace) => ({ ...trace, summary: { ...trace.summary } }));
  for (const trace of finalTraces || []) upsertTool(merged, trace);
  return merged.map((trace) => ({
    ...trace,
    status: trace.status === "running" ? "missing_result" : trace.status,
  }));
}

/* 每轮独立检查 completed，上一轮的缓存不能让本次截断的流被判为成功。 */
async function consumeAgentStream(body, receive) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;
  const dispatch = (line) => {
    if (!line.trim() || completed) return;
    const event = JSON.parse(line);
    if (event.event === "error") throw new Error(event.data?.message || "运行中断");
    receive(event);
    if (event.event === "completed") completed = true;
  };
  try {
    while (!completed) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) dispatch(line);
      if (done) {
        dispatch(buffer);
        break;
      }
    }
    if (!completed) throw new Error("连接已结束，未收到最终回答。已收到的工具记录已保留。");
  } finally {
    try { await reader.cancel(); } catch { /* 已结束的连接无需重试。 */ }
    reader.releaseLock();
  }
}

function textElement(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function appendDetails(container, label, value, code = false) {
  if (value == null || value === "") return;
  container.append(textElement("div", "detail-label", label));
  container.append(textElement(code ? "pre" : "div", "detail-value", value));
}

function toolRow(trace) {
  let row = [...elements.process.children].find((node) => node.dataset.callId === trace.tool_call_id);
  const fresh = !row;
  if (fresh) {
    row = document.createElement("details");
    row.className = "tool-row";
    row.dataset.callId = trace.tool_call_id;
    row.append(document.createElement("summary"));
    row.append(textElement("div", "tool-body", ""));
    elements.process.append(row);
  }
  row.dataset.status = trace.status;
  const presentation = toolPresentation(trace);
  const summary = row.firstElementChild;
  summary.replaceChildren();
  const icon = textElement("span", "tool-icon", trace.status === "success" ? "✓"
    : trace.status === "running" ? "" : trace.status === "error" ? "!" : "–");
  icon.setAttribute("aria-hidden", "true");
  const action = textElement("span", "tool-action", presentation.action);
  const target = textElement("span", "tool-target", presentation.target);
  target.title = presentation.target;
  const place = textElement("span", "tool-location", presentation.location);
  const note = textElement("span", "tool-note", presentation.note);
  const arrow = textElement("span", "chevron", "›");
  summary.append(icon, action, target, place, note, arrow);
  summary.setAttribute("aria-label",
    [presentation.action, presentation.target, presentation.location, statusLabels[trace.status]].filter(Boolean).join("，"));
  const body = row.lastElementChild;
  body.replaceChildren();
  appendDetails(body, "工具", trace.name);
  const args = trace.args || {};
  if (args.query) appendDetails(body, "查询", args.query);
  if (args.target) appendDetails(body, "查找目标", args.target);
  if (args.path) appendDetails(body, "范围", args.path);
  if (presentation.location) appendDetails(body, "读取位置", presentation.location);
  if (args.limit != null) appendDetails(body, "请求数量", String(args.limit));
  if (presentation.note) appendDetails(body, "返回摘要", presentation.note);
  if (trace.summary?.message) appendDetails(body, "工具消息", trace.summary.message);
  const raw = document.createElement("details");
  raw.className = "raw-call";
  raw.append(textElement("summary", "", "原始参数"));
  raw.append(textElement("pre", "", JSON.stringify(args, null, 2)));
  body.append(raw);
  if (trace.status === "error") row.open = true;
  return row;
}

function activityLabel(turn) {
  return ["ls", "glob", "search", "read", "grep"].map((name) => {
    const count = turn.tools.filter((t) => t.name === name && t.status === "success").length;
    return count ? name + " " + count + " 次" : "";
  }).filter(Boolean).join(" · ");
}

function updateActivity() {
  const turn = state.turn;
  if (!turn || turn.mode !== "agentic") return;
  const pending = turn.tools.filter((t) => t.status === "running");
  const errors = turn.tools.filter((t) => t.status === "error").length;
  elements.processTitle.textContent = turn.phase === "running" ? "正在查阅水循环教学资料"
    : turn.phase === "error" ? "执行中断" : activityLabel(turn) || "执行记录";
  elements.activityCount.textContent = turn.tools.length ? turn.tools.length + " 次调用" + (errors ? " · " + errors + " 次失败" : "") : "";
  elements.indicator.dataset.status = turn.phase;
  elements.liveStatus.hidden = turn.phase !== "running";
  elements.liveStatusText.textContent = pending.length
    ? (toolPresentation(pending[0]).action + " " + toolPresentation(pending[0]).target + "…")
    : turn.tools.length ? "等待模型决定下一步…" : "等待模型选择工具…";
  if (turn.phase === "running") elements.runLabel.textContent = "运行中 · " + turn.tools.length + " 次调用";
}

function renderCandidates(hits) {
  elements.process.replaceChildren();
  for (const [index, hit] of hits.entries()) {
    const item = document.createElement("details");
    item.className = "candidate-row";
    const head = document.createElement("summary");
    head.append(textElement("span", "rank", String(index + 1)),
      textElement("span", "candidate-name", basename(hit.path)),
      textElement("span", "tool-location", "L" + hit.start_line + "–" + hit.end_line),
      textElement("span", "tool-note", typeof hit.score === "number" ? hit.score.toFixed(3) : "未返回分数"),
      textElement("span", "chevron", "›"));
    const body = textElement("div", "tool-body", "");
    appendDetails(body, "路径", hit.path);
    appendDetails(body, "检索片段", hit.preview || "");
    item.append(head, body);
    elements.process.append(item);
  }
  elements.processNote.textContent = hits.length ? "这些是检索候选；分数用于排序，不代表答案置信度。" : "没有返回检索候选。";
}

function renderCitations(citations) {
  elements.evidenceSection.hidden = !citations.length;
  elements.evidence.replaceChildren();
  elements.evidenceCount.textContent = String(citations.length);
  citations.forEach((citation, index) => {
    const item = document.createElement("details");
    item.className = "citation-row";
    const head = document.createElement("summary");
    head.append(textElement("span", "citation-number", String(index + 1)),
      textElement("span", "candidate-name", basename(citation.path)),
      textElement("span", "tool-location", "L" + citation.start_line + "–" + citation.end_line),
      textElement("span", "chevron", "›"));
    const body = textElement("div", "tool-body", "");
    body.append(textElement("blockquote", "citation-quote", citation.quote || ""));
    appendDetails(body, "原文路径", citation.path);
    body.append(textElement("span", "source-check", "路径、行号与原文已核对"));
    item.append(head, body);
    elements.evidence.append(item);
  });
}

function setMode(mode) {
  if (state.running) return;
  state.mode = mode;
  elements.modes.forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  elements.description.textContent = modeCopy[mode].description;
  elements.runLabel.textContent = state.results[mode] ? "重新运行" : "运行" + modeCopy[mode].title;
  const stored = state.results[mode];
  if (stored) {
    state.turn = stored;
    renderTurn(stored, true);
  } else {
    elements.resultArea.hidden = true;
    elements.workspace.classList.remove("has-result");
  }
}

function renderTurn(turn, replay = false) {
  elements.resultArea.hidden = false;
  elements.workspace.classList.add("has-result");
  elements.resultMode.textContent = modeCopy[turn.mode].title;
  elements.resultTime.textContent = replay && turn.finishedAt
    ? "上次运行 " + new Date(turn.finishedAt).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) : "";
  elements.notice.className = "run-notice";
  elements.notice.textContent = "";
  elements.process.replaceChildren();
  elements.answerSection.hidden = !turn.result;
  elements.evidenceSection.hidden = true;
  elements.answer.replaceChildren();
  elements.footer.hidden = turn.phase === "running";
  elements.activity.open = turn.phase !== "completed";
  elements.indicator.dataset.status = turn.phase;
  elements.liveStatus.hidden = turn.phase !== "running";
  if (turn.mode === "agentic") {
    turn.tools.forEach(toolRow);
    elements.processNote.textContent = "点开记录可查看调用参数与返回摘要。";
    updateActivity();
  } else {
    elements.processTitle.textContent = turn.phase === "running" ? "正在检索知识库" : "已检索 " + (turn.result?.retrieval_hits?.length || 0) + " 个候选";
    elements.activityCount.textContent = "";
    elements.processNote.textContent = "";
    elements.liveStatusText.textContent = "正在检索 Top-5 并生成回答…";
    if (turn.result) renderCandidates(turn.result.retrieval_hits || []);
  }
  if (turn.result) {
    const result = turn.result;
    elements.answer.innerHTML = renderMarkdown(result.answer);
    elements.metricStatus.textContent = result.finish_reason === "insufficient" ? "未形成有据回答" : "已返回回答";
    elements.metricLatency.textContent = (result.latency_ms / 1000).toFixed(2) + " s";
    const count = result.token_usage?.total_tokens;
    elements.metricTokens.textContent = count == null ? "Token 未返回" : Number(count).toLocaleString() + " tokens";
    elements.metricCount.textContent = turn.mode === "agentic" ? turn.tools.length + " 次工具调用" : (result.retrieval_hits || []).length + " 个候选";
    renderCitations(result.citations || []);
    if (result.finish_reason === "insufficient") {
      elements.activity.open = true;
      elements.notice.textContent = "本次未形成有据回答，可以展开记录查看执行到哪一步。";
    }
  }
  if (turn.phase === "error") {
    elements.notice.className = "run-notice error";
    elements.notice.textContent = turn.error;
    elements.metricStatus.textContent = "运行中断";
    elements.metricLatency.textContent = "";
    elements.metricTokens.textContent = "Token 未返回";
    elements.metricCount.textContent = turn.tools.length + " 条工具记录已保留";
  }
}

function receiveEvent(event) {
  const turn = state.turn;
  if (event.event === "tool_started" || event.event === "tool_completed") {
    const trace = upsertTool(turn.tools, event.data);
    toolRow(trace);
    updateActivity();
  } else if (event.event === "completed") {
    turn.tools = reconcileTools(turn.tools, event.data.tool_traces);
    turn.result = event.data;
    turn.phase = "completed";
    turn.finishedAt = Date.now();
    renderTurn(turn);
  }
}

async function runCurrentMode() {
  if (state.running) return;
  state.running = true;
  const turn = { mode: state.mode, phase: "running", tools: [], result: null, error: "", startedAt: Date.now() };
  state.turn = turn;
  state.results[turn.mode] = turn;
  elements.run.disabled = true;
  elements.modes.forEach((button) => { button.disabled = true; });
  elements.workspace.setAttribute("aria-busy", "true");
  renderTurn(turn);
  elements.runLabel.textContent = "运行中…";
  state.timer = setInterval(() => {
    if (turn.phase === "running") elements.resultTime.textContent = "已等待 " + Math.floor((Date.now() - turn.startedAt) / 1000) + " s";
  }, 1000);
  try {
    const response = await fetch(turn.mode === "agentic" ? "/demo/pilot/stream" : "/demo/pilot/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: turn.mode }),
    });
    if (!response.ok) throw new Error("请求未完成（HTTP " + response.status + "），可重新运行。");
    if (turn.mode === "agentic") {
      if (!response.body) throw new Error("浏览器未收到事件流。");
      await consumeAgentStream(response.body, receiveEvent);
    } else {
      turn.result = await response.json();
      turn.phase = "completed";
      turn.finishedAt = Date.now();
      renderTurn(turn);
    }
  } catch (error) {
    if (turn.phase !== "completed") {
      turn.phase = "error";
      turn.error = error.message || "连接中断，已收到的工具记录已保留。";
      turn.finishedAt = Date.now();
      turn.tools.forEach((trace) => {
        if (trace.status === "running") trace.status = "interrupted";
      });
      renderTurn(turn);
    }
  } finally {
    clearInterval(state.timer);
    state.running = false;
    elements.workspace.setAttribute("aria-busy", "false");
    elements.run.disabled = false;
    elements.modes.forEach((button) => { button.disabled = false; });
    elements.resultTime.textContent = new Date(turn.finishedAt).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    elements.runLabel.textContent = "重新运行";
  }
}

async function initialize() {
  elements.modes.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  elements.run.addEventListener("click", runCurrentMode);
  setMode("traditional");
  try {
    const response = await fetch("/demo/pilot/config", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const config = await response.json();
    elements.question.textContent = config.question;
    elements.connection.lastElementChild.textContent = "水循环教学资料 · 已就绪";
    elements.run.disabled = false;
  } catch {
    elements.question.textContent = "无法读取演示问题，请检查服务是否启动。";
    elements.connection.dataset.status = "error";
    elements.connection.lastElementChild.textContent = "连接失败";
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { toolPresentation, upsertTool, reconcileTools, consumeAgentStream, renderMarkdown, activityLabel };
}
if (typeof document !== "undefined") initialize();
