const variants = {
  A: "证据桌",
  B: "沉浸舞台",
  C: "编辑档案",
};

const sessions = [
  { id: "s1", title: "连续性与左右极限", meta: "今天 · 48 分钟", status: "录制中", tone: "live" },
  { id: "s2", title: "产品复盘：搜索体验", meta: "昨天 · 1 小时 12 分", status: "已完成", tone: "done" },
  { id: "s3", title: "WASAPI 音频排障", meta: "7 月 31 日 · 36 分钟", status: "已固定", tone: "pin" },
  { id: "s4", title: "供应商接口设计评审", meta: "7 月 29 日 · 54 分钟", status: "已完成", tone: "done" },
];

const events = {
  f1: { kind: "frame", time: "18:42", title: "定义左右极限", detail: "显示器 1 · 关键帧 · 变化度 34%", visual: "limit" },
  f2: { kind: "frame", time: "19:06", title: "分段函数图像", detail: "显示器 1 · 关键帧 · 变化度 62%", visual: "graph" },
  f3: { kind: "frame", time: "19:19", title: "题目与四个选项", detail: "显示器 1 · 被回答引用", visual: "quiz" },
  f4: { kind: "frame", time: "19:41", title: "老师给出结论", detail: "显示器 1 · 关键帧 · 变化度 47%", visual: "proof" },
  t1: { kind: "transcript", time: "18:38–18:52", title: "左极限只关心从左侧逼近时的函数值。", detail: "系统声音 · Whisper 原文" },
  t2: { kind: "transcript", time: "19:14–19:27", title: "在 x=1 处，左右极限不同，因此这里不连续。", detail: "系统声音 · 已校订 · 被回答引用" },
  t3: { kind: "transcript", time: "19:34–19:45", title: "注意选项 D 描述的是可去间断点，并不符合这张图。", detail: "系统声音 · 待校订" },
  q1: { kind: "question", time: "19:22", title: "这是什么题？为什么选 D 不对？", detail: "2 分钟上下文 · 使用 1 张关键帧、1 条字幕" },
  gap1: { kind: "gap", time: "27:04–28:16", title: "主动暂停", detail: "全部来源暂停 1 分 12 秒" },
};

const state = {
  variant: new URLSearchParams(location.search).get("variant") || "A",
  session: "s1",
  selected: "f3",
  zoom: "5 分钟",
  recording: true,
  correction: true,
};

if (!variants[state.variant]) state.variant = "A";

const evidenceIds = new Set(["f3", "t2"]);

function icon(name) {
  const icons = {
    search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"></circle><path d="m16 16 4 4"></path></svg>',
    pin: '<svg viewBox="0 0 24 24"><path d="m8 4 8 0-1 6 3 3H6l3-3z"></path><path d="M12 13v7"></path></svg>',
    dots: '<svg viewBox="0 0 24 24"><circle cx="5" cy="12" r="1"></circle><circle cx="12" cy="12" r="1"></circle><circle cx="19" cy="12" r="1"></circle></svg>',
    play: '<svg viewBox="0 0 24 24"><path d="m9 6 9 6-9 6z"></path></svg>',
    pause: '<svg viewBox="0 0 24 24"><path d="M8 6v12M16 6v12"></path></svg>',
    ask: '<svg viewBox="0 0 24 24"><path d="M5 5h14v11H9l-4 3z"></path><path d="M9 9h6M9 12h4"></path></svg>',
    screen: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="13" rx="2"></rect><path d="M9 21h6M12 17v4"></path></svg>',
    mic: '<svg viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"></rect><path d="M5 11a7 7 0 0 0 14 0M12 18v3"></path></svg>',
    speaker: '<svg viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4zM17 9a4 4 0 0 1 0 6M19 6a8 8 0 0 1 0 12"></path></svg>',
    chevron: '<svg viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"></path></svg>',
  };
  return icons[name] || "";
}

function frameVisual(type, compact = false) {
  const content = {
    limit: '<span class="formula">lim<sub>x→1⁻</sub> f(x) = 2</span><i></i><i></i>',
    graph: '<span class="axis axis-x"></span><span class="axis axis-y"></span><span class="curve"></span><b class="point p1"></b><b class="point p2"></b>',
    quiz: '<em>题目 4</em><strong>判断 f(x) 在 x = 1 处的连续性</strong><span>A&nbsp; 连续</span><span>B&nbsp; 跳跃间断</span><span>C&nbsp; 无穷间断</span><span>D&nbsp; 可去间断</span>',
    proof: '<em>结论</em><strong>左极限 ≠ 右极限</strong><span>∴ x = 1 处不连续</span>',
  };
  return `<div class="frame-visual frame-${type} ${compact ? "compact" : ""}">${content[type]}</div>`;
}

function sessionList(mode = "rail") {
  return `<div class="session-list ${mode}">
    ${sessions.map((item) => `
      <button class="session-item ${state.session === item.id ? "active" : ""}" data-session="${item.id}">
        <span class="session-mark ${item.tone}"></span>
        <span class="session-copy"><strong>${item.title}</strong><small>${item.meta}</small></span>
        ${item.tone === "pin" ? `<span class="session-pin">${icon("pin")}</span>` : ""}
      </button>`).join("")}
  </div>`;
}

function timeline({ dense = false, ledger = false } = {}) {
  const selected = state.selected;
  const eventClass = (id) => `${selected === id ? "selected" : ""} ${evidenceIds.has(id) ? "evidence" : ""}`;
  return `<section class="timeline ${dense ? "dense" : ""} ${ledger ? "ledger" : ""}">
    <header class="timeline-toolbar">
      <div><span class="eyebrow">统一时间线</span><strong>00:00 — 48:12</strong></div>
      <div class="zoom-control" aria-label="时间线缩放">
        ${["整段", "5 分钟", "1 分钟", "秒级"].map((zoom) => `<button data-zoom="${zoom}" class="${state.zoom === zoom ? "active" : ""}">${zoom}</button>`).join("")}
      </div>
    </header>
    <div class="time-ruler"><span>18:00</span><span>19:00</span><span>20:00</span><span>21:00</span><span>22:00</span></div>
    <div class="track frame-track">
      <label>画面</label>
      <div class="track-content">
        ${["f1", "f2", "f3", "f4"].map((id) => `<button class="timeline-frame ${eventClass(id)}" data-event="${id}">${frameVisual(events[id].visual, true)}<time>${events[id].time}</time></button>`).join("")}
      </div>
    </div>
    <div class="track transcript-track">
      <label>字幕</label>
      <div class="track-content">
        <button class="segment raw ${eventClass("t1")}" data-event="t1" style="--start:3%;--width:29%"><small>原文</small><span>${events.t1.title}</span></button>
        <button class="segment corrected ${eventClass("t2")}" data-event="t2" style="--start:36%;--width:36%"><small>已校订</small><span>${events.t2.title}</span></button>
        <button class="segment pending ${eventClass("t3")}" data-event="t3" style="--start:74%;--width:24%"><small>待校订</small><span>${events.t3.title}</span></button>
      </div>
    </div>
    <div class="track activity-track">
      <label>事件</label>
      <div class="track-content">
        <button class="question-pin ${eventClass("q1")}" data-event="q1" style="--start:59%">Q</button>
        <button class="gap-block ${eventClass("gap1")}" data-event="gap1" style="--start:78%;--width:12%"><span>暂停 1:12</span></button>
      </div>
    </div>
    <footer class="timeline-legend"><span><i class="legend-evidence"></i>回答引用</span><span><i class="legend-selected"></i>当前选择</span><span>滚轮缩放 · Shift + 滚轮横移</span></footer>
  </section>`;
}

function evidenceAnswer(compact = false) {
  return `<article class="answer-card ${compact ? "compact" : ""}">
    <header><span class="answer-index">Q1</span><div><small>19:22 · 2 分钟上下文</small><strong>这是什么题？为什么选 D 不对？</strong></div><button class="icon-button">${icon("dots")}</button></header>
    <div class="answer-body">
      <p class="answer-label confirmed">会话证据确认</p>
      <p>题目要求判断分段函数在 <b>x = 1</b> 处的连续性。字幕明确指出左右极限不同，因此这里是<strong>跳跃间断点</strong>，不是可去间断点。</p>
      <div class="evidence-chips">
        <button data-event="f3"><span class="mini-frame">${frameVisual("quiz", true)}</span><span><b>关键帧</b><small>19:19 · 显示器 1</small></span></button>
        <button data-event="t2"><span class="quote-mark">“</span><span><b>已校订字幕</b><small>19:14–19:27</small></span></button>
      </div>
      <p class="answer-label supplemental">补充解释</p>
      <p>可去间断要求左右极限相同，只是函数值缺失或不等于该极限；这与当前图像不符。</p>
    </div>
  </article>`;
}

function detailPanel(mode = "standard") {
  const item = events[state.selected] || events.f3;
  const isFrame = item.kind === "frame";
  return `<aside class="detail-panel ${mode}">
    <header><div><span class="eyebrow">证据详情</span><strong>${item.time}</strong></div><button class="icon-button">${icon("dots")}</button></header>
    ${isFrame ? frameVisual(item.visual) : `<div class="detail-symbol ${item.kind}">${item.kind === "transcript" ? "字" : item.kind === "question" ? "问" : "停"}</div>`}
    <div class="detail-copy"><span class="evidence-state">${evidenceIds.has(state.selected) ? "被 Q1 引用" : "时间线项目"}</span><h2>${item.title}</h2><p>${item.detail}</p></div>
    ${item.kind === "transcript" ? `<div class="version-stack"><button class="active"><b>最新有效版本</b><span>${item.title}</span></button><button><b>Whisper 原文</b><span>在 x 等于一处，左右极限不一样，所以这里不连续。</span></button><a href="#">查看差异与模型来源 →</a></div>` : ""}
    <div class="detail-actions"><button>在时间线居中</button><button>复制引用</button></div>
  </aside>`;
}

function recordingCapsule(style = "floating") {
  return `<div class="recording-capsule ${style}">
    <span class="record-dot"></span>
    <div class="capsule-time"><small>记录中</small><strong>48:12</strong></div>
    <span class="capsule-source">${icon("screen")} 2</span>
    <span class="capsule-source healthy">${icon("speaker")}</span>
    <span class="capsule-source healthy">${icon("mic")}</span>
    <button data-recording-toggle title="${state.recording ? "暂停" : "继续"}">${icon(state.recording ? "pause" : "play")}</button>
    <button class="capsule-ask">${icon("ask")} 提问</button>
    <button class="capsule-stop">结束</button>
  </div>`;
}

function brand() {
  return `<div class="brand"><span class="brand-seal">境</span><div><strong>境织</strong><small>JINGZHI</small></div></div>`;
}

function renderA() {
  return `<div class="shell variant-a">
    <aside class="library-rail">
      ${brand()}
      <button class="new-session"><span>＋</span> 新建会话</button>
      <label class="search-field">${icon("search")}<input placeholder="搜索会话与字幕" /></label>
      <div class="library-heading"><span>最近会话</span><button>${icon("dots")}</button></div>
      ${sessionList()}
      <footer><button class="avatar">C</button><span><b>本地资料库</b><small>18 个会话 · 6.8 GB</small></span><button>${icon("chevron")}</button></footer>
    </aside>
    <main class="workspace">
      <header class="workspace-header"><div><span class="breadcrumb">会话 / 今天</span><h1>连续性与左右极限</h1><p><span class="live-pill">● 记录中</span> 今天 18:32 开始 · 系统声音 + 麦克风 · 2 个显示器</p></div><div class="header-actions"><button>固定</button><button>导出</button><button class="primary">生成会话材料</button></div></header>
      <section class="workspace-main">
        <div class="timeline-column">${timeline()}<div class="question-strip"><span>当前问题</span><strong>这是什么题？为什么选 D 不对？</strong><button>查看回答</button></div>${evidenceAnswer(true)}</div>
        ${detailPanel()}
      </section>
    </main>
    ${recordingCapsule()}
  </div>`;
}

function renderB() {
  const current = events[state.selected]?.kind === "frame" ? events[state.selected] : events.f3;
  return `<div class="shell variant-b">
    <header class="cinema-header">${brand()}<div class="cinema-session"><span class="live-pill">● LIVE</span><strong>连续性与左右极限</strong><small>48:12 / 正在记录</small></div><div class="cinema-actions"><button>${icon("search")}</button><button>${icon("dots")}</button></div></header>
    <main class="cinema-stage">
      <section class="stage-view">
        <div class="stage-meta"><span>当前证据</span><strong>${current.time}</strong><small>显示器 1 · 题目讲解</small></div>
        ${frameVisual(current.visual || "quiz")}
        <div class="stage-question"><span>Q1</span><p>这是什么题？为什么选 D 不对？</p><button data-event="t2">2 项证据</button></div>
      </section>
      <aside class="answer-drawer">${evidenceAnswer()}</aside>
    </main>
    <section class="cinema-timeline">${timeline({ dense: true })}</section>
    ${recordingCapsule("docked")}
  </div>`;
}

function renderC() {
  return `<div class="shell variant-c">
    <header class="editorial-header">${brand()}<nav><button class="active">工作台</button><button>会话库</button><button>材料</button><button>设置</button></nav><div class="editorial-status"><span class="record-dot"></span><b>48:12</b><small>正在记录 3 个来源</small></div></header>
    <main class="editorial-main">
      <section class="editorial-lead"><div class="issue-no">会话 018</div><div><span class="eyebrow">2026 年 8 月 2 日 · 数学</span><h1>连续性<br />与左右极限</h1><p>从课堂画面、系统声音与提问中编织出的可验证记录。</p></div><div class="editorial-controls"><button class="round">${icon(state.recording ? "pause" : "play")}</button><button class="accent">${icon("ask")} 记录一个问题</button></div></section>
      <section class="editorial-body">
        <aside class="folio">${sessionList("folio-list")}</aside>
        <div class="ledger-column">${timeline({ ledger: true })}<div class="editorial-answer"><div class="margin-note"><span>Q1</span><time>19:22</time><i></i></div>${evidenceAnswer()}</div></div>
        ${detailPanel("paper")}
      </section>
    </main>
    ${recordingCapsule("minimal")}
  </div>`;
}

function render() {
  const app = document.querySelector("#app");
  app.innerHTML = state.variant === "A" ? renderA() : state.variant === "B" ? renderB() : renderC();
  document.querySelector("#variant-label").textContent = `${state.variant} — ${variants[state.variant]}`;
  bindInteractions();
}

function setVariant(next) {
  state.variant = next;
  const params = new URLSearchParams(location.search);
  params.set("variant", next);
  history.replaceState({}, "", `${location.pathname}?${params}`);
  render();
}

function cycleVariant(step) {
  const keys = Object.keys(variants);
  const current = keys.indexOf(state.variant);
  setVariant(keys[(current + step + keys.length) % keys.length]);
}

function bindInteractions() {
  document.querySelectorAll("[data-session]").forEach((button) => button.addEventListener("click", () => { state.session = button.dataset.session; render(); }));
  document.querySelectorAll("[data-event]").forEach((button) => button.addEventListener("click", () => { state.selected = button.dataset.event; render(); }));
  document.querySelectorAll("[data-zoom]").forEach((button) => button.addEventListener("click", () => { state.zoom = button.dataset.zoom; render(); }));
  document.querySelectorAll("[data-recording-toggle]").forEach((button) => button.addEventListener("click", () => { state.recording = !state.recording; render(); }));
}

document.querySelectorAll("[data-variant-step]").forEach((button) => button.addEventListener("click", () => cycleVariant(Number(button.dataset.variantStep))));

document.addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName) || document.activeElement?.isContentEditable) return;
  cycleVariant(event.key === "ArrowRight" ? 1 : -1);
});

render();
