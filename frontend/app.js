// ============================================================
// Rodundwerk II - Monitoring Dashboard
// ============================================================

// --- Backend API base URL (single point of configuration) ---
const API_BASE = "";

// --- State ---
let segments = [];
let currentSegmentId = null;
let overviewRefreshInterval = null;
let lastTimelineData = null;
let currentTab = "overview";

// --- Constants ---
// Bench-top prototype — no rotating machine, no blade-passing peaks
const FUNDAMENTAL_FREQ = null;
const VPF_FREQ = null;

const STATE_COLORS = {
  ST: "#6c757d",
  TU: "#0d6efd",
  PU: "#198754",
  PH: "#fd7e14",
  RF: "#dc2626",
  unknown: "#adb5bd",
};

const STATE_LABELS = {
  ST: "Standstill",
  TU: "Turbine",
  PU: "Pump",
  PH: "Phase Shifter",
  RF: "Random Fault",
  unknown: "Unknown",
};

const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "#0d1a24",
  font: {
    color: "#b8c7d8",
    family: '"Space Grotesk", "Segoe UI", sans-serif',
    size: 11,
  },
  margin: { t: 30, r: 20, b: 40, l: 55 },
  xaxis: { gridcolor: "#2a3e51", zerolinecolor: "#2a3e51", color: "#7f94aa" },
  yaxis: { gridcolor: "#2a3e51", zerolinecolor: "#2a3e51", color: "#7f94aa" },
};

const PLOTLY_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

// --- DOM Elements ---
const dom = {
  healthDot: document.getElementById("health-indicator"),
  healthText: document.getElementById("health-text"),
  btnTrain: document.getElementById("btn-train"),
  trainStatus: document.getElementById("train-status"),
  segmentSelector: document.getElementById("segment-selector"),
  statusText: document.getElementById("status-text"),
  statusSpinner: document.getElementById("status-spinner"),
  detectionContent: document.getElementById("detection-content"),
  localizationContent: document.getElementById("localization-content"),
  alertsTbody: document.getElementById("alerts-tbody"),
  btnRefreshAlerts: document.getElementById("btn-refresh-alerts"),
  toastContainer: document.getElementById("toast-container"),
  // Header / KPI stats
  headerHealth: document.getElementById("header-health"),
  headerState: document.getElementById("header-state"),
  headerSegments: document.getElementById("header-segments"),
  kpiHealthy: document.getElementById("kpi-healthy"),
  kpiFaults: document.getElementById("kpi-faults"),
  kpiBestError: document.getElementById("kpi-best-error"),
  // Progress banner
  progressBanner: document.getElementById("progress-banner"),
  progressText: document.getElementById("progress-text"),
  // Operator dashboard elements
  timelineChart: document.getElementById("timeline-chart"),
  machineDiagram: document.getElementById("machine-diagram"),
  anomalyRegion: document.getElementById("anomaly-region-chart"),
  alertFeed: document.getElementById("alert-feed"),
  // Detection tab
  modelsStatusGrid: document.getElementById("models-status-grid"),
  btnTrainDet: document.getElementById("btn-train-det"),
  trainStatusDet: document.getElementById("train-status-det"),
  btnRefreshModels: document.getElementById("btn-refresh-models"),
  detSegLabel: document.getElementById("det-seg-label"),
  anomalyScoresChart: document.getElementById("anomaly-scores-chart"),
  // Localization tab
  machineDiagramLoc: document.getElementById("machine-diagram-loc"),
  locTableContent: document.getElementById("localization-table-content"),
  // Reports tab
  reportsErrorChart: document.getElementById("reports-error-chart"),
  reportsTableContent: document.getElementById("reports-table-content"),
  btnRefreshReports: document.getElementById("btn-refresh-reports"),
};

// ============================================================
// Utilities
// ============================================================

function showToast(message, type = "error", duration = 5000) {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${escapeHtml(message)}</span><button class="toast-close">&times;</button>`;
  toast
    .querySelector(".toast-close")
    .addEventListener("click", () => toast.remove());
  dom.toastContainer.appendChild(toast);
  setTimeout(() => {
    if (toast.parentNode) toast.remove();
  }, duration);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function setStatus(text, loading = false) {
  dom.statusText.textContent = text;
  dom.statusSpinner.classList.toggle("hidden", !loading);
}

function setButtonLoading(btn, loading) {
  btn.disabled = loading;
}

function showProgress(text) {
  dom.progressBanner.classList.remove("hidden");
  dom.progressText.textContent = text;
}

function hideProgress() {
  dom.progressBanner.classList.add("hidden");
}

async function apiFetch(url, options = {}) {
  const headers = { ...options.headers };
  if (options.body) {
    headers["Content-Type"] = "application/json";
  }
  const resp = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers,
  });
  if (!resp.ok) {
    let errMsg = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body.error) errMsg = body.error;
      else if (body.detail) errMsg = body.detail;
    } catch {
      /* ignore parse error */
    }
    throw new Error(errMsg);
  }
  return resp.json();
}

// ============================================================
// Operator Dashboard Functions
// ============================================================

async function fetchOverview() {
  try {
    const data = await apiFetch("/api/overview");

    // Update header stats
    const health = data.health !== undefined ? data.health : 100;
    dom.headerHealth.textContent = `${Math.round(health)}`;
    dom.headerHealth.style.color =
      health >= 70 ? "#2fd492" : health >= 40 ? "#f6bf3a" : "#ff6673";

    const stateCode = data.current_state || "--";
    const stateLabel = STATE_LABELS[stateCode] || stateCode;
    const stateColor = STATE_COLORS[stateCode] || "#6c757d";
    dom.headerState.textContent = stateLabel;
    dom.headerState.style.background = stateColor;

    const parts = [];
    if (data.n_segments !== undefined) parts.push(`${data.n_segments} seg`);
    if (data.n_healthy) parts.push(`${data.n_healthy} healthy`);
    if (data.n_anomalies) parts.push(`${data.n_anomalies} fault`);
    dom.headerSegments.textContent = parts.join(" · ");

    // KPI cards
    if (dom.kpiHealthy) dom.kpiHealthy.textContent = data.n_healthy ?? "--";
    if (dom.kpiFaults) dom.kpiFaults.textContent = data.n_anomalies ?? "--";
    if (dom.kpiBestError) {
      const best = data.best_localization_error_cm;
      dom.kpiBestError.innerHTML =
        best != null
          ? `${best.toFixed(1)}<span class="kpi-unit"> cm</span>`
          : `--<span class="kpi-unit"> cm</span>`;
    }
  } catch {
    // Overview not available yet
  }
}

async function fetchTimeline() {
  try {
    const data = await apiFetch("/api/timeline");
    renderTimeline(data);
  } catch {
    dom.timelineChart.innerHTML =
      '<div class="chart-placeholder">No timeline data available</div>';
  }
}

function renderTimeline(data) {
  const states = data.states || [];
  const anomalies = data.anomalies || [];

  lastTimelineData = data;

  if (states.length === 0) {
    dom.timelineChart.innerHTML =
      '<div class="chart-placeholder">No timeline data available</div>';
    return;
  }

  // Group segments by state for legend entries
  const stateGroups = {};
  for (const seg of states) {
    const code = seg.state_code || "ST";
    if (!stateGroups[code]) stateGroups[code] = [];
    stateGroups[code].push(seg);
  }

  const traces = [];

  for (const [code, segs] of Object.entries(stateGroups)) {
    const color = STATE_COLORS[code] || STATE_COLORS.ST;
    const label = STATE_LABELS[code] || code;
    traces.push({
      type: "bar",
      orientation: "h",
      y: segs.map(() => "Timeline"),
      x: segs.map((s) => (s.duration_s || 1) / 60),
      base: segs.map((s) => (s.offset_s !== undefined ? s.offset_s : 0) / 60),
      marker: { color: color },
      name: label,
      customdata: segs.map((s) => s.segment_id),
      hovertext: segs.map(
        (s) =>
          `${label} ${s.segment_id} (${((s.duration_s || 0) / 60).toFixed(2)} min) — click to select`,
      ),
      hoverinfo: "text",
    });
  }

  // Anomaly markers overlay
  if (anomalies.length > 0) {
    traces.push({
      type: "scatter",
      mode: "markers",
      x: anomalies.map((a) => (a.offset_s !== undefined ? a.offset_s : 0) / 60),
      y: anomalies.map(() => "Timeline"),
      marker: {
        symbol: "diamond",
        size: 12,
        color: "#dc2626",
        line: { color: "#ffffff", width: 1.5 },
      },
      name: "Anomaly",
      customdata: anomalies.map((a) => a.segment_id),
      hovertext: anomalies.map(
        (a) =>
          `Anomaly ${a.segment_id} (score: ${(a.score || 0).toFixed(3)}) — click to select`,
      ),
      hoverinfo: "text",
    });
  }

  // Transition markers overlay
  const transitions = data.transitions || [];
  if (transitions.length > 0) {
    traces.push({
      type: "scatter",
      mode: "markers",
      x: transitions.map(
        (t) => (t.offset_s !== undefined ? t.offset_s : 0) / 60,
      ),
      y: transitions.map(() => "Timeline"),
      marker: {
        symbol: "triangle-up",
        size: 11,
        color: transitions.map((t) => (t.is_valid ? "#fd7e14" : "#dc2626")),
        line: { color: "#ffffff", width: 1.5 },
      },
      name: "Transition",
      customdata: transitions.map((t) => t.segment_id),
      hovertext: transitions.map((t) => {
        const label = `${t.from_state} → ${t.to_state}`;
        const valid = t.is_valid ? "" : " (INVALID)";
        return `Transition: ${label}${valid} — click to select`;
      }),
      hoverinfo: "text",
    });
  }

  // Build selection highlight shape if a segment is selected
  const shapes = [];
  if (currentSegmentId) {
    const selSeg = states.find((s) => s.segment_id === currentSegmentId);
    if (selSeg) {
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: selSeg.offset_s / 60,
        x1: (selSeg.offset_s + (selSeg.duration_s || 1)) / 60,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(37, 99, 235, 0.18)",
        line: { color: "rgba(37, 99, 235, 0.6)", width: 2 },
      });
    }
  }

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    barmode: "overlay",
    height: 160,
    margin: { t: 46, r: 20, b: 36, l: 70 },
    xaxis: {
      ...PLOTLY_LAYOUT_BASE.xaxis,
      title: "Time (min)",
      tickformat: ".1f",
    },
    yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, fixedrange: true },
    legend: {
      orientation: "h",
      y: 1.22,
      x: 0,
      font: { size: 10 },
      bgcolor: "rgba(0,0,0,0)",
    },
    showlegend: true,
    shapes: shapes,
  };

  Plotly.react(dom.timelineChart, traces, layout, PLOTLY_CONFIG);

  // Attach click handler for segment selection
  dom.timelineChart.removeAllListeners &&
    dom.timelineChart.removeAllListeners("plotly_click");
  dom.timelineChart.on("plotly_click", function (eventData) {
    if (eventData.points && eventData.points.length > 0) {
      const segId = eventData.points[0].customdata;
      if (segId) {
        // Update segment selector dropdown
        dom.segmentSelector.value = segId;
        onSegmentChange(segId);
      }
    }
  });
}

async function fetchAnomalyRegion(segId) {
  try {
    const data = await apiFetch(`/api/anomaly-region?seg_id=${segId}`);
    renderAnomalyRegion(data);
  } catch {
    dom.anomalyRegion.innerHTML =
      '<div class="chart-placeholder">No anomaly region data</div>';
  }
}

function renderAnomalyRegion(data) {
  // Bench-top: re-purpose the "anomaly region" panel as a 3D localization scatter.
  const methods = data.methods || {};
  const gt = data.ground_truth_cm || null;

  if (!gt && Object.keys(methods).length === 0) {
    dom.anomalyRegion.innerHTML =
      '<div class="chart-placeholder">No localization data</div>';
    return;
  }

  // Delegate to the shared 3D renderer (same loc structure expected by renderBenchTop3D)
  const locData = {
    methods: methods,
    ground_truth_cm: gt,
    mic_positions_cm: Object.values(data.mic_positions_cm || {}),
    mic_names: Object.keys(data.mic_positions_cm || {}),
    vib_positions_cm: Object.values(data.vib_positions_cm || {}),
    vib_names: Object.keys(data.vib_positions_cm || {}),
    all_fault_positions_cm: data.all_fault_positions_cm || {},
    best_method: data.best_method || "",
    best_error_cm: data.best_error_cm,
  };
  renderBenchTop3D(dom.anomalyRegion, locData);
}

async function fetchMachineDiagram(segId) {
  try {
    const data = await apiFetch(`/api/machine-diagram?seg_id=${segId}`);
    renderMachineDiagram(dom.machineDiagram, data);
    if (dom.machineDiagramLoc)
      renderMachineDiagram(dom.machineDiagramLoc, data);
  } catch {
    dom.machineDiagram.innerHTML =
      '<div class="chart-placeholder">No machine diagram data</div>';
  }
}

function renderMachineDiagram(container, data) {
  // Bench-top prototype: SVG plan view (X–Y top plane).
  // Shows the floor footprint of the 41×41×40 cm box: X on horizontal axis,
  // Y (depth) on vertical axis. Mirrors a bird's-eye layout schematic.
  if (data.type !== "bench_top") {
    container.innerHTML =
      '<div class="chart-placeholder">Unsupported diagram type</div>';
    return;
  }

  const micPos = data.mic_positions_cm || {};
  const vibPos = data.vib_positions_cm || {};
  const faultPos = data.known_fault_positions_cm || {};
  const gt = data.ground_truth_cm || null;
  const bestEst = data.best_estimate_cm || null;
  const bestMeth = data.best_method || "";
  const boxW_cm = data.box_cm ? data.box_cm[0] : 41; // X
  const boxH_cm = data.box_cm ? data.box_cm[1] : 41; // Y (depth)

  // SVG canvas
  const svgW = 380,
    svgH = 310;
  const ml = 44,
    mr = 16,
    mt = 34,
    mb = 38;
  const plotW = svgW - ml - mr; // 320 px wide
  const plotH = svgH - mt - mb; // 238 px tall

  // Physical → screen coordinate mappers
  const sx = (x) => ml + (x / boxW_cm) * plotW;
  const sz = (y) => mt + plotH - (y / boxH_cm) * plotH; // Y=0 at bottom

  let svg =
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svgW} ${svgH}" ` +
    `width="100%" height="100%" style="display:block">`;

  // Background
  svg += `<rect width="${svgW}" height="${svgH}" fill="#0f1822" rx="6"/>`;

  // Title
  svg +=
    `<text x="${svgW / 2}" y="22" text-anchor="middle" font-size="11" font-weight="700" ` +
    `fill="#b8c7d8" font-family='"Space Grotesk",sans-serif'>` +
    `Bench-Top Plan View (X\u2013Y Top View)</text>`;

  // Plot-area background + border
  svg +=
    `<rect x="${ml}" y="${mt}" width="${plotW}" height="${plotH}" fill="#0d1a24" ` +
    `stroke="#2a3e51" stroke-width="1.5" stroke-dasharray="5,3" rx="2"/>`;

  // Grid lines + tick labels
  for (let x = 0; x <= boxW_cm; x += 10) {
    const px = sx(x);
    svg += `<line x1="${px}" y1="${mt}" x2="${px}" y2="${mt + plotH}" stroke="#2a3e51" stroke-width="0.7"/>`;
    svg += `<text x="${px}" y="${mt + plotH + 13}" text-anchor="middle" font-size="8" fill="#7f94aa">${x}</text>`;
  }
  for (let y = 0; y <= boxH_cm; y += 10) {
    const pz = sz(y);
    svg += `<line x1="${ml}" y1="${pz}" x2="${ml + plotW}" y2="${pz}" stroke="#2a3e51" stroke-width="0.7"/>`;
    svg += `<text x="${ml - 6}" y="${pz + 3}" text-anchor="end" font-size="8" fill="#7f94aa">${y}</text>`;
  }

  // Axis labels
  svg +=
    `<text x="${ml + plotW / 2}" y="${svgH - 4}" text-anchor="middle" ` +
    `font-size="10" fill="#b8c7d8" font-family='"Space Grotesk",sans-serif'>X (cm)</text>`;
  svg +=
    `<text x="11" y="${mt + plotH / 2}" text-anchor="middle" ` +
    `font-size="10" fill="#b8c7d8" font-family='"Space Grotesk",sans-serif' ` +
    `transform="rotate(-90 11 ${mt + plotH / 2})">Y (cm)</text>`;

  // Known fault positions — muted diamonds (background)
  for (const [, fpos] of Object.entries(faultPos)) {
    const px = sx(fpos[0]),
      pz = sz(fpos[1]);
    svg +=
      `<polygon points="${px},${pz - 6} ${px + 5},${pz} ${px},${pz + 6} ${px - 5},${pz}" ` +
      `fill="#2a3e51" stroke="#4a6070" stroke-width="0.8" opacity="0.8"/>`;
  }

  // Vibration sensors — orange squares
  for (const [vname, vpos] of Object.entries(vibPos)) {
    const px = sx(vpos[0]),
      pz = sz(vpos[1]);
    const label = vname.replace("vibration_", "");
    svg +=
      `<rect x="${px - 5}" y="${pz - 5}" width="10" height="10" ` +
      `fill="#d97706" stroke="#ffffff" stroke-width="1.5" rx="1"/>`;
    svg +=
      `<text x="${px + 8}" y="${pz + 4}" font-size="7.5" fill="#d97706" font-weight="700" ` +
      `font-family="-apple-system,sans-serif">${escapeHtml(label)}</text>`;
  }

  // Microphones — blue circles
  for (const [mname, mpos] of Object.entries(micPos)) {
    const px = sx(mpos[0]),
      pz = sz(mpos[1]);
    const label = mname.replace("mic_", "");
    svg += `<circle cx="${px}" cy="${pz}" r="5.5" fill="#0284c7" stroke="#ffffff" stroke-width="1.5"/>`;
    svg +=
      `<text x="${px - 9}" y="${pz - 8}" font-size="7.5" fill="#0284c7" font-weight="700" ` +
      `text-anchor="middle" font-family="-apple-system,sans-serif">${escapeHtml(label)}</text>`;
  }

  // Ground truth — solid red diamond
  if (gt && gt.length >= 2) {
    const px = sx(gt[0]),
      pz = sz(gt[1]);
    svg +=
      `<polygon points="${px},${pz - 10} ${px + 8},${pz} ${px},${pz + 10} ${px - 8},${pz}" ` +
      `fill="#dc2626" stroke="#ffffff" stroke-width="2"/>`;
    svg +=
      `<text x="${px + 12}" y="${pz - 8}" font-size="8.5" fill="#dc2626" font-weight="700" ` +
      `font-family="-apple-system,sans-serif">GT</text>`;
  }

  // Best estimate — green cross
  if (bestEst && bestEst.length >= 2) {
    const px = sx(bestEst[0]),
      pz = sz(bestEst[1]);
    const r = 8;
    svg +=
      `<line x1="${px - r}" y1="${pz}" x2="${px + r}" y2="${pz}" ` +
      `stroke="#16a34a" stroke-width="3" stroke-linecap="round"/>`;
    svg +=
      `<line x1="${px}" y1="${pz - r}" x2="${px}" y2="${pz + r}" ` +
      `stroke="#16a34a" stroke-width="3" stroke-linecap="round"/>`;
    svg += `<circle cx="${px}" cy="${pz}" r="3" fill="#16a34a"/>`;
    const methLabel = (LOC_METHOD_LABELS[bestMeth] || bestMeth || "Est").split(
      " ",
    )[0];
    svg +=
      `<text x="${px + 12}" y="${pz + 12}" font-size="8" fill="#16a34a" font-weight="700" ` +
      `font-family="-apple-system,sans-serif">${escapeHtml(methLabel)}</text>`;
  }

  // Legend — compact box, top-right corner
  const lx = ml + plotW - 122,
    ly = mt + 8;
  svg +=
    `<rect x="${lx - 4}" y="${ly - 7}" width="122" height="60" ` +
    `fill="rgba(13,26,36,0.9)" rx="4" stroke="#2a3e51" stroke-width="0.8"/>`;
  // Mic legend row
  svg += `<circle cx="${lx + 5}" cy="${ly + 4}" r="4.5" fill="#0284c7"/>`;
  svg += `<text x="${lx + 14}" y="${ly + 7}" font-size="8" fill="#b8c7d8" font-family='"Space Grotesk",sans-serif'>Microphone</text>`;
  // Vib sensor legend row
  svg += `<rect x="${lx + 1}" y="${ly + 16}" width="9" height="9" fill="#d97706" rx="1"/>`;
  svg += `<text x="${lx + 14}" y="${ly + 23}" font-size="8" fill="#b8c7d8" font-family='"Space Grotesk",sans-serif'>Vibration sensor</text>`;
  // GT legend row
  svg +=
    `<polygon points="${lx + 5},${ly + 30} ${lx + 10},${ly + 35} ${lx + 5},${ly + 40} ${lx},${ly + 35}" ` +
    `fill="#dc2626"/>`;
  svg += `<text x="${lx + 14}" y="${ly + 37}" font-size="8" fill="#b8c7d8" font-family='"Space Grotesk",sans-serif'>Ground truth</text>`;
  // Best-estimate legend row
  svg += `<line x1="${lx}" y1="${ly + 49}" x2="${lx + 10}" y2="${ly + 49}" stroke="#16a34a" stroke-width="2.5"/>`;
  svg += `<line x1="${lx + 5}" y1="${ly + 44}" x2="${lx + 5}" y2="${ly + 54}" stroke="#16a34a" stroke-width="2.5"/>`;
  svg += `<text x="${lx + 14}" y="${ly + 52}" font-size="8" fill="#b8c7d8" font-family='"Space Grotesk",sans-serif'>Best estimate</text>`;

  svg += "</svg>";
  container.innerHTML = svg;
}

async function fetchOperatorAlerts() {
  try {
    const data = await apiFetch("/api/alerts");
    const alerts = data.alerts || data || [];
    renderAlertFeed(alerts);
  } catch {
    dom.alertFeed.innerHTML = '<div class="chart-placeholder">No alerts</div>';
  }
}

function renderAlertFeed(alerts) {
  if (!alerts || alerts.length === 0) {
    dom.alertFeed.innerHTML = '<div class="chart-placeholder">No alerts</div>';
    return;
  }

  // Show newest first
  const sorted = [...alerts].reverse();

  let html = "";
  for (const alert of sorted) {
    const level = alert.level || alert.severity || "INFO";
    const time = alert.time || alert.timestamp || "";
    const source = alert.source || "";
    const message = alert.message || alert.msg || "";
    const details = alert.details || {};
    const direction = details.direction || null;
    const timeStr =
      typeof time === "number"
        ? new Date(time * 1000).toLocaleTimeString()
        : String(time);

    // Build optional tags
    let tagsHtml = "";
    if (source) {
      tagsHtml += `<span class="alert-source-badge">${escapeHtml(source)}</span>`;
    }
    if (direction) {
      const dirLabel = direction.charAt(0).toUpperCase() + direction.slice(1);
      tagsHtml += `<span class="alert-direction-badge">${escapeHtml(dirLabel)}</span>`;
    }

    html += `<div class="alert-feed-item">
            <span class="alert-severity-badge alert-severity-${escapeHtml(level)}">${escapeHtml(level)}</span>
            <span class="alert-feed-time">${escapeHtml(timeStr)}</span>
            ${tagsHtml}
            <span class="alert-feed-message">${escapeHtml(message)}</span>
        </div>`;
  }

  dom.alertFeed.innerHTML = html;
}

// ============================================================
// API Functions (Expert Tools)
// ============================================================

async function checkHealth() {
  try {
    const data = await apiFetch("/api/health");
    dom.healthDot.className = "health-dot health-ok";
    dom.healthText.textContent = data.status || "OK";
  } catch {
    dom.healthDot.className = "health-dot health-error";
    dom.healthText.textContent = "Disconnected";
  }
}

async function fetchSegments() {
  try {
    const data = await apiFetch("/api/segments");
    segments = data.segments || data || [];
    dom.segmentSelector.innerHTML = "";

    if (segments.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "-- No segments --";
      dom.segmentSelector.appendChild(opt);
      return;
    }

    // Separate into healthy (nominal) and fault-position groups
    const healthy = segments.filter((s) => s.is_healthy);
    const faults = segments.filter((s) => !s.is_healthy);

    function makeOpt(seg, i) {
      const opt = document.createElement("option");
      opt.value = seg.id !== undefined ? seg.id : `seg_${i}`;
      const state = seg.state_code || "??";
      const dur = seg.duration_s ? `${seg.duration_s}s` : "";
      const label = seg.folder || opt.value;
      opt.textContent = `${label} [${state}] ${dur}`;
      return opt;
    }

    if (healthy.length > 0) {
      const group = document.createElement("optgroup");
      group.label = "Healthy (Nominal)";
      healthy.forEach((seg, i) => group.appendChild(makeOpt(seg, i)));
      dom.segmentSelector.appendChild(group);
    }

    if (faults.length > 0) {
      const group = document.createElement("optgroup");
      group.label = "Fault Positions";
      faults.forEach((seg, i) =>
        group.appendChild(makeOpt(seg, healthy.length + i)),
      );
      dom.segmentSelector.appendChild(group);
    }

    // Auto-select first segment
    dom.segmentSelector.selectedIndex = 0;
    const firstId = dom.segmentSelector.value;
    if (firstId !== "") {
      await onSegmentChange(firstId);
    }
  } catch (err) {
    showToast(`Failed to fetch segments: ${err.message}`);
  }
}

async function fetchDetection(segId) {
  try {
    const det = await apiFetch(`/api/segments/${segId}/detection`);

    const state = det.state_code || "unknown";
    const stateColor = STATE_COLORS[state] || STATE_COLORS.ST;
    const stateLabel = STATE_LABELS[state] || state;
    const score = det.anomaly_score !== undefined ? det.anomaly_score : 0;
    const votes = det.detector_votes || {};
    const isTrained = det.trained !== false;
    const isAnomaly = det.is_anomaly || false;
    const stateConf =
      det.state_confidence !== undefined ? det.state_confidence : 0;

    // Determine score bar color
    let scoreClass = "score-normal";
    if (score > 0.7) scoreClass = "score-anomaly";
    else if (score > 0.4) scoreClass = "score-warning";
    const scorePct = Math.min(Math.max(score * 100, 0), 100).toFixed(1);

    // Anomaly indicator
    const anomalyClass = isAnomaly
      ? "anomaly-indicator-red"
      : "anomaly-indicator-green";
    const anomalyLabel = isAnomaly ? "ANOMALY DETECTED" : "Normal";

    // Build votes chips
    let votesHtml = "";
    for (const [detector, verdict] of Object.entries(votes)) {
      const flagged = verdict === true || verdict === 1;
      const chipClass = flagged ? "vote-anomaly" : "vote-normal";
      const chipLabel = flagged ? "anomaly" : "normal";
      votesHtml += `<span class="vote-chip ${chipClass}">${escapeHtml(detector)}: ${chipLabel}</span>`;
    }

    // State confidence
    const stateConfPct = (stateConf * 100).toFixed(0);

    // Untrained notice
    const noticeHtml = !isTrained
      ? `<div class="detection-notice">Models not trained. Click "Train Models" for full detection.</div>`
      : "";

    // Mode Clustering section
    let clusteringHtml = "";
    const clustering = det.clustering;
    if (clustering) {
      const clState = clustering.state_code || "unknown";
      const clColor = STATE_COLORS[clState] || STATE_COLORS.ST;
      const clLabel = STATE_LABELS[clState] || clState;
      const clConf =
        clustering.confidence !== undefined
          ? (clustering.confidence * 100).toFixed(0)
          : "?";
      const corridorOk = clustering.within_corridor;
      const corridorClass = corridorOk ? "corridor-ok" : "corridor-warning";
      const corridorDev =
        clustering.corridor_deviation !== undefined
          ? clustering.corridor_deviation
          : 0;
      const corridorLabel = corridorOk
        ? "Within corridor"
        : `Outside corridor (${corridorDev.toFixed(1)}\u03C3)`;

      clusteringHtml = `
                <div class="clustering-section">
                    <h3>Mode Clustering</h3>
                    <div class="detection-state">
                        <span class="state-badge" style="background:${clColor}">${escapeHtml(clState)}</span>
                        <span class="state-label">${escapeHtml(clLabel)}</span>
                        <span class="state-conf">${clConf}%</span>
                    </div>
                    <div class="${corridorClass}">${escapeHtml(corridorLabel)}</div>
                </div>
            `;
    }

    dom.detectionContent.innerHTML = `
            ${noticeHtml}
            <div class="anomaly-indicator ${anomalyClass}">${escapeHtml(anomalyLabel)}</div>
            <div class="detection-state">
                <span class="state-badge" style="background:${stateColor}">${escapeHtml(state)}</span>
                <span class="state-label">${escapeHtml(stateLabel)}</span>
                ${stateConf > 0 ? `<span class="state-conf">${stateConfPct}%</span>` : ""}
            </div>
            <div class="score-section">
                <div class="score-label">
                    <span>Anomaly Score</span>
                    <span>${scorePct}%</span>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar ${scoreClass}" style="width:${scorePct}%"></div>
                </div>
            </div>
            ${votesHtml ? `<div class="votes-section"><h3>Detector Votes</h3><div class="votes-chips">${votesHtml}</div></div>` : ""}
            ${clusteringHtml}
        `;

    // Render localization
    renderLocalization(det);

    // Localization tab — table
    if (det.localization && det.localization.methods) {
      renderLocalizationTable(dom.locTableContent, det.localization);
    }

    // Refresh alerts (expert table)
    await fetchAlerts();

    // Fire-and-forget refresh of operator overview
    fetchOverview();
  } catch (err) {
    dom.detectionContent.innerHTML = `<div class="chart-placeholder">Error loading detection: ${escapeHtml(err.message)}</div>`;
  }
}

// ============================================================
// Bench-top 3-D Localization Visualization
// ============================================================

const LOC_METHOD_COLORS = {
  neural_cnn: "#16a34a",
  srp_phat: "#7c3aed",
  tdoa_triangulation: "#ea580c",
  fused: "#0891b2",
};

const LOC_METHOD_LABELS = {
  neural_cnn: "Neural (LocalizationCNNS2)",
  srp_phat: "SRP-PHAT",
  tdoa_triangulation: "TDOA triangulation",
  fused: "Fused",
};

function renderLocalization(det) {
  const loc = det.localization;
  if (!loc || !loc.methods || Object.keys(loc.methods).length === 0) {
    dom.localizationContent.innerHTML =
      '<div class="chart-placeholder">No localization data</div>';
    return;
  }
  renderBenchTop3D(dom.localizationContent, loc);
}

function renderBenchTop3D(container, loc) {
  // Ensure a Plotly div + summary table are present
  if (!container.querySelector(".loc-plotly-div")) {
    container.innerHTML =
      '<div class="loc-plotly-div" style="height:340px"></div>' +
      '<div class="loc-summary"></div>';
  }
  const plotDiv = container.querySelector(".loc-plotly-div");
  const summaryEl = container.querySelector(".loc-summary");

  const traces = [];
  const methods = loc.methods || {};
  const gt = loc.ground_truth_cm || null;
  const micPos = loc.mic_positions_cm || [];
  const vibPos = loc.vib_positions_cm || [];
  const allFaults = loc.all_fault_positions_cm || {};
  const bestMethod = loc.best_method || "";
  const micNames = loc.mic_names || micPos.map((_, i) => `mic_${i}`);
  const vibNames = loc.vib_names || vibPos.map((_, i) => `vib_${i}`);

  // Microphone positions (blue circles)
  if (micPos.length > 0) {
    traces.push({
      type: "scatter3d",
      mode: "markers+text",
      x: micPos.map((p) => p[0]),
      y: micPos.map((p) => p[1]),
      z: micPos.map((p) => p[2]),
      text: micNames,
      textposition: "top center",
      marker: {
        size: 7,
        color: "#0284c7",
        symbol: "circle",
        line: { color: "#ffffff", width: 1 },
      },
      name: "Microphones",
    });
  }

  // Vibration sensor positions (orange squares)
  if (vibPos.length > 0) {
    traces.push({
      type: "scatter3d",
      mode: "markers+text",
      x: vibPos.map((p) => p[0]),
      y: vibPos.map((p) => p[1]),
      z: vibPos.map((p) => p[2]),
      text: vibNames,
      textposition: "top center",
      marker: {
        size: 6,
        color: "#d97706",
        symbol: "square",
        line: { color: "#ffffff", width: 1 },
      },
      name: "Vibration sensors",
    });
  }

  // All known fault positions (gray diamonds)
  const fpKeys = Object.keys(allFaults);
  if (fpKeys.length > 0) {
    const fps = fpKeys.map((k) => allFaults[k]);
    traces.push({
      type: "scatter3d",
      mode: "markers",
      x: fps.map((p) => p[0]),
      y: fps.map((p) => p[1]),
      z: fps.map((p) => p[2]),
      hovertext: fpKeys,
      hoverinfo: "text",
      marker: { size: 5, color: "#94a3b8", symbol: "diamond", opacity: 0.6 },
      name: "Known positions",
    });
  }

  // Ground truth (solid red diamond)
  if (gt) {
    traces.push({
      type: "scatter3d",
      mode: "markers+text",
      x: [gt[0]],
      y: [gt[1]],
      z: [gt[2]],
      text: ["Ground truth"],
      textposition: "top right",
      marker: {
        size: 12,
        color: "#dc2626",
        symbol: "diamond",
        line: { color: "#ffffff", width: 2 },
      },
      name: "Ground truth",
    });
  }

  // Per-method estimates
  for (const [key, m] of Object.entries(methods)) {
    const est = m.estimated_cm;
    if (!est || est.length < 3) continue;
    const isBest = key === bestMethod;
    const color = LOC_METHOD_COLORS[key] || "#6b7280";
    const label = LOC_METHOD_LABELS[key] || key;
    const errStr = m.error_cm != null ? ` (${m.error_cm.toFixed(1)} cm)` : "";
    traces.push({
      type: "scatter3d",
      mode: "markers+text",
      x: [est[0]],
      y: [est[1]],
      z: [est[2]],
      text: [`${label}${errStr}`],
      textposition: "middle right",
      marker: {
        size: isBest ? 11 : 7,
        color: color,
        symbol: isBest ? "cross" : "circle",
        line: { color: "#ffffff", width: isBest ? 2 : 1 },
      },
      name: label + (isBest ? " \u2605" : ""),
    });
  }

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    font: {
      color: "#b8c7d8",
      size: 10,
      family: '"Space Grotesk", "Segoe UI", sans-serif',
    },
    margin: { t: 30, r: 10, b: 10, l: 10 },
    scene: {
      xaxis: {
        title: "X (cm)",
        range: [-5, 46],
        gridcolor: "#2a3e51",
        zerolinecolor: "#2a3e51",
        color: "#7f94aa",
      },
      yaxis: {
        title: "Y (cm)",
        range: [-5, 46],
        gridcolor: "#2a3e51",
        zerolinecolor: "#2a3e51",
        color: "#7f94aa",
      },
      zaxis: {
        title: "Z (cm)",
        range: [-2, 45],
        gridcolor: "#2a3e51",
        zerolinecolor: "#2a3e51",
        color: "#7f94aa",
      },
      bgcolor: "#0d1a24",
      camera: { eye: { x: 1.7, y: -1.7, z: 1.2 } },
    },
    legend: {
      orientation: "h",
      y: -0.04,
      x: 0.5,
      xanchor: "center",
      font: { size: 9, color: "#b8c7d8" },
      bgcolor: "rgba(0,0,0,0)",
    },
    showlegend: true,
    title: {
      text: "Bench-top 3-D Localization",
      font: { size: 12, color: "#7f94aa" },
    },
  };

  Plotly.react(plotDiv, traces, layout, PLOTLY_CONFIG);

  // Summary table
  if (summaryEl) {
    const order = ["neural_cnn", "srp_phat", "tdoa_triangulation", "fused"];
    let rows = "";
    for (const key of order) {
      const m = methods[key];
      if (!m) continue;
      const isBest = key === bestMethod;
      const est = m.estimated_cm;
      const estStr = est
        ? `(${est[0].toFixed(1)}, ${est[1].toFixed(1)}, ${est[2].toFixed(1)})`
        : "";
      const errStr = m.error_cm != null ? `${m.error_cm.toFixed(2)} cm` : "N/A";
      const star = isBest
        ? ' <span style="color:#16a34a;font-weight:700">\u2605</span>'
        : "";
      rows +=
        `<tr><td>${escapeHtml(LOC_METHOD_LABELS[key] || key)}${star}</td>` +
        `<td>${escapeHtml(estStr)}</td><td>${escapeHtml(errStr)}</td></tr>`;
    }
    summaryEl.innerHTML =
      `<table class="features-table" style="margin-top:8px">` +
      `<thead><tr><th>Method</th><th>Estimate (x,y,z) cm</th><th>Error</th></tr></thead>` +
      `<tbody>${rows}</tbody></table>`;
  }
}

async function trainModels() {
  showProgress("Training detection models...");
  setStatus("Training detection models...", true);
  setButtonLoading(dom.btnTrain, true);
  if (dom.btnTrainDet) setButtonLoading(dom.btnTrainDet, true);
  dom.trainStatus.className = "train-badge badge-training";
  dom.trainStatus.textContent = "Training...";
  if (dom.trainStatusDet) {
    dom.trainStatusDet.className = "train-badge badge-training";
    dom.trainStatusDet.textContent = "Training...";
  }

  try {
    const result = await apiFetch("/api/train", {
      method: "POST",
      body: JSON.stringify({}),
    });

    const nSegs = result.n_segments || 0;
    dom.trainStatus.className = "train-badge badge-trained";
    dom.trainStatus.textContent = `Trained (${nSegs} segs)`;
    if (dom.trainStatusDet) {
      dom.trainStatusDet.className = "train-badge badge-trained";
      dom.trainStatusDet.textContent = `Trained (${nSegs} segs)`;
    }
    showToast(`Models trained on ${nSegs} segments`, "success");

    // Auto-detect all segments
    showProgress("Detecting anomalies in all segments...");
    setStatus("Detecting anomalies...", true);
    await apiFetch("/api/detect-all", { method: "POST" });

    // Refresh all operator panels
    showProgress("Refreshing dashboard...");
    await Promise.all([
      fetchOverview(),
      fetchTimeline(),
      fetchOperatorAlerts(),
    ]);

    // Refresh vane wheel and machine diagram with last segment
    if (segments.length > 0) {
      const lastSeg = segments[segments.length - 1];
      const lastId = lastSeg.id || `seg_${segments.length - 1}`;
      await Promise.all([
        fetchAnomalyRegion(lastId),
        fetchMachineDiagram(lastId),
      ]);
    }

    // Re-fetch detection for current segment in expert tools
    if (currentSegmentId) {
      await fetchDetection(currentSegmentId);
    }

    setStatus("Detection complete");
  } catch (err) {
    dom.trainStatus.className = "train-badge badge-failed";
    dom.trainStatus.textContent = "Failed";
    if (dom.trainStatusDet) {
      dom.trainStatusDet.className = "train-badge badge-failed";
      dom.trainStatusDet.textContent = "Failed";
    }
    showToast(`Training failed: ${err.message}`);
    setStatus("Training failed");
  } finally {
    setButtonLoading(dom.btnTrain, false);
    if (dom.btnTrainDet) setButtonLoading(dom.btnTrainDet, false);
    hideProgress();
  }
}

async function checkTrainStatus() {
  try {
    const data = await apiFetch("/api/train/status");
    const setAll = (cls, txt) => {
      dom.trainStatus.className = cls;
      dom.trainStatus.textContent = txt;
      if (dom.trainStatusDet) {
        dom.trainStatusDet.className = cls;
        dom.trainStatusDet.textContent = txt;
      }
    };
    if (data.is_trained) {
      setAll("train-badge badge-trained", `Trained (${data.n_segments} segs)`);
    } else if (data.status === "training") {
      setAll("train-badge badge-training", "Training...");
    } else if (data.status === "failed") {
      setAll("train-badge badge-failed", "Failed");
    }
  } catch {
    // Server not available — leave as not trained
  }
}

async function fetchAlerts() {
  try {
    const data = await apiFetch("/api/alerts");
    const alerts = data.alerts || data || [];

    if (alerts.length === 0) {
      dom.alertsTbody.innerHTML =
        '<tr class="empty-row"><td colspan="4">No alerts</td></tr>';
      return;
    }

    let html = "";
    for (const alert of alerts) {
      const level = alert.level || alert.severity || "INFO";
      const time = alert.time || alert.timestamp || "";
      const source = alert.source || alert.component || "";
      const message = alert.message || alert.msg || "";
      html += `<tr class="alert-row-${escapeHtml(level)}">
                <td class="alert-time">${escapeHtml(String(time))}</td>
                <td><span class="alert-level alert-level-${escapeHtml(level)}">${escapeHtml(level)}</span></td>
                <td class="alert-source">${escapeHtml(source)}</td>
                <td>${escapeHtml(message)}</td>
            </tr>`;
    }
    dom.alertsTbody.innerHTML = html;

    // Auto-scroll to bottom
    const container = document.getElementById("alerts-content");
    container.scrollTop = container.scrollHeight;

    // Also refresh the operator alert feed
    renderAlertFeed(alerts);
  } catch (err) {
    showToast(`Failed to fetch alerts: ${err.message}`);
  }
}

// ============================================================
// Event Handlers
// ============================================================

async function onSegmentChange(segId) {
  if (!segId) return;
  currentSegmentId = segId;
  setStatus("Loading segment data...", true);

  // Update detection tab label
  if (dom.detSegLabel) dom.detSegLabel.textContent = segId;

  // Re-render timeline highlight without re-fetching
  if (lastTimelineData) {
    renderTimeline(lastTimelineData);
  }

  try {
    await Promise.all([
      fetchDetection(segId),
      fetchAnomalyRegion(segId),
      fetchMachineDiagram(segId),
    ]);
    setStatus("Ready");
  } catch (err) {
    setStatus("Error loading segment");
  }
}

// ============================================================
// Initialization
// ============================================================

// --- Tab switching ---------------------------------------------------------

function switchTab(tabName) {
  currentTab = tabName;
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-page").forEach((page) => {
    page.classList.toggle("tab-active", page.id === `tab-${tabName}`);
  });
  // Lazy-load tab content on first switch
  if (tabName === "detection" && dom.anomalyScoresChart) {
    renderAnomalyScoresChart();
  }
  if (tabName === "reports") {
    renderReports();
  }
  if (tabName === "detection" || tabName === "reports") {
    checkModelsStatus();
  }
}

// --- Models status ---------------------------------------------------------

async function checkModelsStatus() {
  if (!dom.modelsStatusGrid) return;
  try {
    const status = await apiFetch("/api/models-status");
    const entries = Object.entries(status);
    if (entries.length === 0) {
      dom.modelsStatusGrid.innerHTML =
        '<div class="placeholder-text">No model data</div>';
      return;
    }
    let html = "";
    for (const [key, info] of entries) {
      const ok = info.trained;
      const sizeStr = ok && info.size_kb ? `${info.size_kb} KB` : "";
      const stateClass = ok ? "ok" : "missing";
      const stateLabel = ok
        ? `✓ Trained${sizeStr ? " · " + sizeStr : ""}`
        : "⚠ Not found";
      html += `<div class="model-item">
          <div class="model-name">${escapeHtml(info.display)}</div>
          <div class="model-state ${stateClass}">${escapeHtml(stateLabel)}</div>
        </div>`;
    }
    dom.modelsStatusGrid.innerHTML = html;
  } catch {
    dom.modelsStatusGrid.innerHTML =
      '<div class="placeholder-text">Failed to load model status</div>';
  }
}

// --- Anomaly scores chart --------------------------------------------------

async function renderAnomalyScoresChart() {
  if (!dom.anomalyScoresChart) return;
  try {
    const data = await apiFetch("/api/reports");
    const rows = data.rows || [];
    if (rows.length === 0) {
      dom.anomalyScoresChart.innerHTML =
        '<div class="placeholder-text">No report data</div>';
      return;
    }

    const labels = rows.map((r) => r.folder || r.id);
    const errors = rows.map((r) => r.best_error_cm);

    const trace = {
      type: "bar",
      x: labels,
      y: errors,
      marker: {
        color: errors.map((e) =>
          e == null
            ? "#4a6070"
            : e < 5
              ? "#2fd492"
              : e < 20
                ? "#f6bf3a"
                : "#ff6673",
        ),
        line: { color: "#2a3e51", width: 1 },
      },
      text: errors.map((e) => (e != null ? `${e.toFixed(1)} cm` : "N/A")),
      textposition: "outside",
      name: "Best Loc. Error",
    };
    const layout = {
      ...PLOTLY_LAYOUT_BASE,
      title: {
        text: "Best Localization Error per Segment",
        font: { size: 13, color: "#b8c7d8" },
      },
      xaxis: {
        ...PLOTLY_LAYOUT_BASE.xaxis,
        title: "Fault Position",
        tickangle: -22,
      },
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: "Error (cm)" },
    };
    Plotly.react(dom.anomalyScoresChart, [trace], layout, PLOTLY_CONFIG);
  } catch {
    dom.anomalyScoresChart.innerHTML =
      '<div class="placeholder-text">Failed to load scores</div>';
  }
}

// --- Localization results table in localization tab -----------------------

function renderLocalizationTable(container, loc) {
  if (!container) return;
  const methods = loc.methods || {};
  const keys = ["neural_cnn", "srp_phat", "tdoa_triangulation", "fused"];
  const bestMethod = loc.best_method || "";

  const hasAny = keys.some((k) => methods[k]);
  if (!hasAny) {
    container.innerHTML =
      '<div class="placeholder-text">No localization results</div>';
    return;
  }

  let rows = "";
  for (const key of keys) {
    const m = methods[key];
    if (!m) continue;
    const isBest = key === bestMethod;
    const est = m.estimated_cm;
    const estStr = est
      ? `(${est[0].toFixed(1)}, ${est[1].toFixed(1)}, ${est[2].toFixed(1)})`
      : "N/A";
    const errStr = m.error_cm != null ? `${m.error_cm.toFixed(2)} cm` : "N/A";
    const bestMark = isBest ? ' <span class="best-mark">★ BEST</span>' : "";
    rows += `<tr>
      <td>${escapeHtml(LOC_METHOD_LABELS[key] || key)}${bestMark}</td>
      <td class="loc-mono">${escapeHtml(estStr)}</td>
      <td class="${m.error_cm != null && m.error_cm < 5 ? "err-good" : m.error_cm != null && m.error_cm < 20 ? "err-warn" : "err-bad"}">${escapeHtml(errStr)}</td>
    </tr>`;
  }

  const gt = loc.ground_truth_cm;
  const gtStr = gt
    ? `(${gt[0].toFixed(1)}, ${gt[1].toFixed(1)}, ${gt[2].toFixed(1)})`
    : "N/A";

  container.innerHTML = `
    <div class="loc-gt-row">Ground truth: <span class="loc-mono">${escapeHtml(gtStr)}</span></div>
    <table class="loc-results-table">
      <thead><tr><th>Method</th><th>Estimate (x, y, z) cm</th><th>Error</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// --- Reports ---------------------------------------------------------------

async function renderReports() {
  if (!dom.reportsErrorChart || !dom.reportsTableContent) return;
  try {
    const data = await apiFetch("/api/reports");
    const rows = data.rows || [];

    if (rows.length === 0) {
      dom.reportsErrorChart.innerHTML =
        '<div class="placeholder-text">No report data</div>';
      dom.reportsTableContent.innerHTML =
        '<div class="placeholder-text">No data</div>';
      return;
    }

    // Build grouped bar chart — one bar group per fault position,
    // one bar per method.
    const methods = ["neural_cnn", "srp_phat", "tdoa_triangulation", "fused"];
    const methodColors = {
      neural_cnn: "#2fd492",
      srp_phat: "#a78bfa",
      tdoa_triangulation: "#fb923c",
      fused: "#38bdf8",
    };
    const labels = rows.map((r) => r.folder || r.id);
    const traces = methods.map((mKey) => ({
      type: "bar",
      name: LOC_METHOD_LABELS[mKey] || mKey,
      x: labels,
      y: rows.map((r) => r.methods[mKey]?.error_cm ?? null),
      marker: { color: methodColors[mKey] || "#6b8099" },
    }));
    const layout = {
      ...PLOTLY_LAYOUT_BASE,
      barmode: "group",
      title: {
        text: "Localization Error by Method and Fault Position",
        font: { size: 13, color: "#b8c7d8" },
      },
      xaxis: {
        ...PLOTLY_LAYOUT_BASE.xaxis,
        title: "Fault Position",
        tickangle: -22,
      },
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: "Error (cm)" },
      legend: {
        orientation: "h",
        y: -0.25,
        font: { color: "#b8c7d8", size: 10 },
        bgcolor: "rgba(0,0,0,0)",
      },
    };
    Plotly.react(dom.reportsErrorChart, traces, layout, PLOTLY_CONFIG);

    // Build summary table
    let tableRows = "";
    for (const r of rows) {
      const gt = r.ground_truth_cm;
      const gtStr = gt ? `(${gt[0]}, ${gt[1]}, ${gt[2]})` : "N/A";
      const best =
        r.best_error_cm != null ? `${r.best_error_cm.toFixed(2)} cm` : "N/A";
      const bestMeth =
        LOC_METHOD_LABELS[r.best_method] || r.best_method || "N/A";
      tableRows += `<tr>
        <td class="loc-mono">${escapeHtml(r.folder || r.id)}</td>
        <td class="loc-mono">${escapeHtml(gtStr)}</td>
        <td>${escapeHtml(bestMeth)}</td>
        <td class="${r.best_error_cm != null && r.best_error_cm < 5 ? "err-good" : r.best_error_cm != null && r.best_error_cm < 20 ? "err-warn" : "err-bad"}">${escapeHtml(best)}</td>
      </tr>`;
    }
    dom.reportsTableContent.innerHTML = `
      <table class="loc-results-table">
        <thead><tr><th>Folder</th><th>Ground Truth</th><th>Best Method</th><th>Best Error</th></tr></thead>
        <tbody>${tableRows}</tbody>
      </table>`;
  } catch (err) {
    dom.reportsErrorChart.innerHTML = `<div class="placeholder-text">Error loading report: ${escapeHtml(err.message)}</div>`;
  }
}

async function init() {
  // Tab switching
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // Wire up event listeners
  dom.btnTrain.addEventListener("click", trainModels);
  if (dom.btnTrainDet) dom.btnTrainDet.addEventListener("click", trainModels);
  if (dom.btnRefreshModels)
    dom.btnRefreshModels.addEventListener("click", checkModelsStatus);

  dom.segmentSelector.addEventListener("change", (e) => {
    onSegmentChange(e.target.value);
  });

  dom.btnRefreshAlerts.addEventListener("click", fetchAlerts);
  if (dom.btnRefreshReports)
    dom.btnRefreshReports.addEventListener("click", renderReports);

  // Operator dashboard: fetch overview + timeline on load
  await fetchOverview();
  await fetchTimeline();
  await fetchOperatorAlerts();

  // Auto-refresh overview every 30 seconds
  overviewRefreshInterval = setInterval(fetchOverview, 30000);

  // Check health and training status
  await checkHealth();
  await checkTrainStatus();

  // Periodically check health
  setInterval(checkHealth, 30000);

  // Pre-load models status (visible on detection tab)
  checkModelsStatus();

  // Try to load segments if available
  try {
    await fetchSegments();
  } catch {
    // No segments yet -- that is fine
  }
}

document.addEventListener("DOMContentLoaded", init);
