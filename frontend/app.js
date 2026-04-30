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
// Illwerke tab state
let illwerkeLoaded = false;
let illwerkeSelectedDay = null;

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
  SP1: "#0891b2",
  SP2: "#7c3aed",
  SP3: "#059669",
  unknown: "#adb5bd",
};

const STATE_LABELS = {
  ST: "Standstill",
  TU: "Turbine",
  PU: "Pump",
  PH: "Phase Shifter",
  RF: "Random Fault",
  SP1: "Speed 1",
  SP2: "Speed 2",
  SP3: "Speed 3",
  unknown: "Unknown",
};

// Illwerke operating mode display
const IW_MODE_COLORS = {
  Standstill: "#6c757d",
  Turbine: "#4fb8ff",
  Phasenschieber: "#fd7e14",
  Pump: "#2fd492",
  Transitioning: "#9b59b6",
};
const IW_MODE_LABELS = {
  Standstill: "Standstill",
  Turbine: "Turbine",
  Phasenschieber: "Phasenschieber",
  Pump: "Pump",
  Transitioning: "Transitioning",
};
const IW_MODEL_DISPLAY = {
  cnf: "CNF — Normalizing Flow",
  ocsvm_anomaly: "OC-SVM (default \u03bd)",
  ocsvm_anomaly_nu_001: "OC-SVM (\u03bd\u202f=\u202f0.01)",
  ocsvm_anomaly_nu_003: "OC-SVM (\u03bd\u202f=\u202f0.03)",
  ocsvm_anomaly_nu_01: "OC-SVM (\u03bd\u202f=\u202f0.10)",
  lstm_ae: "LSTM Autoencoder",
  cnn_ae: "CNN Autoencoder",
};
// Mode-classifier backbone display names
const IW_MODE_SOURCE_DISPLAY = {
  pipeline: "Physics Pipeline (L4)",
  kmeans: "K-Means (reference)",
  cnf: "CNF",
  ocsvm: "OC-SVM",
  lstm_ae: "LSTM-AE",
  cnn_ae: "CNN-AE",
  sota: "ResML\u202f+\u202fViterbi (SOTA)",
  dtss: "DTSS (Unsupervised)",
};

// Severity colour map for pipeline anomaly events
const IW_SEVERITY_COLORS = {
  alert: "#ff6673",
  watch: "#f6bf3a",
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
  // Illwerke tab
  btnDtssTrain: document.getElementById("btn-dtss-train"),
  dtssTrainStatus: document.getElementById("dtss-train-status"),
  dtssStatusContent: document.getElementById("dtss-status-content"),
  iwKpiDays: document.getElementById("iw-kpi-days"),
  iwKpiDaysRange: document.getElementById("iw-kpi-days-range"),
  iwKpiWindows: document.getElementById("iw-kpi-windows"),
  iwKpiWindowsSub: document.getElementById("iw-kpi-windows-sub"),
  iwKpiTurbine: document.getElementById("iw-kpi-turbine"),
  iwKpiPH: document.getElementById("iw-kpi-ph"),
  iwKpiWatch: document.getElementById("iw-kpi-watch"),
  iwKpiAlert: document.getElementById("iw-kpi-alert"),
  iwKpiAnom: document.getElementById("iw-kpi-anom"),
  iwDaySelect: document.getElementById("iw-day-select"),
  iwGanttModelSelect: document.getElementById("iw-gantt-model"),
  iwGanttModeSource: document.getElementById("iw-gantt-mode-source"),
  iwGanttSubtitle: document.getElementById("iw-gantt-subtitle"),
  iwGantt: document.getElementById("iw-gantt"),
  iwProcessChart: document.getElementById("iw-process-chart"),
  iwProcVarsChart: document.getElementById("iw-proc-vars-chart"),
  iwModeDonut: document.getElementById("iw-mode-donut"),
  iwScoreTimeline: document.getElementById("iw-score-timeline"),
  iwModelsTable: document.getElementById("iw-models-table"),
  iwEventsChart: document.getElementById("iw-events-chart"),
  iwEventsTable: document.getElementById("iw-events-table"),
  iwEventsSeverity: document.getElementById("iw-events-severity"),
  iwTransitionsTable: document.getElementById("iw-transitions-table"),
  iwTransitionsSummary: document.getElementById("iw-transitions-summary"),
  iwOraclePanel: document.getElementById("iw-oracle-panel"),
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

    // Group into 4 optgroups: S2-Healthy, S2-Fault, S3-Healthy, S3-Fault
    const s2Healthy = segments.filter(
      (s) => s.is_healthy && s.dataset !== "third",
    );
    const s2Faults = segments.filter(
      (s) => !s.is_healthy && s.dataset !== "third",
    );
    const s3Healthy = segments.filter(
      (s) => s.is_healthy && s.dataset === "third",
    );
    const s3Faults = segments.filter(
      (s) => !s.is_healthy && s.dataset === "third",
    );

    function makeOpt(seg, i) {
      const opt = document.createElement("option");
      opt.value = seg.id !== undefined ? seg.id : `seg_${i}`;
      const state = seg.state_code || "??";
      const dur = seg.duration_s ? `${seg.duration_s}s` : "";
      const label = seg.folder || opt.value;
      opt.textContent = `${label} [${state}] ${dur}`;
      return opt;
    }

    function addGroup(label, items, offset) {
      if (items.length === 0) return;
      const group = document.createElement("optgroup");
      group.label = label;
      items.forEach((seg, i) => group.appendChild(makeOpt(seg, offset + i)));
      dom.segmentSelector.appendChild(group);
    }

    addGroup("S2 — Healthy (Nominal)", s2Healthy, 0);
    addGroup("S2 — Fault Positions", s2Faults, s2Healthy.length);
    addGroup(
      "S3 — Healthy (Nominal)",
      s3Healthy,
      s2Healthy.length + s2Faults.length,
    );
    addGroup(
      "S3 — Fault Positions",
      s3Faults,
      s2Healthy.length + s2Faults.length + s3Healthy.length,
    );

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
  neural_cnn_s3: "#15803d",
  neural_cnn_s2_zeroshot: "#65a30d",
};

const LOC_METHOD_LABELS = {
  neural_cnn: "Neural (LocalizationCNNS2)",
  srp_phat: "SRP-PHAT",
  tdoa_triangulation: "TDOA triangulation",
  fused: "Fused",
  neural_cnn_s3: "Neural S3 (LocalizationCNNS3)",
  neural_cnn_s2_zeroshot: "S2 Zero-shot (5-mic)",
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

  // Compute scene bounds dynamically from all known positions + margin
  const allPts = [
    ...micPos,
    ...vibPos,
    ...(gt ? [gt] : []),
    ...Object.values(allFaults),
    ...Object.values(methods)
      .map((m) => m.estimated_cm)
      .filter(Boolean),
  ];
  const margin = 3;
  function axisRange(idx) {
    if (allPts.length === 0) return [-5, 50];
    const vals = allPts.map((p) => p[idx]);
    return [Math.min(...vals) - margin, Math.max(...vals) + margin];
  }
  const xRange = axisRange(0);
  const yRange = axisRange(1);
  const zRange = axisRange(2);

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
        range: xRange,
        gridcolor: "#2a3e51",
        zerolinecolor: "#2a3e51",
        color: "#7f94aa",
      },
      yaxis: {
        title: "Y (cm)",
        range: yRange,
        gridcolor: "#2a3e51",
        zerolinecolor: "#2a3e51",
        color: "#7f94aa",
      },
      zaxis: {
        title: "Z (cm)",
        range: zRange,
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
    const order = [
      "neural_cnn",
      "neural_cnn_s3",
      "neural_cnn_s2_zeroshot",
      "srp_phat",
      "tdoa_triangulation",
      "fused",
    ];
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
  if (tabName === "illwerke") {
    loadIllwerkeTab();
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
  const keys = [
    "neural_cnn",
    "neural_cnn_s3",
    "neural_cnn_s2_zeroshot",
    "srp_phat",
    "tdoa_triangulation",
    "fused",
  ];
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
    const methods = [
      "neural_cnn",
      "neural_cnn_s3",
      "neural_cnn_s2_zeroshot",
      "srp_phat",
      "tdoa_triangulation",
      "fused",
    ];
    const methodColors = {
      neural_cnn: "#2fd492",
      neural_cnn_s3: "#15803d",
      neural_cnn_s2_zeroshot: "#65a30d",
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

// ============================================================
// Illwerke Real-Plant Campaign Dashboard
// ============================================================

// ============================================================
// DTSS — Unsupervised Mode Detection
// ============================================================

function renderDtssStatus(data) {
  const status = data.training_status || "idle";
  const exists = data.artifacts_exist;

  // Update badge
  if (dom.dtssTrainStatus) {
    const map = {
      idle: ["badge-not-trained", "Not Run"],
      training: ["badge-training", "Training…"],
      done: ["badge-trained", "Trained"],
      failed: ["badge-failed", "Failed"],
    };
    const [cls, txt] = map[status] || ["badge-not-trained", status];
    dom.dtssTrainStatus.className = `train-badge ${cls}`;
    dom.dtssTrainStatus.textContent = txt;
  }

  if (dom.btnDtssTrain) {
    dom.btnDtssTrain.disabled = status === "training";
  }

  if (!dom.dtssStatusContent) return;

  if (status === "training") {
    dom.dtssStatusContent.innerHTML =
      '<div class="placeholder-text"><span class="spinner" style="display:inline-block;width:12px;height:12px;border:2px solid #4fb8ff;border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px"></span>Training in progress — this takes ~45 min on CPU. You can refresh the page; training continues in the server background.</div>';
    return;
  }

  if (data.error) {
    dom.dtssStatusContent.innerHTML = `<div class="placeholder-text" style="color:#ff6673">Training failed: ${escapeHtml(data.error)}</div>`;
    return;
  }

  if (!exists) {
    dom.dtssStatusContent.innerHTML =
      '<div class="placeholder-text">No artifacts found. Click "&#9654; Run Training" to start the DTSS pipeline (~45 min on CPU). Outputs will appear in the Mode Timeline dropdown as "DTSS (Unsupervised)".</div>';
    return;
  }

  // Artifacts exist — render summary
  const v = data.validation || {};
  const hi = v.head_independence || {};
  const hiPass = hi.independent !== false;
  const clMap = data.cluster_to_label || {};
  const nSeg = data.n_segments || 0;

  let clRows = "";
  for (const [k, lbl] of Object.entries(clMap)) {
    const isTransient = String(lbl).startsWith("Transitioning");
    clRows += `<tr>
      <td>C${escapeHtml(k)}</td>
      <td><span class="state-badge" style="background:${isTransient ? "#9b59b6" : "#2a3e51"};font-size:0.75rem">${escapeHtml(lbl)}</span></td>
    </tr>`;
  }

  dom.dtssStatusContent.innerHTML = `
    <div class="dtss-summary-grid">
      <div class="dtss-metric">
        <div class="dtss-metric-label">Segments</div>
        <div class="dtss-metric-value">${nSeg.toLocaleString()}</div>
      </div>
      <div class="dtss-metric">
        <div class="dtss-metric-label">Micro-events</div>
        <div class="dtss-metric-value">${(v.n_micro_events ?? "--").toLocaleString()}</div>
      </div>
      <div class="dtss-metric">
        <div class="dtss-metric-label">Changepoints</div>
        <div class="dtss-metric-value">${(v.n_changepoints ?? "--").toLocaleString()}</div>
      </div>
      <div class="dtss-metric">
        <div class="dtss-metric-label">Head χ² test</div>
        <div class="dtss-metric-value" style="color:${hiPass ? "#2fd492" : "#ff6673"}">${hiPass ? "PASS" : "FAIL"} (p=${hi.p_value != null ? hi.p_value.toFixed(3) : "—"})</div>
      </div>
      <div class="dtss-metric">
        <div class="dtss-metric-label">Cavitation flagged</div>
        <div class="dtss-metric-value">${v.cavitation_flagged_pct != null ? v.cavitation_flagged_pct.toFixed(1) + "%" : "—"}</div>
      </div>
      <div class="dtss-metric">
        <div class="dtss-metric-label">PCA components</div>
        <div class="dtss-metric-value">${v.pca_n_components ?? "—"}</div>
      </div>
    </div>
    ${clRows ? `<table class="iw-models-table" style="margin-top:10px"><thead><tr><th>Cluster</th><th>Semantic Label</th></tr></thead><tbody>${clRows}</tbody></table>` : ""}
    <div style="margin-top:8px;font-size:0.78rem;color:var(--text-muted)">Select "DTSS (Unsupervised)" in the Mode Timeline source dropdown to visualise results.</div>`;
}

async function checkDtssStatus() {
  try {
    const data = await apiFetch("/api/illwerke/dtss/status");
    renderDtssStatus(data);
    // If training is running, poll every 30 s
    if (data.training_status === "training") {
      setTimeout(checkDtssStatus, 30000);
    }
  } catch {
    // Non-fatal — DTSS section just stays empty
  }
}

async function startDtssTraining() {
  if (!dom.btnDtssTrain) return;
  dom.btnDtssTrain.disabled = true;
  if (dom.dtssTrainStatus) {
    dom.dtssTrainStatus.className = "train-badge badge-training";
    dom.dtssTrainStatus.textContent = "Starting…";
  }
  try {
    await apiFetch("/api/illwerke/dtss/train", { method: "POST" });
    showToast(
      "DTSS training started — check back in ~45 min.",
      "success",
      8000,
    );
    // Begin polling
    checkDtssStatus();
  } catch (err) {
    showToast(`Failed to start DTSS training: ${err.message}`);
    if (dom.btnDtssTrain) dom.btnDtssTrain.disabled = false;
    if (dom.dtssTrainStatus) {
      dom.dtssTrainStatus.className = "train-badge badge-failed";
      dom.dtssTrainStatus.textContent = "Failed";
    }
  }
}

async function loadIllwerkeTab() {
  if (illwerkeLoaded) return;
  illwerkeLoaded = true; // Prevent double-load; reset on error below

  try {
    setStatus("Loading Illwerke pipeline data…", true);

    // Primary data fetches — all in parallel
    const [
      overview,
      pipelineGantt,
      pipelineScores,
      pipelineEvents,
      pipelineTransitions,
      pipelineValidation,
      models,
    ] = await Promise.all([
      apiFetch("/api/illwerke/overview"),
      apiFetch("/api/illwerke/pipeline/gantt"),
      apiFetch("/api/illwerke/pipeline/scores?max_points=4000"),
      apiFetch("/api/illwerke/pipeline/events"),
      apiFetch("/api/illwerke/pipeline/transitions"),
      apiFetch("/api/illwerke/pipeline/validation"),
      apiFetch("/api/illwerke/models"),
    ]);

    // Also try to load legacy gantt for mode-source dropdown population
    let legacyGantt = null;
    try {
      legacyGantt = await apiFetch("/api/illwerke/gantt");
    } catch {
      /* non-fatal */
    }

    // ---- Populate anomaly-model overlay selector ----
    if (dom.iwGanttModelSelect) {
      dom.iwGanttModelSelect.innerHTML =
        '<option value="">\u2014 None \u2014</option>';
      for (const m of models.models || []) {
        if (m.status === "ok") {
          const opt = document.createElement("option");
          opt.value = m.key;
          opt.textContent = IW_MODEL_DISPLAY[m.key] || m.display;
          dom.iwGanttModelSelect.appendChild(opt);
        }
      }
    }

    // ---- Populate mode-source selector ----
    if (dom.iwGanttModeSource) {
      // Start from pipeline sources (always pipeline first)
      const pipelineSources = pipelineGantt.available_mode_sources || [
        "pipeline",
      ];
      const legacySources = legacyGantt?.available_mode_sources || [];
      const merged = [...new Set([...pipelineSources, ...legacySources])];

      dom.iwGanttModeSource.innerHTML = "";
      for (const src of merged) {
        const opt = document.createElement("option");
        opt.value = src;
        opt.textContent = IW_MODE_SOURCE_DISPLAY[src] || src;
        if (src === "pipeline") opt.selected = true;
        dom.iwGanttModeSource.appendChild(opt);
      }
    }

    // ---- Render all sections ----
    renderIllwerkeKPIs(overview, pipelineValidation);
    renderIllwerkeGantt(pipelineGantt);
    renderIllwerkeModeDonut(overview, pipelineValidation);
    renderIllwerkeAnomalyEvents(pipelineEvents);
    renderIllwerkeScoreTimeline(pipelineScores);
    renderIllwerkeTransitions(pipelineTransitions);
    renderIllwerkePhysicsOracle(pipelineValidation, overview);
    renderIllwerkeModels(models);
    populateIllwerkeDaySelect(pipelineGantt.days || []);

    if (pipelineGantt.days && pipelineGantt.days.length > 0) {
      illwerkeSelectedDay = pipelineGantt.days[0];
      if (dom.iwDaySelect) dom.iwDaySelect.value = illwerkeSelectedDay;
      loadIllwerkeDailyChart(illwerkeSelectedDay);
    }

    // Non-blocking: load DTSS status panel
    checkDtssStatus();
    setStatus("Ready");
  } catch (err) {
    illwerkeLoaded = false; // Allow retry
    setStatus("Ready");
    showToast(
      "Failed to load Illwerke data: " + escapeHtml(err.message),
      "error",
    );
  }
}

function populateIllwerkeDaySelect(days) {
  if (!dom.iwDaySelect) return;
  dom.iwDaySelect.innerHTML = "";
  for (const day of days) {
    const opt = document.createElement("option");
    opt.value = day;
    // Format: "Wed 15 Apr"
    const dt = new Date(day + "T12:00:00Z");
    opt.textContent = dt.toLocaleDateString("en-GB", {
      weekday: "short",
      day: "numeric",
      month: "short",
    });
    dom.iwDaySelect.appendChild(opt);
  }
}

// Shared helper: build Plotly layout.shapes for mode background bands.
// alphaOverride (optional float) replaces the per-mode alpha.
function _buildModeShapes(hours, modes, alphaOverride) {
  const alphas = {
    Standstill: alphaOverride ?? 0.12,
    Turbine: alphaOverride ?? 0.12,
    Phasenschieber: alphaOverride ?? 0.12,
    Pump: alphaOverride ?? 0.12,
    Transitioning: alphaOverride ?? 0.12,
    Unknown: alphaOverride ?? 0.08,
  };
  const baseColors = {
    Standstill: "108,117,125",
    Turbine: "79,184,255",
    Phasenschieber: "253,126,20",
    Pump: "47,212,146",
    Transitioning: "155,89,182",
    Unknown: "108,117,125",
  };
  const shapes = [];
  if (!hours.length || !modes.length) return shapes;
  const delta = hours.length > 1 ? hours[1] - hours[0] : 1;
  let segMode = modes[0];
  let segStart = hours[0];
  for (let i = 1; i <= modes.length; i++) {
    if (i === modes.length || modes[i] !== segMode) {
      const x1 = i < hours.length ? hours[i] : hours[hours.length - 1] + delta;
      const rgb = baseColors[segMode] || "0,0,0";
      const a = alphas[segMode] ?? 0.06;
      shapes.push({
        type: "rect", xref: "x", yref: "paper",
        x0: segStart, x1, y0: 0, y1: 1,
        fillcolor: `rgba(${rgb},${a})`,
        line: { width: 0 }, layer: "below",
      });
      segMode = modes[i];
      segStart = hours[i];
    }
  }
  return shapes;
}

// Shared helper: render one mode-dwell KPI card value into `el`.
function _renderModeDwellKpi(el, modeKey, modeName, ov, validation, totalWindows) {
  if (!el) return;
  const modeDur = ov.mode_duration_s || {};
  const dwellRatios = (validation || {}).dwell_ratios || ov.dwell_ratios || {};
  const pct = dwellRatios[modeKey] != null ? (dwellRatios[modeKey] * 100).toFixed(1) : null;
  const h = modeDur[modeName] ? (modeDur[modeName] / 3600).toFixed(1) + " h" : null;
  if (pct) {
    el.innerHTML = `${pct}<span class="kpi-unit">%</span>${h ? ` <span class="kpi-delta">${h}</span>` : ""}`;
  } else if (ov.mode_counts?.[modeName]) {
    const c = ov.mode_counts[modeName];
    const p = totalWindows ? ((c / totalWindows) * 100).toFixed(1) : "0";
    el.innerHTML = `${c.toLocaleString()}<span class="kpi-unit"> (${p}%)</span>`;
  } else {
    el.textContent = "—";
  }
}

function renderIllwerkeKPIs(ov, validation) {
  // --- Campaign days ---
  // n_days may be 0 if NPZ cache is empty — derive from date_range
  let displayDays = ov.n_days;
  if ((!displayDays || displayDays === 0) && ov.date_range?.length === 2) {
    const msPerDay = 86400000;
    const d0 = new Date(ov.date_range[0] + "T12:00:00Z");
    const d1 = new Date(ov.date_range[1] + "T12:00:00Z");
    displayDays = Math.round((d1 - d0) / msPerDay) + 1;
  }
  if (dom.iwKpiDays) dom.iwKpiDays.textContent = displayDays || "--";
  if (dom.iwKpiDaysRange && ov.date_range?.length === 2) {
    const [d0, d1] = ov.date_range.map((d) => {
      const dt = new Date(d + "T12:00:00Z");
      return dt.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
    });
    dom.iwKpiDaysRange.textContent = `${d0} – ${d1}`;
  }

  // --- Steady windows / total ---
  // NPZ-based window count may be 0; fall back to displaying dwell coverage
  const totalWindows = ov.total_windows || 0;
  const dwellFromOv = ov.dwell_ratios || {};
  const hasWindows = totalWindows > 0;
  if (dom.iwKpiWindows) {
    if (hasWindows) {
      dom.iwKpiWindows.textContent = totalWindows.toLocaleString();
    } else {
      // Compute total campaign seconds from mode_duration_s
      const totalS = Object.values(ov.mode_duration_s || {}).reduce(
        (s, v) => s + v,
        0,
      );
      const totalH = (totalS / 3600).toFixed(0);
      dom.iwKpiWindows.innerHTML = `${totalH}<span class="kpi-unit"> h</span>`;
    }
  }
  if (dom.iwKpiWindowsSub && validation) {
    const cov = validation.steady_coverage_pct;
    if (cov != null)
      dom.iwKpiWindowsSub.textContent = `${cov.toFixed(1)}% steady`;
  }

  // --- Turbine / Phasenschieber dwell ---
  _renderModeDwellKpi(dom.iwKpiTurbine, "TU", "Turbine", ov, validation, totalWindows);
  _renderModeDwellKpi(dom.iwKpiPH, "PH", "Phasenschieber", ov, validation, totalWindows);

  // --- Watch events (pipeline L5) ---
  const pe = ov.pipeline_events || {};
  if (dom.iwKpiWatch) {
    const n = pe.watch_events ?? "—";
    dom.iwKpiWatch.textContent = n;
    if (pe.watch_events != null)
      dom.iwKpiWatch.style.color = pe.watch_events > 50 ? "#f6bf3a" : "#eaf4ff";
  }

  // --- Alert events (pipeline L5) ---
  if (dom.iwKpiAlert) {
    const n = pe.alert_events ?? "—";
    dom.iwKpiAlert.textContent = n;
    if (pe.alert_events != null)
      dom.iwKpiAlert.style.color = pe.alert_events > 20 ? "#ff6673" : "#eaf4ff";
  }

  // --- Legacy CNF anom (if the element still exists) ---
  if (dom.iwKpiAnom && ov.cnf_anomaly_rate != null) {
    const rate = (ov.cnf_anomaly_rate * 100).toFixed(1);
    dom.iwKpiAnom.innerHTML = `${rate}<span class="kpi-unit">%</span>`;
    dom.iwKpiAnom.style.color =
      ov.cnf_anomaly_rate < 0.05
        ? "#2fd492"
        : ov.cnf_anomaly_rate < 0.15
          ? "#f6bf3a"
          : "#ff6673";
  }
}

function renderIllwerkeGantt(data) {
  if (!dom.iwGantt) return;
  const rows = data.rows || [];
  const days = data.days || [];

  // Update subtitle to reflect active mode source
  if (dom.iwGanttSubtitle) {
    const srcKey = data.mode_source || "kmeans";
    const srcLabel = IW_MODE_SOURCE_DISPLAY[srcKey] || srcKey;
    const prefix =
      srcKey === "kmeans"
        ? "K-Means labels"
        : `Mode predictions \u2014 ${srcLabel}`;
    dom.iwGanttSubtitle.textContent = `${prefix} \u00b7 each row\u202f=\u202fone day \u00b7 colour\u202f=\u202fmode \u00b7 hover for times & process vars \u00b7 click to load day detail`;
  }

  if (!rows.length) {
    dom.iwGantt.innerHTML =
      '<div class="placeholder-text">No Illwerke timeline data</div>';
    return;
  }

  // Group rows by mode — one Plotly trace per mode (for legend + colour)
  const byMode = {};
  for (const r of rows) {
    if (!byMode[r.mode]) byMode[r.mode] = [];
    byMode[r.mode].push(r);
  }

  const traces = [];
  // Render Standstill first (background), then PH, Turbine, Pump, Transitioning last
  const modeOrder = [
    "Standstill",
    "Phasenschieber",
    "Turbine",
    "Pump",
    "Transitioning",
  ];
  for (const mode of modeOrder) {
    const modeRows = byMode[mode];
    if (!modeRows || !modeRows.length) continue;
    traces.push({
      type: "bar",
      orientation: "h",
      name: IW_MODE_LABELS[mode] || mode,
      y: modeRows.map((r) => r.day),
      x: modeRows.map((r) => r.duration_h),
      base: modeRows.map((r) => r.start_h),
      marker: {
        color: IW_MODE_COLORS[mode] || "#6c757d",
        opacity: mode === "Standstill" ? 0.55 : 0.9,
        line: { width: 0 },
      },
      customdata: modeRows.map((r) => [
        r.start_hm,
        r.end_hm,
        r.rpm_mean,
        r.power_mean,
        r.flow_mean,
        r.day,
      ]),
      hovertemplate:
        `<b>${IW_MODE_LABELS[mode] || mode}</b><br>` +
        `%{customdata[0]} – %{customdata[1]}  (%{customdata[5]})<br>` +
        `RPM: %{customdata[2]} rpm &nbsp;·&nbsp; Power: %{customdata[3]} MW &nbsp;·&nbsp; Flow: %{customdata[4]} m³/s` +
        `<extra></extra>`,
    });
  }

  // Anomaly overlay — scatter trace of flagged windows for the selected model
  const apts = data.anomaly_pts || [];
  if (apts.length > 0) {
    const modelLabel =
      IW_MODEL_DISPLAY[data.anomaly_model] || data.anomaly_model || "Anomaly";
    traces.push({
      type: "scatter",
      mode: "markers",
      name: `Anomaly \u00b7 ${modelLabel}`,
      x: apts.map((p) => p.hour),
      y: apts.map((p) => p.day),
      marker: {
        symbol: "circle",
        size: 5,
        color: "#ff4557",
        opacity: 0.72,
        line: { width: 0 },
      },
      hovertemplate: `<b>Anomaly</b> (${modelLabel})<br>%{y} &nbsp; %{x:.2f}h<extra></extra>`,
    });
  }

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    barmode: "overlay",
    height: 340,
    margin: { t: 26, r: 24, b: 54, l: 102 },
    xaxis: {
      ...PLOTLY_LAYOUT_BASE.xaxis,
      title: { text: "Hour of Day (UTC)", font: { size: 11 } },
      range: [0, 24],
      tickvals: [0, 3, 6, 9, 12, 15, 18, 21, 24],
      ticktext: [
        "00:00",
        "03:00",
        "06:00",
        "09:00",
        "12:00",
        "15:00",
        "18:00",
        "21:00",
        "24:00",
      ],
      fixedrange: true,
    },
    yaxis: {
      ...PLOTLY_LAYOUT_BASE.yaxis,
      autorange: "reversed", // Latest day at top
      categoryorder: "array",
      categoryarray: [...days].reverse(),
      fixedrange: true,
    },
    legend: {
      orientation: "h",
      y: 1.12,
      x: 0,
      font: { size: 10.5 },
      bgcolor: "rgba(0,0,0,0)",
    },
    bargap: 0.28,
  };

  Plotly.react(dom.iwGantt, traces, layout, PLOTLY_CONFIG);

  // Click → select day and load process chart
  dom.iwGantt.removeAllListeners &&
    dom.iwGantt.removeAllListeners("plotly_click");
  dom.iwGantt.on("plotly_click", (evData) => {
    const pt = evData.points?.[0];
    if (!pt) return;
    const day = pt.y;
    if (!day) return;
    illwerkeSelectedDay = day;
    if (dom.iwDaySelect) dom.iwDaySelect.value = day;
    loadIllwerkeDailyChart(day);
  });
}

// ---------------------------------------------------------------------------
// Process Variables daily timeseries chart (RPM · Power · Flow · Head)
// ---------------------------------------------------------------------------
async function loadIllwerkeProcVarsChart(date) {
  if (!dom.iwProcVarsChart || !date) return;
  try {
    Plotly.purge(dom.iwProcVarsChart);
  } catch {
    /* no plot yet */
  }
  dom.iwProcVarsChart.innerHTML =
    '<div class="placeholder-text">Loading process variables…</div>';
  try {
    const data = await apiFetch(
      `/api/illwerke/pipeline/process_vars?date=${date}`,
    );
    if (!data.ready) {
      dom.iwProcVarsChart.innerHTML =
        '<div class="placeholder-text">Process variable cache not ready — try again shortly</div>';
      return;
    }
    renderIllwerkeProcVarsChart(data, date);
  } catch (err) {
    dom.iwProcVarsChart.innerHTML = `<div class="placeholder-text">Process vars unavailable: ${escapeHtml(err.message)}</div>`;
  }
}

function renderIllwerkeProcVarsChart(data, date) {
  if (!dom.iwProcVarsChart) return;
  const tsHm = data.ts_iso || []; // "HH:MM" strings
  const rpm = data.rpm || [];
  const power = data.power || [];
  const flowTU = data.flow_tu || [];
  const flowPU = data.flow_pu || [];
  const head = data.head || [];

  if (!tsHm.length) {
    dom.iwProcVarsChart.innerHTML =
      '<div class="placeholder-text">No process variable data for this day</div>';
    return;
  }

  // Convert "HH:MM" to fractional hour for x-axis
  const hrs = tsHm.map((s) => {
    const [h, m] = s.split(":").map(Number);
    return h + m / 60;
  });

  // Effective flow = max(flowTU, flowPU) at each point
  const flow = flowTU.map((tu, i) => Math.max(tu, flowPU[i] ?? 0));

  const dtLabel = (() => {
    if (!date) return "";
    const dt = new Date(date + "T12:00:00Z");
    return dt.toLocaleDateString("en-GB", {
      weekday: "long",
      day: "numeric",
      month: "long",
      year: "numeric",
    });
  })();

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      name: "RPM",
      x: hrs,
      y: rpm,
      line: { color: "#4fb8ff", width: 1.8 },
      yaxis: "y3",
      hovertemplate: "%{x:.2f}h → %{y:.0f} rpm<extra>RPM</extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Power (MW)",
      x: hrs,
      y: power,
      line: { color: "#ff8f5c", width: 1.8 },
      yaxis: "y",
      hovertemplate: "%{x:.2f}h → %{y:.1f} MW<extra>Power</extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Flow (m³/s)",
      x: hrs,
      y: flow,
      line: { color: "#2fd494", width: 1.6, dash: "dot" },
      yaxis: "y2",
      hovertemplate: "%{x:.2f}h → %{y:.1f} m³/s<extra>Flow</extra>",
    },
  ];

  if (head.some((v) => v !== 0)) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Head (m)",
      x: hrs,
      y: head,
      line: { color: "#a78bfa", width: 1.4, dash: "longdash" },
      yaxis: "y4",
      hovertemplate: "%{x:.2f}h → %{y:.1f} m<extra>Head</extra>",
      visible: "legendonly",
    });
  }

  Plotly.react(
    dom.iwProcVarsChart,
    traces,
    {
      ...PLOTLY_LAYOUT_BASE,
      height: 260,
      margin: { t: 16, r: 110, b: 54, l: 64 },
      xaxis: {
        ...PLOTLY_LAYOUT_BASE.xaxis,
        title: {
          text: dtLabel + " — measured at 1 Hz, averaged per minute",
          font: { size: 9.5 },
        },
        range: [0, 24],
        tickvals: [0, 4, 8, 12, 16, 20, 24],
        ticktext: [
          "00:00",
          "04:00",
          "08:00",
          "12:00",
          "16:00",
          "20:00",
          "24:00",
        ],
        fixedrange: true,
      },
      yaxis: {
        ...PLOTLY_LAYOUT_BASE.yaxis,
        title: { text: "Power (MW)", font: { size: 10, color: "#ff8f5c" } },
        fixedrange: true,
      },
      yaxis2: {
        title: { text: "Flow (m³/s)", font: { size: 10, color: "#2fd494" } },
        overlaying: "y",
        side: "right",
        showgrid: false,
        tickfont: { color: "#2fd494", size: 9 },
        fixedrange: true,
        zeroline: false,
      },
      yaxis3: {
        title: { text: "RPM", font: { size: 10, color: "#4fb8ff" } },
        overlaying: "y",
        side: "right",
        anchor: "free",
        position: 1.0,
        showgrid: false,
        tickfont: { color: "#4fb8ff", size: 9 },
        fixedrange: true,
        zeroline: false,
      },
      yaxis4: {
        overlaying: "y",
        side: "right",
        anchor: "free",
        position: 1.08,
        showgrid: false,
        visible: false,
        fixedrange: true,
      },
      legend: {
        orientation: "h",
        y: 1.12,
        x: 0,
        font: { size: 10 },
        bgcolor: "rgba(0,0,0,0)",
      },
    },
    PLOTLY_CONFIG,
  );
}

function renderIllwerkeDayModeTimeline(timelineData, procData, date) {
  if (!dom.iwProcVarsChart) return;
  if (!timelineData || !timelineData.hours || !timelineData.hours.length) {
    dom.iwProcVarsChart.innerHTML =
      '<div class="placeholder-text">No detailed timeline available for this day</div>';
    return;
  }

  const hours = timelineData.hours || [];
  const modes = timelineData.modes || [];
  const rpm = procData.rpm || [];
  const power = procData.power || [];
  const flowTU = procData.flow_tu || [];
  const flowPU = procData.flow_pu || [];
  const flow = flowTU.map((tu, idx) => Math.max(tu || 0, flowPU[idx] || 0));

  const dtLabel = (() => {
    if (!date) return "";
    const dt = new Date(date + "T12:00:00Z");
    return dt.toLocaleDateString("en-GB", {
      weekday: "long",
      day: "numeric",
      month: "long",
      year: "numeric",
    });
  })();

  const shapes = _buildModeShapes(hours, modes);

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      name: "Power (MW)",
      x: hours,
      y: power,
      line: { color: "#ff8f5c", width: 2 },
      yaxis: "y",
      hovertemplate: "%{x:.2f}h → %{y:.1f} MW<extra>Power</extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      name: "RPM",
      x: hours,
      y: rpm,
      line: { color: "#4fb8ff", width: 1.8, dash: "dot" },
      yaxis: "y3",
      hovertemplate: "%{x:.2f}h → %{y:.0f} rpm<extra>RPM</extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Flow (m³/s)",
      x: hours,
      y: flow,
      line: { color: "#2fd494", width: 1.6 },
      yaxis: "y2",
      hovertemplate: "%{x:.2f}h → %{y:.1f} m³/s<extra>Flow</extra>",
    },
  ];

  Plotly.react(
    dom.iwProcVarsChart,
    traces,
    {
      ...PLOTLY_LAYOUT_BASE,
      height: 280,
      margin: { t: 18, r: 94, b: 54, l: 64 },
      shapes,
      xaxis: {
        ...PLOTLY_LAYOUT_BASE.xaxis,
        title: { text: dtLabel, font: { size: 10 } },
        range: [0, 24],
        tickvals: [0, 4, 8, 12, 16, 20, 24],
        ticktext: ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
        fixedrange: true,
      },
      yaxis: {
        ...PLOTLY_LAYOUT_BASE.yaxis,
        title: { text: "Power (MW)", font: { size: 10, color: "#ff8f5c" } },
        fixedrange: true,
      },
      yaxis2: {
        title: { text: "Flow (m³/s)", font: { size: 10, color: "#2fd494" } },
        overlaying: "y",
        side: "right",
        showgrid: false,
        tickfont: { color: "#2fd494", size: 9 },
        fixedrange: true,
      },
      yaxis3: {
        title: { text: "RPM", font: { size: 10, color: "#4fb8ff" } },
        overlaying: "y",
        side: "right",
        anchor: "free",
        position: 0.98,
        showgrid: false,
        tickfont: { color: "#4fb8ff", size: 9 },
        fixedrange: true,
      },
      legend: {
        orientation: "h",
        y: 1.12,
        x: 0,
        font: { size: 10 },
        bgcolor: "rgba(0,0,0,0)",
      },
    },
    PLOTLY_CONFIG,
  );
}

function renderIllwerkeModeDonut(ov, validation) {
  if (!dom.iwModeDonut) return;

  // Prefer physics pipeline dwell ratios → more accurate than latent-window counts
  const dwellRatios = (validation || {}).dwell_ratios || ov.dwell_ratios || {};
  const modeDuration = ov.mode_duration_s || {};

  // Build display dataset: use dwell_ratios (fractional) if available
  let modes, values, totalLabel;
  const _DWELL_MODE_MAP = {
    ST: "Standstill",
    TU: "Turbine",
    PU: "Pump",
    PH: "Phasenschieber",
    TRANSITION: "Transitioning",
    UNKNOWN: "Unknown",
  };

  if (Object.keys(dwellRatios).length) {
    modes = Object.keys(dwellRatios).filter((k) => dwellRatios[k] > 0.0001);
    values = modes.map((k) => dwellRatios[k]);
    // Convert label keys (ST/TU/PU/PH) → display names
    modes = modes.map((k) => _DWELL_MODE_MAP[k] || k);
    const totalH =
      Object.values(modeDuration).reduce((s, v) => s + v, 0) / 3600;
    totalLabel = totalH > 0 ? `${totalH.toFixed(0)} h` : "campaign";
  } else {
    // Fallback: latent-window counts
    const counts = ov.mode_counts || {};
    modes = Object.keys(counts).filter((m) => counts[m] > 0);
    values = modes.map((m) => counts[m]);
    totalLabel = (ov.total_windows || 0).toLocaleString() + " win";
  }

  if (!modes.length) {
    dom.iwModeDonut.innerHTML =
      '<div class="placeholder-text">No mode data</div>';
    return;
  }

  Plotly.react(
    dom.iwModeDonut,
    [
      {
        type: "pie",
        hole: 0.52,
        labels: modes.map((m) => IW_MODE_LABELS[m] || m),
        values,
        marker: {
          colors: modes.map((m) => IW_MODE_COLORS[m] || "#6c757d"),
          line: { color: "#0b1218", width: 2 },
        },
        textinfo: "percent+label",
        textfont: { size: 11, color: "#eaf4ff" },
        hovertemplate: "<b>%{label}</b><br>%{percent}<extra></extra>",
      },
    ],
    {
      ...PLOTLY_LAYOUT_BASE,
      height: 300,
      margin: { t: 16, r: 16, b: 16, l: 16 },
      showlegend: false,
      annotations: [
        {
          text: `<b>${totalLabel}</b>`,
          x: 0.5,
          y: 0.5,
          xanchor: "center",
          yanchor: "middle",
          showarrow: false,
          font: { size: 14, color: "#eaf4ff" },
        },
      ],
    },
    PLOTLY_CONFIG,
  );
}

async function loadIllwerkeDailyChart(date) {
  if (!dom.iwProcessChart || !date) return;

  try {
    Plotly.purge(dom.iwProcessChart);
  } catch {
    /* no plot yet */
  }
  try {
    Plotly.purge(dom.iwProcVarsChart);
  } catch {
    /* no plot yet */
  }

  dom.iwProcessChart.innerHTML = '<div class="placeholder-text">Loading daily timeline…</div>';
  if (dom.iwProcVarsChart) {
    dom.iwProcVarsChart.innerHTML =
      '<div class="placeholder-text">Loading process variables…</div>';
  }

  const scoreFetch = apiFetch(
    `/api/illwerke/pipeline/scores?max_points=2000&date=${date}`,
  );
  const timelineFetch = apiFetch(
    `/api/illwerke/pipeline/scores?max_points=1440&date=${date}`,
  );
  const procFetch = apiFetch(`/api/illwerke/pipeline/process_vars?date=${date}`);

  let scoreData = null;
  let timelineData = null;
  let procData = null;

  try {
    scoreData = await scoreFetch;
    renderIllwerkeDailyZChart(scoreData, date);
  } catch (err) {
    const errorMessage = escapeHtml(err.message || "Unknown error");
    dom.iwProcessChart.innerHTML = `<div class="placeholder-text">Error loading daily timeline: ${errorMessage}</div>`;
  }

  if (dom.iwProcVarsChart) {
    try {
      [timelineData, procData] = await Promise.all([timelineFetch, procFetch]);
      renderIllwerkeDayModeTimeline(timelineData, procData, date);
    } catch (err) {
      const errorMessage = escapeHtml(err.message || "Unknown error");
      dom.iwProcVarsChart.innerHTML =
        `<div class="placeholder-text">Error loading day timeline: ${errorMessage}</div>`;
    }
  }
}

function renderIllwerkeDailyZChart(data, date) {
  if (!dom.iwProcessChart) return;
  const tsIso = data.ts_iso || [];
  const zScores = data.z_scores || [];
  const alertLevel = data.alert_level || [];
  const modes = data.modes || [];
  const watchThr = data.watch_threshold ?? 4.0;
  const alertThr = data.alert_threshold ?? 6.0;

  if (!tsIso.length) {
    dom.iwProcessChart.innerHTML =
      '<div class="placeholder-text">No data for this day</div>';
    return;
  }

  // Convert ISO strings to "HH:MM" labels for x-axis
  const hours = tsIso.map((s) => {
    const dt = new Date(s + ":00Z");
    return dt.getUTCHours() + dt.getUTCMinutes() / 60;
  });

  // One trace per mode
  const modeIndices = {};
  for (let i = 0; i < modes.length; i++) {
    const m = modes[i];
    if (!modeIndices[m]) modeIndices[m] = [];
    modeIndices[m].push(i);
  }

  const traces = Object.entries(modeIndices).map(([mode, idxArr]) => ({
    type: "scatter",
    mode: "markers",
    name: IW_MODE_LABELS[mode] || mode,
    x: idxArr.map((i) => hours[i]),
    y: idxArr.map((i) => zScores[i]),
    marker: {
      size: idxArr.map((i) =>
        alertLevel[i] >= 2 ? 7 : alertLevel[i] === 1 ? 5 : 3,
      ),
      color: IW_MODE_COLORS[mode] || "#6c757d",
      opacity: idxArr.map((i) => (alertLevel[i] > 0 ? 1.0 : 0.5)),
      line: {
        width: idxArr.map((i) => (alertLevel[i] >= 2 ? 1.5 : 0)),
        color: "#ff6673",
      },
    },
    hovertemplate: `<b>${IW_MODE_LABELS[mode] || mode}</b><br>%{x:.2f}h → z=%{y:.1f}<extra></extra>`,
  }));

  // Threshold lines
  if (hours.length >= 2) {
    const x0 = hours[0],
      x1 = hours[hours.length - 1];
    traces.push({
      type: "scatter",
      mode: "lines",
      name: `Watch (${watchThr}σ)`,
      x: [x0, x1],
      y: [watchThr, watchThr],
      line: { color: "#f6bf3a", dash: "dot", width: 1.5 },
      hoverinfo: "skip",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      name: `Alert (${alertThr}σ)`,
      x: [x0, x1],
      y: [alertThr, alertThr],
      line: { color: "#ff6673", dash: "dash", width: 1.5 },
      hoverinfo: "skip",
    });
  }

  const dtLabel = (() => {
    if (!date) return "";
    const dt = new Date(date + "T12:00:00Z");
    return dt.toLocaleDateString("en-GB", {
      weekday: "long",
      day: "numeric",
      month: "long",
      year: "numeric",
    });
  })();

  Plotly.react(
    dom.iwProcessChart,
    traces,
    {
      ...PLOTLY_LAYOUT_BASE,
      height: 300,
      margin: { t: 20, r: 24, b: 54, l: 64 },
      xaxis: {
        ...PLOTLY_LAYOUT_BASE.xaxis,
        title: { text: dtLabel, font: { size: 10.5 } },
        range: [0, 24],
        tickvals: [0, 4, 8, 12, 16, 20, 24],
        ticktext: [
          "00:00",
          "04:00",
          "08:00",
          "12:00",
          "16:00",
          "20:00",
          "24:00",
        ],
      },
      yaxis: {
        ...PLOTLY_LAYOUT_BASE.yaxis,
        title: { text: "Anomaly Z-Score (σ)", font: { size: 11 } },
        rangemode: "tozero",
      },
      legend: {
        orientation: "h",
        y: -0.3,
        font: { size: 10 },
        bgcolor: "rgba(0,0,0,0)",
      },
    },
    PLOTLY_CONFIG,
  );
}

function renderIllwerkeProcessChart(data) {
  if (!dom.iwProcessChart) return;
  const hours = data.hours || [];
  const power = data.power_mw || [];
  const rpm = data.rpm || [];
  const modes = data.modes || [];
  const isAnom = data.is_anomaly || [];

  const shapes = _buildModeShapes(hours, modes, 0.14);

  // Anomaly marker positions
  const anomH = hours.filter((_, i) => isAnom[i]);
  const anomP = power.filter((_, i) => isAnom[i]);

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      name: "Power (MW)",
      x: hours,
      y: power,
      line: { color: "#ff8f5c", width: 2 },
      yaxis: "y",
      hovertemplate: "%{x:.2f}h → %{y:.1f} MW<extra>Power</extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      name: "RPM",
      x: hours,
      y: rpm,
      line: { color: "#4fb8ff", width: 1.5, dash: "dot" },
      yaxis: "y2",
      hovertemplate: "%{x:.2f}h → %{y:.0f} rpm<extra>RPM</extra>",
    },
  ];

  if (anomH.length) {
    traces.push({
      type: "scatter",
      mode: "markers",
      name: "CNF Anomaly",
      x: anomH,
      y: anomP,
      marker: {
        symbol: "x",
        size: 9,
        color: "#ff6673",
        line: { color: "#fff", width: 1.5 },
      },
      yaxis: "y",
      hovertemplate: "Anomaly at %{x:.2f}h<extra></extra>",
    });
  }

  const dtLabel = (() => {
    if (!data.date) return "";
    const dt = new Date(data.date + "T12:00:00Z");
    return dt.toLocaleDateString("en-GB", {
      weekday: "long",
      day: "numeric",
      month: "long",
      year: "numeric",
    });
  })();

  Plotly.react(
    dom.iwProcessChart,
    traces,
    {
      ...PLOTLY_LAYOUT_BASE,
      height: 300,
      margin: { t: 20, r: 68, b: 54, l: 64 },
      shapes,
      xaxis: {
        ...PLOTLY_LAYOUT_BASE.xaxis,
        title: { text: dtLabel, font: { size: 10.5 } },
        range: [0, 24],
        tickvals: [0, 4, 8, 12, 16, 20, 24],
        ticktext: [
          "00:00",
          "04:00",
          "08:00",
          "12:00",
          "16:00",
          "20:00",
          "24:00",
        ],
      },
      yaxis: {
        ...PLOTLY_LAYOUT_BASE.yaxis,
        title: { text: "Power (MW)", font: { size: 11 } },
        range: [-320, 320],
        zeroline: true,
        zerolinecolor: "#2a3e51",
      },
      yaxis2: {
        title: { text: "RPM", font: { size: 11, color: "#4fb8ff" } },
        overlaying: "y",
        side: "right",
        range: [0, 450],
        gridcolor: "rgba(0,0,0,0)",
        color: "#4fb8ff",
        zeroline: false,
      },
      legend: {
        orientation: "h",
        y: -0.28,
        font: { size: 10 },
        bgcolor: "rgba(0,0,0,0)",
      },
    },
    PLOTLY_CONFIG,
  );
}

function renderIllwerkeScoreTimeline(data) {
  if (!dom.iwScoreTimeline) return;
  const tsIso = data.ts_iso || [];
  if (!tsIso.length) {
    dom.iwScoreTimeline.innerHTML =
      '<div class="placeholder-text">No score data available</div>';
    return;
  }

  // Support both pipeline z-scores and legacy CNF log-likelihood scores
  const scores = data.z_scores || data.scores || [];
  const alertLevel =
    data.alert_level || data.is_anomaly?.map((v) => (v ? 1 : 0)) || [];
  const modes = data.modes || [];
  const watchThr = data.watch_threshold ?? data.threshold ?? null;
  const alertThr = data.alert_threshold ?? null;
  const isPipeline = !!data.z_scores;
  const yLabel = isPipeline ? "Anomaly Z-Score (σ)" : "Log-likelihood Score";

  // One trace per mode
  const modeIndices = {};
  for (let i = 0; i < modes.length; i++) {
    const m = modes[i];
    if (!modeIndices[m]) modeIndices[m] = [];
    modeIndices[m].push(i);
  }

  const traces = Object.entries(modeIndices).map(([mode, idxArr]) => ({
    type: "scatter",
    mode: "markers",
    name: IW_MODE_LABELS[mode] || mode,
    x: idxArr.map((i) => tsIso[i]),
    y: idxArr.map((i) => scores[i]),
    marker: {
      size: idxArr.map((i) => {
        const al = alertLevel[i];
        return al >= 2 ? 6 : al === 1 ? 4 : 3;
      }),
      color: IW_MODE_COLORS[mode] || "#6c757d",
      opacity: idxArr.map((i) => (alertLevel[i] > 0 ? 1.0 : 0.4)),
      line: {
        width: idxArr.map((i) => (alertLevel[i] >= 2 ? 1.2 : 0)),
        color: "#ff6673",
      },
    },
    hovertemplate: `<b>${IW_MODE_LABELS[mode] || mode}</b><br>%{x}<br>${isPipeline ? "Z" : "Score"}: %{y:.2f}<extra></extra>`,
  }));

  // Threshold lines
  if (tsIso.length >= 2) {
    if (watchThr != null) {
      traces.push({
        type: "scatter",
        mode: "lines",
        name: isPipeline
          ? `Watch (${watchThr}σ)`
          : `Threshold (${watchThr.toFixed(2)})`,
        x: [tsIso[0], tsIso[tsIso.length - 1]],
        y: [watchThr, watchThr],
        line: { color: "#f6bf3a", dash: "dot", width: 1.5 },
        hoverinfo: "skip",
      });
    }
    if (alertThr != null && alertThr !== watchThr) {
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `Alert (${alertThr}σ)`,
        x: [tsIso[0], tsIso[tsIso.length - 1]],
        y: [alertThr, alertThr],
        line: { color: "#ff6673", dash: "dash", width: 1.5 },
        hoverinfo: "skip",
      });
    }
  }

  Plotly.react(
    dom.iwScoreTimeline,
    traces,
    {
      ...PLOTLY_LAYOUT_BASE,
      height: 280,
      margin: { t: 20, r: 24, b: 64, l: 68 },
      xaxis: {
        ...PLOTLY_LAYOUT_BASE.xaxis,
        title: { text: "Date (UTC)", font: { size: 11 } },
        tickangle: -30,
      },
      yaxis: {
        ...PLOTLY_LAYOUT_BASE.yaxis,
        title: { text: yLabel, font: { size: 11 } },
        rangemode: isPipeline ? "tozero" : "normal",
      },
      legend: {
        orientation: "h",
        y: -0.35,
        font: { size: 10 },
        bgcolor: "rgba(0,0,0,0)",
      },
    },
    PLOTLY_CONFIG,
  );
}

// ── Anomaly Events (pipeline L5) ──────────────────────────────────────────────
function renderIllwerkeAnomalyEvents(data) {
  if (!data) return;
  const events = data.events || data.anomaly_events || [];

  // ---- Scatter chart ----
  if (dom.iwEventsChart) {
    if (!events.length) {
      dom.iwEventsChart.innerHTML =
        '<div class="placeholder-text">No anomaly events</div>';
    } else {
      const byLevel = { alert: [], watch: [] };
      for (const ev of events) {
        const sev = (ev.severity || "watch").toLowerCase();
        if (byLevel[sev] === undefined) byLevel["watch"].push(ev);
        else byLevel[sev].push(ev);
      }
      const traces = Object.entries(byLevel)
        .filter(([, arr]) => arr.length)
        .map(([sev, arr]) => ({
          type: "scatter",
          mode: "markers",
          name: sev === "alert" ? "Alert" : "Watch",
          x: arr.map((e) => e.start_utc || e.t_start_iso || e.ts_start),
          y: arr.map((e) => e.peak_z ?? e.peak_z_score ?? e.peak_score ?? 0),
          marker: {
            size: arr.map((e) => {
              const z = e.peak_z ?? e.peak_z_score ?? 4;
              return Math.min(16, Math.max(5, z * 1.2));
            }),
            color: IW_SEVERITY_COLORS[sev] || "#6c757d",
            opacity: 0.85,
            symbol: sev === "alert" ? "circle" : "circle-open",
          },
          customdata: arr.map((e) => [
            e.mode || "—",
            e.sub_mode || "—",
            e.duration_s != null ? `${e.duration_s.toFixed(0)} s` : "—",
            (e.peak_z ?? e.peak_z_score ?? e.peak_score ?? 0).toFixed(2),
          ]),
          hovertemplate:
            "<b>%{customdata[0]} / %{customdata[1]}</b><br>" +
            "%{x}<br>Peak: %{customdata[3]}σ  ·  %{customdata[2]}<extra>" +
            (sev === "alert" ? "Alert" : "Watch") +
            "</extra>",
        }));

      Plotly.react(
        dom.iwEventsChart,
        traces,
        {
          ...PLOTLY_LAYOUT_BASE,
          height: 220,
          margin: { t: 12, r: 20, b: 56, l: 56 },
          xaxis: {
            ...PLOTLY_LAYOUT_BASE.xaxis,
            title: { text: "Date (UTC)", font: { size: 10.5 } },
            tickangle: -25,
          },
          yaxis: {
            ...PLOTLY_LAYOUT_BASE.yaxis,
            title: { text: "Peak Z-Score (σ)", font: { size: 11 } },
            rangemode: "tozero",
          },
          legend: {
            orientation: "h",
            y: -0.35,
            font: { size: 10.5 },
            bgcolor: "rgba(0,0,0,0)",
          },
        },
        PLOTLY_CONFIG,
      );
    }
  }

  // ---- Table ----
  if (dom.iwEventsTable) {
    if (!events.length) {
      dom.iwEventsTable.innerHTML =
        '<div class="placeholder-text">No anomaly events</div>';
    } else {
      const rows = events
        .slice(0, 200)
        .map((ev) => {
          const sev = (ev.severity || "watch").toLowerCase();
          const badgeCls =
            sev === "alert" ? "iw-severity-alert" : "iw-severity-watch";
          const t = ev.start_utc || ev.t_start_iso || ev.ts_start || "—";
          const tShort =
            t.length > 16 ? t.substring(0, 16).replace("T", " ") : t;
          const mode = ev.mode || "—";
          const sub = ev.sub_mode || "—";
          const dur =
            ev.duration_s != null ? `${ev.duration_s.toFixed(0)} s` : "—";
          const peak =
            (ev.peak_z ?? ev.peak_z_score) != null
              ? `${(ev.peak_z ?? ev.peak_z_score).toFixed(2)}σ`
              : ev.peak_score != null
                ? ev.peak_score.toFixed(2)
                : "—";
          return `<tr>
            <td class="mono">${escapeHtml(tShort)}</td>
            <td>${escapeHtml(mode)}</td>
            <td>${escapeHtml(sub)}</td>
            <td>${escapeHtml(dur)}</td>
            <td>${escapeHtml(peak)}</td>
            <td><span class="${badgeCls}">${escapeHtml(ev.severity || "watch")}</span></td>
          </tr>`;
        })
        .join("");

      dom.iwEventsTable.innerHTML = `
        <table class="iw-events-table iw-models-table">
          <thead><tr>
            <th>Time (UTC)</th><th>Mode</th><th>Sub-mode</th>
            <th>Duration</th><th>Peak σ</th><th>Severity</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>`;

      if (events.length > 200) {
        dom.iwEventsTable.innerHTML +=
          `<div class="placeholder-text" style="margin-top:6px;font-size:11px">` +
          `Showing 200 of ${events.length} events. Use severity filter to narrow.</div>`;
      }
    }
  }

  // ---- Severity summary counts above the table ----
  const summary = data.summary || {};
  const total = summary.total_events ?? events.length;
  const watch =
    summary.watch_events ??
    events.filter((e) => (e.severity || "").toLowerCase() === "watch").length;
  const alert =
    summary.alert_events ??
    events.filter((e) => (e.severity || "").toLowerCase() === "alert").length;

  // Inject counts into KPI cards if they haven't been set yet
  if (dom.iwKpiWatch && dom.iwKpiWatch.textContent === "--")
    dom.iwKpiWatch.textContent = watch;
  if (dom.iwKpiAlert && dom.iwKpiAlert.textContent === "--")
    dom.iwKpiAlert.textContent = alert;
}

// ── Machine State Transitions (pipeline L2) ──────────────────────────────────
function renderIllwerkeTransitions(data) {
  const transitions = data.transitions || data.segments || [];

  if (dom.iwTransitionsSummary) {
    const total = transitions.length;
    const verified = transitions.filter(
      (t) => t.signature_match === true,
    ).length;
    const failed = transitions.filter(
      (t) => t.signature_match === false,
    ).length;
    dom.iwTransitionsSummary.innerHTML =
      `<span style="color:#eaf4ff"><b>${total}</b> transitions</span> · ` +
      `<span style="color:#2fd492"><b>${verified}</b> signature-verified</span> · ` +
      `<span style="color:#ff6673"><b>${failed}</b> unverified</span>`;
  }

  if (!dom.iwTransitionsTable) return;
  if (!transitions.length) {
    dom.iwTransitionsTable.innerHTML =
      '<div class="placeholder-text">No transition data</div>';
    return;
  }

  const rows = transitions
    .slice(0, 150)
    .map((tr) => {
      const tStr = (tr.start_utc || tr.t_start_iso || tr.ts_start || "—")
        .substring(0, 16)
        .replace("T", " ");
      const tType = tr.transition_type || tr.type || "—";
      const dur = tr.duration_s != null ? `${tr.duration_s.toFixed(0)} s` : "—";
      const sm = tr.signature_match;
      let sig;
      if (sm === true)
        sig = '<span class="iw-sig-ok" title="Signature verified">✓ OK</span>';
      else if (sm === false)
        sig =
          '<span class="iw-sig-fail" title="Signature mismatch">✗ Fail</span>';
      else sig = '<span class="iw-sig-null">—</span>';
      const day = tStr.substring(0, 10);
      return `<tr>
        <td class="mono">${escapeHtml(tStr)}</td>
        <td>${escapeHtml(tType)}</td>
        <td>${sig}</td>
        <td>${escapeHtml(dur)}</td>
        <td>${escapeHtml(day)}</td>
      </tr>`;
    })
    .join("");

  dom.iwTransitionsTable.innerHTML = `
    <table class="iw-transitions-table iw-models-table">
      <thead><tr>
        <th>Time (UTC)</th><th>Transition</th><th>Signature</th>
        <th>Duration</th><th>Day</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  if (transitions.length > 150) {
    dom.iwTransitionsTable.innerHTML +=
      `<div class="placeholder-text" style="margin-top:6px;font-size:11px">` +
      `Showing 150 of ${transitions.length} transitions.</div>`;
  }
}

// ── Physics Oracle & Signal Health (pipeline L1) ──────────────────────────────
function renderIllwerkePhysicsOracle(validationData, overviewData) {
  if (!dom.iwOraclePanel) return;
  const v = validationData || {};
  const ov = overviewData || {};

  // Dwell ratios bar chart (inline Plotly) ─────────────────
  const dwellRatios = v.dwell_ratios || ov.dwell_ratios || {};
  const _DWELL_MODE_LABELS = {
    ST: "Standstill",
    TU: "Turbine",
    PU: "Pump",
    PH: "Phasenschieber",
    TRANSITION: "Transitioning",
    UNKNOWN: "Unknown",
  };
  const modeKeys = Object.keys(dwellRatios).filter(
    (k) => dwellRatios[k] > 0.0001,
  );
  const modeColors = {
    ST: "#6c757d",
    TU: "#4fb8ff",
    PU: "#2fd492",
    PH: "#fd7e14",
    TRANSITION: "#9b59b6",
    UNKNOWN: "#e74c3c",
  };

  let dwellHtml = "";
  if (modeKeys.length) {
    const dwellDiv = document.createElement("div");
    dwellDiv.style.cssText = "height:180px;margin-bottom:8px";
    dom.iwOraclePanel.innerHTML = "";
    dom.iwOraclePanel.appendChild(dwellDiv);

    Plotly.react(
      dwellDiv,
      [
        {
          type: "bar",
          orientation: "h",
          x: modeKeys.map((k) => (dwellRatios[k] * 100).toFixed(1)),
          y: modeKeys.map((k) => _DWELL_MODE_LABELS[k] || k),
          text: modeKeys.map((k) => `${(dwellRatios[k] * 100).toFixed(1)}%`),
          textposition: "outside",
          textfont: { size: 11.5, color: "#eaf4ff" },
          marker: { color: modeKeys.map((k) => modeColors[k] || "#6c757d") },
          hovertemplate: "%{y}: %{x}%<extra></extra>",
        },
      ],
      {
        ...PLOTLY_LAYOUT_BASE,
        height: 180,
        margin: { t: 8, r: 60, b: 36, l: 120 },
        xaxis: {
          ...PLOTLY_LAYOUT_BASE.xaxis,
          range: [
            0,
            Math.max(...modeKeys.map((k) => dwellRatios[k] * 100)) * 1.25,
          ],
          title: { text: "Campaign Dwell (%)", font: { size: 10.5 } },
        },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, tickfont: { size: 11.5 } },
        showlegend: false,
      },
      PLOTLY_CONFIG,
    );
  } else {
    dom.iwOraclePanel.innerHTML = "";
  }

  // ── Metrics grid ─────────────────────────────────────────
  const cov = v.steady_coverage_pct;
  const covHtml =
    cov != null
      ? `<span style="color:${cov >= 90 ? "#2fd492" : cov >= 75 ? "#f6bf3a" : "#ff6673"}">${cov.toFixed(1)}%</span>`
      : "—";

  const freeze =
    v.sensor_freeze_channels ??
    Object.keys(v.sensor_freeze?.faulted_channels || {}).length;
  const freezeHtml =
    freeze != null
      ? `<span style="color:${freeze === 0 ? "#2fd492" : "#f6bf3a"}">${freeze} ch</span>`
      : "—";

  const chiPass = v.head_chi2_pass;
  const chiHtml =
    chiPass === true
      ? '<span style="color:#2fd492">✓ Pass</span>'
      : chiPass === false
        ? '<span style="color:#ff6673">✗ Fail</span>'
        : "—";

  const unknownFrac = v.unknown_fraction ?? v.UNKNOWN_fraction;
  const unknownHtml =
    unknownFrac != null
      ? `<span style="color:${unknownFrac < 0.05 ? "#2fd492" : unknownFrac < 0.1 ? "#f6bf3a" : "#ff6673"}">${(unknownFrac * 100).toFixed(1)}%</span>`
      : "—";

  const nTrans = v.n_transitions ?? "—";
  const oracleAcc = v.oracle_accuracy;
  const accHtml =
    oracleAcc != null
      ? `<span style="color:${oracleAcc >= 0.95 ? "#2fd492" : "#f6bf3a"}">${(oracleAcc * 100).toFixed(1)}%</span>`
      : "—";

  const gridDiv = document.createElement("div");
  gridDiv.className = "iw-oracle-grid";
  gridDiv.innerHTML = `
    <div class="iw-oracle-metric"><span class="iw-oracle-label">Steady Coverage</span><span class="iw-oracle-value">${covHtml}</span></div>
    <div class="iw-oracle-metric"><span class="iw-oracle-label">Frozen Sensors</span><span class="iw-oracle-value">${freezeHtml}</span></div>
    <div class="iw-oracle-metric"><span class="iw-oracle-label">Head χ² Test</span><span class="iw-oracle-value">${chiHtml}</span></div>
    <div class="iw-oracle-metric"><span class="iw-oracle-label">Unknown Fraction</span><span class="iw-oracle-value">${unknownHtml}</span></div>
    <div class="iw-oracle-metric"><span class="iw-oracle-label">Transitions Typed</span><span class="iw-oracle-value"><span style="color:#eaf4ff">${nTrans}</span></span></div>
    <div class="iw-oracle-metric"><span class="iw-oracle-label">Oracle Accuracy</span><span class="iw-oracle-value">${accHtml}</span></div>
  `;
  dom.iwOraclePanel.appendChild(gridDiv);
}

function renderIllwerkeModels(data) {
  if (!dom.iwModelsTable) return;
  const models = data.models || [];
  if (!models.length) {
    dom.iwModelsTable.innerHTML =
      '<div class="placeholder-text">No model data available</div>';
    return;
  }

  let rows = "";
  for (const m of models) {
    const badge =
      m.status === "ok"
        ? `<span class="iw-badge-ok">OK</span>`
        : `<span class="iw-badge-missing">${escapeHtml(m.status || "missing")}</span>`;

    const rateRaw = m.anomaly_rate;
    let rateTxt = "—";
    let rateClass = "";
    if (rateRaw != null) {
      const pct = (rateRaw * 100).toFixed(2);
      rateTxt = `${pct}%`;
      rateClass =
        rateRaw < 0.02
          ? "iw-anom-rate-good"
          : rateRaw < 0.15
            ? "iw-anom-rate-warn"
            : "iw-anom-rate-high";
    }

    const windowTxt =
      m.n_anomalous != null
        ? `${m.n_anomalous.toLocaleString()} / ${(m.n_windows || 0).toLocaleString()}`
        : "—";

    rows += `<tr>
      <td class="iw-model-name">${escapeHtml(m.display)}</td>
      <td>${badge}</td>
      <td class="iw-metric ${rateClass}">${escapeHtml(rateTxt)}</td>
      <td class="iw-metric">${escapeHtml(windowTxt)}</td>
    </tr>`;
  }

  dom.iwModelsTable.innerHTML = `
    <table class="iw-models-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Status</th>
          <th>Anomaly Rate</th>
          <th>Anomalous / Total Windows</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
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

  // DTSS training button
  if (dom.btnDtssTrain) {
    dom.btnDtssTrain.addEventListener("click", startDtssTraining);
  }

  // Illwerke day selector
  if (dom.iwDaySelect) {
    dom.iwDaySelect.addEventListener("change", (e) => {
      const day = e.target.value;
      if (day) {
        illwerkeSelectedDay = day;
        loadIllwerkeDailyChart(day);
      }
    });
  }

  // Shared helper: re-fetch Gantt with current mode-source + anomaly model
  async function reloadIllwerkeGantt() {
    const modelKey = dom.iwGanttModelSelect ? dom.iwGanttModelSelect.value : "";
    const modeSource = dom.iwGanttModeSource
      ? dom.iwGanttModeSource.value
      : "pipeline";
    let url;
    if (modeSource === "pipeline") {
      // Always use pipeline endpoint — no model overlay needed
      url = "/api/illwerke/pipeline/gantt";
    } else {
      const params = new URLSearchParams();
      if (modelKey) params.set("model", modelKey);
      if (modeSource !== "kmeans") params.set("mode_source", modeSource);
      url =
        "/api/illwerke/gantt" +
        (params.toString() ? "?" + params.toString() : "");
    }
    try {
      const gantt = await apiFetch(url);
      renderIllwerkeGantt(gantt);
    } catch (err) {
      showToast("Gantt update failed: " + escapeHtml(err.message), "error");
    }
  }

  // Illwerke gantt model selector — re-fetch gantt with anomaly overlay
  if (dom.iwGanttModelSelect) {
    dom.iwGanttModelSelect.addEventListener("change", reloadIllwerkeGantt);
  }

  // Illwerke gantt mode-source selector — re-fetch gantt with selected mode source
  if (dom.iwGanttModeSource) {
    dom.iwGanttModeSource.addEventListener("change", reloadIllwerkeGantt);
  }

  // Severity filter for anomaly events table
  if (dom.iwEventsSeverity) {
    dom.iwEventsSeverity.addEventListener("change", async () => {
      const sev = dom.iwEventsSeverity.value || "";
      try {
        const url = sev
          ? `/api/illwerke/pipeline/events?severity=${encodeURIComponent(sev)}`
          : "/api/illwerke/pipeline/events";
        const data = await apiFetch(url);
        renderIllwerkeAnomalyEvents(data);
      } catch (err) {
        showToast("Events filter failed: " + escapeHtml(err.message), "error");
      }
    });
  }

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
