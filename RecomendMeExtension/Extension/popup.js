const predictBtn = document.getElementById("predictBtn");

const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const ratingValueEl = document.getElementById("ratingValue");
const starValueEl = document.getElementById("starValue");
const caseValueEl = document.getElementById("caseValue");
const normalizedRequesterEl = document.getElementById("normalizedRequester");
const normalizedRecommenderEl = document.getElementById("normalizedRecommender");

const graphSectionEl = document.getElementById("graphSection");
const graphCanvas = document.getElementById("graphCanvas");
const graphMetaEl = document.getElementById("graphMeta");

const suggestionsSectionEl = document.getElementById("suggestionsSection");
const suggestionsListEl = document.getElementById("suggestionsList");
const suggestionsMetaEl = document.getElementById("suggestionsMeta");

function renderStars(n) {
  return "★".repeat(n) + "☆".repeat(5 - n);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function rgb(r, g, b) {
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

function ratingColor(avgRating, fallbackGroup) {
  if (avgRating == null || Number.isNaN(avgRating)) {
    if (fallbackGroup === "focus_requester") return "#60a5fa";
    if (fallbackGroup === "focus_recommender") return "#f59e0b";
    if (fallbackGroup === "shared_neighbor") return "#a78bfa";
    return "#93c5fd";
  }

  const t = clamp((avgRating - 1) / 4, 0, 1);

  let r, g, b;
  if (t < 0.33) {
    const u = t / 0.33;
    r = lerp(59, 34, u);
    g = lerp(130, 211, u);
    b = lerp(246, 238, u);
  } else if (t < 0.66) {
    const u = (t - 0.33) / 0.33;
    r = lerp(34, 250, u);
    g = lerp(211, 204, u);
    b = lerp(238, 21, u);
  } else {
    const u = (t - 0.66) / 0.34;
    r = lerp(250, 239, u);
    g = lerp(204, 68, u);
    b = lerp(21, 68, u);
  }

  return rgb(r, g, b);
}

function edgeColor(rating) {
  return ratingColor(rating ?? 3, null);
}

function edgeWidth(rating) {
  if (rating == null || Number.isNaN(rating)) return 1.0;
  return 0.8 + (rating - 1) * 1.1;
}

function drawArrow(ctx, x1, y1, x2, y2, color, width) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 6 + width;

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.globalAlpha = 0.55;

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - headLen * Math.cos(angle - Math.PI / 7),
    y2 - headLen * Math.sin(angle - Math.PI / 7)
  );
  ctx.lineTo(
    x2 - headLen * Math.cos(angle + Math.PI / 7),
    y2 - headLen * Math.sin(angle + Math.PI / 7)
  );
  ctx.closePath();
  ctx.fill();

  ctx.globalAlpha = 1;
}

function jitter(amount) {
  return (Math.random() - 0.5) * amount;
}

function placeCluster(nodes, cx, cy, rx, ry, startAngle, endAngle, positions) {
  const n = nodes.length;
  if (!n) return;

  nodes.forEach((node, i) => {
    const t = n === 1 ? 0.5 : i / (n - 1);
    const angle = lerp(startAngle, endAngle, t);
    positions[node.id] = {
      x: cx + rx * Math.cos(angle) + jitter(18),
      y: cy + ry * Math.sin(angle) + jitter(22)
    };
  });
}

function drawNetwork(canvas, graph) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  if (!nodes.length) return;

  const requester = nodes.find(n => n.group === "focus_requester");
  const recommender = nodes.find(n => n.group === "focus_recommender");
  const requesterOnly = nodes.filter(n => n.group === "requester_neighbor");
  const recommenderOnly = nodes.filter(n => n.group === "recommender_neighbor");
  const shared = nodes.filter(n => n.group === "shared_neighbor");

  const positions = {};

  if (requester) positions[requester.id] = { x: width * 0.23, y: height * 0.50 };
  if (recommender) positions[recommender.id] = { x: width * 0.77, y: height * 0.50 };

  placeCluster(
    requesterOnly,
    width * 0.26,
    height * 0.50,
    width * 0.18,
    height * 0.28,
    Math.PI * 0.65,
    Math.PI * 1.35,
    positions
  );

  placeCluster(
    recommenderOnly,
    width * 0.74,
    height * 0.50,
    width * 0.18,
    height * 0.28,
    -Math.PI * 0.35,
    Math.PI * 0.35,
    positions
  );

  shared.forEach((node, i) => {
    const t = shared.length === 1 ? 0.5 : i / Math.max(shared.length - 1, 1);
    positions[node.id] = {
      x: lerp(width * 0.40, width * 0.60, t) + jitter(22),
      y: height * 0.50 + jitter(120)
    };
  });

  for (const edge of edges) {
    const a = positions[edge.source];
    const b = positions[edge.target];
    if (!a || !b) continue;
    drawArrow(ctx, a.x, a.y, b.x, b.y, edgeColor(edge.rating), edgeWidth(edge.rating));
  }

  if (requester && recommender) {
    const a = positions[requester.id];
    const b = positions[recommender.id];
    drawArrow(ctx, a.x, a.y, b.x, b.y, "#cbd5e1", 2.2);
  }

  for (const node of nodes) {
    const p = positions[node.id];
    if (!p) continue;

    const radius = Math.max(3, Math.min(node.size || 6, 9));
    const fill = ratingColor(node.avg_rating_received, node.group);

    ctx.beginPath();
    ctx.fillStyle = fill;
    ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
    ctx.fill();

    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "#203048";
    ctx.stroke();
  }

  ctx.font = "10px Arial";
  ctx.fillStyle = "#dbe5f4";
  ctx.textAlign = "center";

  for (const node of nodes) {
    const p = positions[node.id];
    if (!p) continue;

    const radius = Math.max(3, Math.min(node.size || 6, 9));
    ctx.fillText(node.label, p.x, p.y - radius - 6);
  }
}

function avatarDataUrl(label) {
  const safeLabel = String(label || "?").slice(-4);
  const bg = "#17304f";
  const border = "#29456b";
  const text = "#dbe5f4";

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <circle cx="48" cy="48" r="46" fill="${bg}" stroke="${border}" stroke-width="2"/>
      <circle cx="48" cy="38" r="14" fill="${text}" opacity="0.9"/>
      <path d="M24 76c4-13 17-20 24-20s20 7 24 20" fill="${text}" opacity="0.9"/>
      <text x="48" y="90" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="${text}">${safeLabel}</text>
    </svg>
  `;

  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function renderSuggestions(data) {
  if (!suggestionsSectionEl || !suggestionsListEl || !suggestionsMetaEl) {
    return;
  }

  suggestionsListEl.innerHTML = "";
  suggestionsMetaEl.textContent = "";
  suggestionsMetaEl.classList.add("hidden");

  const suggestions = data?.suggestions || [];

  if (!suggestions.length) {
    suggestionsSectionEl.classList.remove("hidden");
    return;
  }

  suggestions.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "suggestionCard";

    card.innerHTML = `
  <div class="suggestionHeader">
    <img class="suggestionAvatar" alt="Profile picture for ${item.candidate_id}" src="${avatarDataUrl(item.candidate_id)}" />
    <div class="suggestionHeaderText">
      <div class="suggestionId">${item.candidate_id}</div>
      <div class="suggestionSubtle">Option ${idx + 1}</div>
    </div>
  </div>

  <div class="suggestionRow">
    <span class="suggestionLabel">Predicted for requester</span>
    <span class="suggestionValue">${item.predicted_rating_for_requester} (${renderStars(item.predicted_star_rating_for_requester)})</span>
  </div>

  <div class="suggestionRow">
    <span class="suggestionLabel">Case</span>
    <span class="suggestionValue">${item.case}</span>
  </div>
`;

    suggestionsListEl.appendChild(card);
  });

  suggestionsSectionEl.classList.remove("hidden");
}

predictBtn.addEventListener("click", async () => {
  resultEl.classList.add("hidden");
  graphSectionEl.classList.add("hidden");

  if (suggestionsSectionEl) suggestionsSectionEl.classList.add("hidden");
  if (suggestionsListEl) suggestionsListEl.innerHTML = "";
  if (suggestionsMetaEl) {
    suggestionsMetaEl.textContent = "";
    suggestionsMetaEl.classList.add("hidden");
  }

  statusEl.textContent = "Reading page...";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) throw new Error("No active tab found.");

    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        function cleanText(s) {
          return (s || "").replace(/\s+/g, " ").trim().toLowerCase();
        }

        function isVisible(el) {
          if (!el) return false;
          const style = window.getComputedStyle(el);
          return style.display !== "none" && style.visibility !== "hidden";
        }

        function collectRoots(root, roots = []) {
          roots.push(root);
          const all = root.querySelectorAll ? root.querySelectorAll("*") : [];
          for (const el of all) {
            if (el.shadowRoot) collectRoots(el.shadowRoot, roots);
          }
          return roots;
        }

        function queryAllDeep(selector) {
          const roots = collectRoots(document);
          const out = [];
          for (const root of roots) {
            if (!root.querySelectorAll) continue;
            out.push(...root.querySelectorAll(selector));
          }
          return out;
        }

        function getAllTextNodes(scope) {
          return Array.from(scope.querySelectorAll("*")).filter(el => {
            if (!isVisible(el)) return false;
            const txt = cleanText(el.innerText || el.textContent || "");
            if (!txt) return false;
            return el.children.length === 0 || txt.length < 80;
          });
        }

        function findModalRoot() {
          const candidates = queryAllDeep('[role="dialog"], [aria-modal="true"], div');
          for (const el of candidates) {
            const txt = cleanText(el.innerText || el.textContent || "");
            if (txt.includes("edit request") && txt.includes("recommender email") && txt.includes("my email")) {
              return el;
            }
          }
          return document.body;
        }

        function nextControlAfter(labelEl, scope) {
          const controls = Array.from(scope.querySelectorAll("input, textarea, select")).filter(isVisible);
          if (!controls.length) return null;

          const labelPos = labelEl.compareDocumentPosition.bind(labelEl);
          for (const c of controls) {
            const rel = labelPos(c);
            if (rel & Node.DOCUMENT_POSITION_FOLLOWING) {
              return c;
            }
          }
          return null;
        }

        function findLabelElement(scope, labelText) {
          const needle = cleanText(labelText);
          const textNodes = getAllTextNodes(scope);

          let best = null;
          let bestScore = -1;

          for (const el of textNodes) {
            const txt = cleanText(el.innerText || el.textContent || "");
            let score = -1;
            if (txt === needle || txt === `${needle}:`) score = 100;
            else if (txt.startsWith(needle)) score = 80;
            else if (txt.includes(needle)) score = 50;

            if (score > bestScore) {
              best = el;
              bestScore = score;
            }
          }
          return best;
        }

        function valueFor(scope, labelText) {
          const labelEl = findLabelElement(scope, labelText);
          if (!labelEl) return { found: false, value: "" };

          const control = nextControlAfter(labelEl, scope);
          if (!control) return { found: false, value: "" };

          return {
            found: true,
            value: (control.value || "").trim()
          };
        }

        try {
          const modal = findModalRoot();

          const recommender = valueFor(modal, "Recommender Email");
          const target = valueFor(modal, "Recommend me to");
          const requester = valueFor(modal, "My email");
          const note = valueFor(modal, "Note to Recommender");

          return {
            ok: true,
            requester: requester.value,
            recommender: recommender.value,
            target: target.value,
            note: note.value
          };
        } catch (err) {
          return {
            ok: false,
            error: String(err)
          };
        }
      }
    });

    const data = results?.[0]?.result;
    if (!data) throw new Error("No scrape result returned.");
    if (!data.ok) throw new Error(data.error || "Injected scraper failed.");
    if (!data.requester || !data.recommender) {
      throw new Error("Requester or recommender field missing from page.");
    }

    statusEl.textContent = "Predicting...";

    let recommender_id = data.recommender;
    if (/^u\d+$/i.test(recommender_id)) {
      recommender_id = `rec_${recommender_id.toLowerCase()}@demo.edu`;
    }

    const combinedNote = data.target
      ? `Recommend me to: ${data.target}\nNote: ${data.note || ""}`
      : (data.note || "");

    const predictResponse = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        requester_id: data.requester,
        recommender_id,
        requestor_note: combinedNote
      })
    });

    const predictText = await predictResponse.text();
    if (!predictResponse.ok) {
      throw new Error(`API error ${predictResponse.status}: ${predictText}`);
    }

    const apiData = JSON.parse(predictText);

    ratingValueEl.textContent = apiData.predicted_rating;
    starValueEl.textContent = `${apiData.predicted_star_rating} (${renderStars(apiData.predicted_star_rating)})`;
    caseValueEl.textContent = apiData.case;
    normalizedRequesterEl.textContent = apiData.normalized_requester_id || "";
    normalizedRecommenderEl.textContent = apiData.normalized_recommender_id || "";
    resultEl.classList.remove("hidden");

    statusEl.textContent = "Loading network...";

    const networkResponse = await fetch("http://127.0.0.1:8000/network", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        requester_id: apiData.normalized_requester_id || data.requester,
        recommender_id: apiData.normalized_recommender_id || recommender_id
      })
    });

    const networkText = await networkResponse.text();
    if (!networkResponse.ok) {
      throw new Error(`Network API error ${networkResponse.status}: ${networkText}`);
    }

    const networkData = JSON.parse(networkText);

    drawNetwork(graphCanvas, networkData);
    graphMetaEl.textContent =
      `Nodes: ${networkData.nodes.length} | Edges: ${networkData.edges.length} | Shared neighbors: ${networkData.shared_neighbors_count}`;
    graphSectionEl.classList.remove("hidden");

    statusEl.textContent = "Finding suggestions...";

    const suggestionsResponse = await fetch("http://127.0.0.1:8000/suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        requester_id: apiData.normalized_requester_id || data.requester,
        recommender_id: apiData.normalized_recommender_id || recommender_id,
        requestor_note: combinedNote,
        max_suggestions: 5
      })
    });

    const suggestionsText = await suggestionsResponse.text();
    if (!suggestionsResponse.ok) {
      throw new Error(`Suggestions API error ${suggestionsResponse.status}: ${suggestionsText}`);
    }

    const suggestionsData = JSON.parse(suggestionsText);
    renderSuggestions(suggestionsData);

    statusEl.textContent = "Done.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = String(err);
  }
});