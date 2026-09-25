// history.js — search, filter, sort, clear scan history

const searchInput = document.getElementById("search-input");
const filterSelect = document.getElementById("filter-select");
const sortSelect = document.getElementById("sort-select");
const clearBtn = document.getElementById("clear-btn");
const body = document.getElementById("history-body");
const emptyState = document.getElementById("empty-state");

let debounceTimer;
searchInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(loadHistory, 300);
});
filterSelect.addEventListener("change", loadHistory);
sortSelect.addEventListener("change", loadHistory);
clearBtn.addEventListener("click", async () => {
  if (!confirm("Clear all scan history? This cannot be undone.")) return;
  await fetch("/api/history", { method: "DELETE" });
  loadHistory();
});

async function loadHistory() {
  const [sortBy, sortDir] = sortSelect.value.split("-");
  const params = new URLSearchParams({
    search: searchInput.value.trim(),
    prediction: filterSelect.value,
    sort_by: sortBy,
    sort_dir: sortDir,
  });

  const res = await fetch(`/api/history?${params.toString()}`);
  const data = await res.json();

  if (!data.scans.length) {
    body.innerHTML = "";
    emptyState.style.display = "block";
    return;
  }
  emptyState.style.display = "none";

  body.innerHTML = data.scans.map(s => `
    <tr>
      <td class="url-cell" title="${s.url}">${s.url}</td>
      <td>${predictionBadge(s.prediction)}</td>
      <td>${s.risk_score}/100</td>
      <td>${s.confidence}%</td>
      <td>${s.model_used}</td>
      <td>${fmtDate(s.scanned_at)}</td>
    </tr>
  `).join("");
}

loadHistory();
