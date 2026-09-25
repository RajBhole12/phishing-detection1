// dashboard.js — populates the dashboard stats, charts, and recent scans table

async function loadDashboard() {
  try {
    const res = await fetch("/api/statistics");
    const stats = await res.json();

    document.getElementById("stat-total").textContent = stats.total_scans;
    document.getElementById("stat-phishing").textContent = stats.phishing_detected;
    document.getElementById("stat-suspicious").textContent = stats.suspicious_urls;
    document.getElementById("stat-legit").textContent = stats.legitimate_urls;

    if (!stats.models_trained) {
      document.getElementById("dash-alert").innerHTML =
        `<div class="alert alert-warn">Models have not been trained yet. Visit <a href="/training" style="text-decoration:underline;">AI Model / Training</a> to get started.</div>`;
    }

    renderDistributionChart(stats);
    renderActivityChart(stats.activity_over_time);
    renderRecentScans(stats.recent_scans);
  } catch (err) {
    console.error(err);
  }
}

function renderDistributionChart(stats) {
  const ctx = document.getElementById("chart-distribution");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Legitimate", "Suspicious", "Phishing"],
      datasets: [{
        data: [stats.legitimate_urls, stats.suspicious_urls, stats.phishing_detected],
        backgroundColor: [CHART_COLORS.safe, CHART_COLORS.warn, CHART_COLORS.danger],
        borderRadius: 6,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: CHART_COLORS.grid } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderActivityChart(activity) {
  const ctx = document.getElementById("chart-activity");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: activity.map(a => a.day),
      datasets: [{
        label: "Scans",
        data: activity.map(a => a.count),
        borderColor: CHART_COLORS.accent,
        backgroundColor: "rgba(79,140,255,0.12)",
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: CHART_COLORS.grid } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderRecentScans(scans) {
  const body = document.getElementById("recent-scans-body");
  if (!scans.length) {
    body.innerHTML = `<tr><td colspan="5" style="color:var(--text-faint);">No scans yet — try the URL Scanner.</td></tr>`;
    return;
  }
  body.innerHTML = scans.map(s => `
    <tr>
      <td class="url-cell" title="${s.url}">${s.url}</td>
      <td>${predictionBadge(s.prediction)}</td>
      <td>${s.risk_score}/100</td>
      <td>${s.model_used}</td>
      <td>${fmtDate(s.scanned_at)}</td>
    </tr>
  `).join("");
}

loadDashboard();
