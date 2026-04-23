const queryEl = document.getElementById("query");
const systemEl = document.getElementById("system");
const topkEl = document.getElementById("topk");
const searchBtn = document.getElementById("searchBtn");
const clearBtn = document.getElementById("clearBtn");
const resultsEl = document.getElementById("results");
const statusEl = document.getElementById("status");

function renderResults(results) {
  resultsEl.innerHTML = "";

  if (!results.length) {
    resultsEl.innerHTML = "<p>No results found.</p>";
    return;
  }

  results.forEach(item => {
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `
      <div class="result-rank">${item.rank}. Result</div>
      <div class="result-docid">${item.doc_id}</div>
      <div class="result-preview">${item.preview}</div>
    `;
    resultsEl.appendChild(card);
  });
}

async function runSearch() {
  const query = queryEl.value.trim();
  const system = systemEl.value;
  const top_k = parseInt(topkEl.value, 10);

  if (!query) {
    statusEl.textContent = "Please enter a query.";
    resultsEl.innerHTML = "";
    return;
  }

  statusEl.textContent = "Searching...";
  resultsEl.innerHTML = "";

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query, system, top_k })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Search failed.");
    }

    statusEl.textContent = `System: ${data.system} | Top K: ${data.top_k}`;
    renderResults(data.results);
  } catch (error) {
    statusEl.textContent = `Error: ${error.message}`;
    resultsEl.innerHTML = "";
  }
}

searchBtn.addEventListener("click", runSearch);

clearBtn.addEventListener("click", () => {
  queryEl.value = "";
  topkEl.value = 10;
  systemEl.value = "Hybrid (RRF)";
  statusEl.textContent = "";
  resultsEl.innerHTML = "";
});
