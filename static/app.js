

const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: false,
};


const choice = { option_type: "call", exercise: "european" };


document.querySelectorAll(".segmented").forEach((group) => {
  const key = group.dataset.group;
  group.querySelectorAll(".seg").forEach((btn) => {
    btn.addEventListener("click", () => {
      group.querySelectorAll(".seg").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      choice[key] = btn.dataset.value;
    });
  });
});


const $ = (id) => document.getElementById(id);
const fmt = (x, dp = 4) =>
  Number.isFinite(x) ? x.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp }) : "—";

function showLoading(on) {
  $("loading").classList.toggle("hidden", !on);
}

function typesetMath() {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
}


function render(data) {
  
  $("binom-price").textContent = fmt(data.binom_price);
  $("bs-price").textContent = fmt(data.bs_price);
  $("abs-diff").textContent = fmt(data.abs_diff, 5);
  $("card-N").textContent = `(N = ${$("N").value})`;

  
  const g = data.greeks;
  $("g-delta").textContent = fmt(g.delta, 4);
  $("g-gamma").textContent = fmt(g.gamma, 5);
  $("g-vega").textContent = fmt(g.vega, 4);
  $("g-theta").textContent = fmt(g.theta, 4);
  $("g-rho").textContent = fmt(g.rho, 4);

  
  const rn = data.risk_neutral;
  $("rn-N").textContent = rn.draw_N;
  $("rn-dt").textContent = fmt(rn.dt, 5);
  $("rn-u").textContent = fmt(rn.u, 5);
  $("rn-d").textContent = fmt(rn.d, 5);
  $("rn-p").textContent = fmt(rn.p, 5);
  $("rn-disc").textContent = fmt(rn.disc_step, 5);
  $("rn-disct").textContent = fmt(rn.disc_total, 5);
  $("rn-exp").textContent = fmt(rn.expected_payoff, 4);
  $("rn-dexp").textContent = fmt(rn.discounted_expected_payoff, 4);

  
  Plotly.react("lattice", data.lattice_fig.data, data.lattice_fig.layout, PLOTLY_CONFIG);
  Plotly.react("convergence", data.conv_fig.data, data.conv_fig.layout, PLOTLY_CONFIG);

  typesetMath();
}


async function priceOption() {
  const payload = {
    S: $("S").value,
    K: $("K").value,
    r: $("r").value,
    sigma: $("sigma").value,
    T: $("T").value,
    N: $("N").value,
    n_max: $("n_max").value,
    option_type: choice.option_type,
    exercise: choice.exercise,
  };

  $("err").textContent = "";
  showLoading(true);

  try {
    const resp = await fetch("/api/price", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `Request failed (${resp.status})`);
    }
    render(data);
  } catch (e) {
    $("err").textContent = e.message;
  } finally {
    showLoading(false);
  }
}


$("pricer-form").addEventListener("submit", (e) => {
  e.preventDefault();
  priceOption();
});


window.addEventListener("load", priceOption);


async function postJSON(url, payload) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || `Request failed (${resp.status})`);
  }
  return data;
}

function renderSmileTable(rows) {
  const body = $("smile-table-body");
  body.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const ivText = row.iv !== null ? `${(row.iv * 100).toFixed(2)}%` : "—";
    tr.innerHTML = `
      <td>${fmt(row.strike, 2)}</td>
      <td>${fmt(row.market_price, 2)}</td>
      <td>${row.price_source}</td>
      <td>${ivText}</td>
      <td>${row.iv !== null ? row.method : "failed"}</td>
    `;
    body.appendChild(tr);
  });
}

async function fetchSmile(ticker, expiry) {
  const payload = {
    ticker,
    expiry,
    option_type: choice.option_type,
    r: $("r").value,
  };
  const data = await postJSON("/api/market/smile", payload);

  $("mkt-quality").textContent = `${data.n_used} / ${data.n_dropped} / ${data.n_iv_failed}`;
  Plotly.react("smile", data.fig.data, data.fig.layout, PLOTLY_CONFIG);
  renderSmileTable(data.rows);
}

async function loadLiveData() {
  const ticker = $("mkt-ticker").value.trim().toUpperCase();
  if (!ticker) return;

  $("mkt-err").textContent = "";
  showLoading(true);
  try {
    
    const stats = await postJSON("/api/market/stats", { ticker });
    $("S").value = stats.spot.toFixed(2);
    $("sigma").value = stats.hist_vol.toFixed(4);
    $("mkt-spot").textContent = fmt(stats.spot, 2);
    $("mkt-histvol").textContent = `${(stats.hist_vol * 100).toFixed(2)}%`;
    $("mkt-asof").textContent = stats.as_of;

    
    const expData = await postJSON("/api/market/expirations", { ticker });
    const select = $("mkt-expiry");
    select.innerHTML = "";
    expData.expirations.forEach((exp) => {
      const opt = document.createElement("option");
      opt.value = exp;
      opt.textContent = exp;
      select.appendChild(opt);
    });
    select.disabled = false;

    
    await fetchSmile(ticker, expData.expirations[0]);

    
    await priceOption();
  } catch (e) {
    $("mkt-err").textContent = e.message;
  } finally {
    showLoading(false);
  }
}

$("mkt-fetch-btn").addEventListener("click", loadLiveData);



$("mkt-expiry").addEventListener("change", async () => {
  const ticker = $("mkt-ticker").value.trim().toUpperCase();
  const expiry = $("mkt-expiry").value;
  if (!ticker || !expiry) return;
  $("mkt-err").textContent = "";
  showLoading(true);
  try {
    await fetchSmile(ticker, expiry);
  } catch (e) {
    $("mkt-err").textContent = e.message;
  } finally {
    showLoading(false);
  }
});



document.querySelectorAll('.segmented[data-group="option_type"] .seg').forEach((btn) => {
  btn.addEventListener("click", async () => {
    const ticker = $("mkt-ticker").value.trim().toUpperCase();
    const expiry = $("mkt-expiry").value;
    if (!ticker || !expiry || $("mkt-expiry").disabled) return;
    try {
      await fetchSmile(ticker, expiry);
    } catch (e) {
      $("mkt-err").textContent = e.message;
    }
  });
});
