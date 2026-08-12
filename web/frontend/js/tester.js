let selectedModelId = null;

async function loadTrainedModels() {
  const select = document.getElementById("model-select");
  const emptyState = document.getElementById("tester-empty-state");
  const main = document.getElementById("tester-main");

  const res = await fetch("/api/models");
  const models = await res.json();
  const trained = models.filter((m) => m.status === "trained");

  if (trained.length === 0) {
    emptyState.style.display = "";
    main.style.display = "none";
    return false;
  }

  emptyState.style.display = "none";
  main.style.display = "";

  select.innerHTML = "";
  for (const m of trained) {
    const opt = document.createElement("option");
    opt.value = m.model_id;
    const f1 = typeof m.val_f1 === "number" ? m.val_f1.toFixed(4) : "?";
    opt.textContent = `${m.name} (val_f1=${f1})`;
    select.appendChild(opt);
  }
  selectedModelId = select.value;
  return true;
}

async function randomizeDigits() {
  const grid = document.getElementById("digit-grid");
  grid.innerHTML = '<p class="placeholder-note">Loading…</p>';

  const res = await fetch("/api/tester/sample");
  const samples = await res.json();

  grid.innerHTML = "";
  for (const s of samples) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "digit-card";

    const img = document.createElement("img");
    img.src = `data:image/png;base64,${s.pixels}`;
    img.alt = "handwritten digit";
    card.appendChild(img);

    const resultLabel = document.createElement("span");
    resultLabel.className = "digit-result";
    card.appendChild(resultLabel);

    card.addEventListener("click", () => predictDigit(s.sample_id, card, resultLabel));
    grid.appendChild(card);
  }
}

async function predictDigit(sampleId, card, resultLabel) {
  if (!selectedModelId) return;
  resultLabel.textContent = "…";
  try {
    const res = await fetch("/api/tester/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: selectedModelId, sample_id: sampleId }),
    });
    const data = await res.json();
    if (!res.ok) {
      resultLabel.textContent = "error";
      return;
    }
    resultLabel.textContent = `→ ${data.predicted_digit} (true: ${data.true_digit})`;
    flashFeedback(card, data.correct);
  } catch (err) {
    resultLabel.textContent = "error";
  }
}

function flashFeedback(card, correct) {
  card.classList.remove("flash-good", "flash-bad");
  void card.offsetWidth; // restart CSS animation on repeated clicks
  card.classList.add(correct ? "flash-good" : "flash-bad");
  setTimeout(() => card.classList.remove("flash-good", "flash-bad"), 800);

  document.body.classList.remove("page-flash-good", "page-flash-bad");
  void document.body.offsetWidth;
  document.body.classList.add(correct ? "page-flash-good" : "page-flash-bad");
  setTimeout(() => document.body.classList.remove("page-flash-good", "page-flash-bad"), 800);
}

document.getElementById("model-select").addEventListener("change", (ev) => {
  selectedModelId = ev.target.value;
});
document.getElementById("randomize-btn").addEventListener("click", randomizeDigits);

loadTrainedModels().then((hasModels) => {
  if (hasModels) randomizeDigits();
});
