"use strict";

const cfg = {
  bRows: 32,
  aCols: 8
};

const dom = {
  sceneRoot: document.getElementById("sceneRoot"),
  sceneTitle: document.getElementById("sceneTitle"),
  sceneSubtitle: document.getElementById("sceneSubtitle"),
  sceneFormula: document.getElementById("sceneFormula"),
  sceneCounter: document.getElementById("sceneCounter"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  playBtn: document.getElementById("playBtn"),
  speed: document.getElementById("speed"),
  speedLabel: document.getElementById("speedLabel")
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function colorForRow(rowIndex) {
  const hue = Math.floor((rowIndex / cfg.bRows) * 340);
  return `hsl(${hue} 78% 48%)`;
}

function addLegend(container) {
  const legend = el("div", "legend");
  const items = [
    ["#6ee7ff", "Columns are a = 0..7"],
    ["#9ef7c9", "Rows are b = 0..31"],
    ["#ffd166", "After XOR: color follows b' = b xor a"]
  ];
  for (const [color, text] of items) {
    const chip = el("div", "chip");
    const dot = el("span", "dot");
    dot.style.background = color;
    chip.append(dot, document.createTextNode(text));
    legend.append(chip);
  }
  container.append(legend);
}

function renderXorGrid(applyXor) {
  const wrap = el("div", "stepLayout");
  const panel = el("div", "panel");
  panel.append(el("div", "panelTitle", applyXor ? "After XOR" : "Before XOR"));

  const box = el("div", "gridBox");
  const matrix = el("div", "matrix xorMatrix");
  matrix.style.gridTemplateColumns = "36px repeat(8, 24px)";

  matrix.append(el("div", "axisLabel"));
  for (let a = 0; a < cfg.aCols; a += 1) {
    matrix.append(el("div", "axisLabel", `a=${a}`));
  }

  for (let b = 0; b < cfg.bRows; b += 1) {
    matrix.append(el("div", "axisLabel", `b=${b}`));
    for (let a = 0; a < cfg.aCols; a += 1) {
      const bx = b ^ a;
      const rowColor = applyXor ? colorForRow(bx) : colorForRow(b);
      const cell = el("div", "cell xorCell");
      cell.style.background = rowColor;
      cell.title = applyXor
        ? `a=${a}, b=${b} -> b'=${bx}`
        : `a=${a}, b=${b} -> b'=${bx} (not applied yet)`;
      if (b === 9 && a === 5) {
        cell.classList.add("highlight");
      }
      matrix.append(cell);
    }
  }

  box.append(matrix);
  panel.append(
    el(
      "div",
      "note",
      applyXor
        ? "XOR applied: each cell color is based on b' = b xor a."
        : "Before XOR: each row keeps a single color based on b."
    )
  );
  wrap.append(panel);
  return wrap;
}

let phase = 0;
let timer = null;

function renderPhase(nextPhase) {
  phase = nextPhase <= 0 ? 0 : 1;
  const isApplied = phase === 1;
  dom.sceneCounter.textContent = isApplied ? "XOR Applied" : "Before XOR";
  dom.sceneTitle.textContent = "Single XOR Transform on One Grid (B x A = 32 x 8)";
  dom.sceneSubtitle.textContent = isApplied
    ? "After applying XOR: each cell uses row color from b' = b xor a."
    : "Start state: each row is colored by b only (same color across columns).";
  dom.sceneFormula.textContent = "Transform: (a, b) -> (a, b xor a)\nExample highlight: (a=5,b=9) -> b'=12";

  dom.sceneRoot.classList.add("fadeOut");
  window.setTimeout(() => {
    dom.sceneRoot.innerHTML = "";
    const root = el("div", "stepLayout");
    addLegend(root);
    root.append(renderXorGrid(isApplied));
    dom.sceneRoot.append(root);
    dom.sceneRoot.classList.remove("fadeOut");
  }, 220);
}

function stopPlay() {
  if (timer !== null) {
    clearInterval(timer);
    timer = null;
    dom.playBtn.textContent = "Play";
  }
}

function startPlay() {
  stopPlay();
  const speed = parseFloat(dom.speed.value);
  const interval = Math.max(800, Math.floor(1800 / speed));
  timer = setInterval(() => renderPhase(phase === 0 ? 1 : 0), interval);
  dom.playBtn.textContent = "Pause";
}

dom.prevBtn.addEventListener("click", () => {
  stopPlay();
  renderPhase(0);
});

dom.nextBtn.addEventListener("click", () => {
  stopPlay();
  renderPhase(1);
});

dom.playBtn.addEventListener("click", () => {
  if (timer === null) startPlay();
  else stopPlay();
});

dom.speed.addEventListener("input", () => {
  dom.speedLabel.textContent = `${parseFloat(dom.speed.value).toFixed(1)}x`;
  if (timer !== null) startPlay();
});

renderPhase(0);
