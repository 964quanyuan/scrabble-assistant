// simple static roster — you can also dynamically fetch a JSON index.
const videos = [
  { 
    src: "videos/scrabble_demo_basic.webm", 
    title: "Basic Demo", 
    desc: "A short demonstration of the Scrabble Assistant's core move-finding logic in action.",
    face: "(▀̿Ĺ̯▀̿ ̿)"
  },
  { 
    src: "videos/scrabble_bingo.webm", 
    title: "Bingo Swarm", 
    desc: "Notice the model slipping into a coma briefly when faced with a plethora of high-scoring bingo opportunities.",
    face: "༼ʘ̚ل͜ʘ̚༽"
  },
  { 
    src: "videos/scrabble_demo_invalid.webm", 
    title: "Bruh Moments", 
    desc: "Error-handling and invalid move detection under edge-case board states.",
    face: "ლ(ಠ益ಠლ)"
  },
  { 
    src: "videos/scrabble_oxy.webm", 
    title: "Pushing the Limits", 
    desc: "Stress test the model on extreme-scoring words and extreme board density.",
    face: "ಥ_ಥ"
  },
];

const gallery = document.getElementById('gallery');

// --- artifacts data (3 files) ---
const artifacts = [
  { src: 'images/kpt.gif', type: 'gif', title: 'KPT Artifact' },
  { src: 'images/ocr.gif', type: 'image', title: 'OCR Artifact' },
  { src: 'images/solu.gif', type: 'gif', title: 'Solution Artifact' },
];

// helper to build the artifacts card (scrambled stack)
function makeArtifactsCard() {
  const card = document.createElement('div');
  card.className = 'card artifacts-card';
  card.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;">
      <div class="artifacts-thumb" id="artifacts-thumb" aria-hidden="false">
        <img src="${artifacts[0].src}" alt="${artifacts[0].title}" />
        <img src="${artifacts[1].src}" alt="${artifacts[1].title}" />
        <img src="${artifacts[2].src}" alt="${artifacts[2].title}" />
      </div>
      <div class="artifacts-title">Scrambled Artifacts — click to open gallery</div>
    </div>
  `;
  // click opens modal at index 0
  card.querySelector('#artifacts-thumb').addEventListener('click', () => openArtifactGallery(0));
  return card;
}

// insert artifacts card at left column (fall back to gallery if missing)
const leftArtifacts = document.getElementById('artifacts-container');
if (leftArtifacts) {
  leftArtifacts.appendChild(makeArtifactsCard());
} else {
  // fallback: put it at the top of the gallery
  gallery.insertBefore(makeArtifactsCard(), gallery.firstChild);
}

// existing video cards rendering (unchanged)
videos.forEach(v => {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <div class="video-row">
      <video controls preload="metadata" playsinline>
        <source src="${v.src}" type="video/webm">
        Your browser doesn't support WebM.
      </video>
      <div class="text-col">
        <h3>${v.title}</h3>
        <p class="desc">${v.desc}</p>
        <p class="face">${v.face}</p>
      </div>
    </div>
  `;
  gallery.appendChild(card);
});

// --- artifact modal creation + gallery logic ---
let artifactIndex = 0;
const modal = document.createElement('div');
modal.className = 'artifact-modal';
modal.innerHTML = `
  <button class="artifact-close" aria-label="Close">✕</button>
  <div class="artifact-viewer" role="dialog" aria-modal="true">
    <button class="artifact-nav prev" aria-label="Previous">◀</button>
    <div class="artifact-container" style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;"></div>
    <button class="artifact-nav next" aria-label="Next">▶</button>
  </div>
  <div class="artifact-caption" id="artifact-caption"></div>
`;
document.body.appendChild(modal);

const container = modal.querySelector('.artifact-container');
const captionEl = modal.querySelector('#artifact-caption');
const closeBtn = modal.querySelector('.artifact-close');
const prevBtn = modal.querySelector('.artifact-nav.prev');
const nextBtn = modal.querySelector('.artifact-nav.next');

function renderArtifact(idx) {
  container.innerHTML = '';
  const item = artifacts[idx];
  captionEl.textContent = item.title || '';
  if (item.type === 'gif' || item.type === 'image') {
    // use <img> for both gif and png for simplicity; gifs will animate
    const el = document.createElement('img');
    el.src = item.src;
    el.alt = item.title || '';
    container.appendChild(el);
  } else {
    const el = document.createElement('img');
    el.src = item.src;
    el.alt = item.title || '';
    container.appendChild(el);
  }
}

function openArtifactGallery(start = 0) {
  artifactIndex = start;
  renderArtifact(artifactIndex);
  modal.classList.add('open');
  // trap focus minimally
  closeBtn.focus();
  document.addEventListener('keydown', onKeyDown);
}

function closeArtifactGallery() {
  modal.classList.remove('open');
  document.removeEventListener('keydown', onKeyDown);
}

function onKeyDown(e) {
  if (!modal.classList.contains('open')) return;
  if (e.key === 'Escape') closeArtifactGallery();
  if (e.key === 'ArrowLeft') showPrev();
  if (e.key === 'ArrowRight') showNext();
}

function showPrev() {
  artifactIndex = (artifactIndex - 1 + artifacts.length) % artifacts.length;
  renderArtifact(artifactIndex);
}
function showNext() {
  artifactIndex = (artifactIndex + 1) % artifacts.length;
  renderArtifact(artifactIndex);
}

prevBtn.addEventListener('click', showPrev);
nextBtn.addEventListener('click', showNext);
closeBtn.addEventListener('click', closeArtifactGallery);
modal.addEventListener('click', (ev) => {
  if (ev.target === modal) closeArtifactGallery(); // click backdrop to close
});