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