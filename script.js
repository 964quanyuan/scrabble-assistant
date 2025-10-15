// simple static roster — you can also dynamically fetch a JSON index.
const videos = [
  { 
    src: "videos/scrabble_demo_basic.webm", 
    title: "Basic Demo", 
    desc: "A short demonstration of the Scrabble Assistant's core move-finding logic in action.\n<span class='highlight'>(▀̿Ĺ̯▀̿ ̿)</span>"
  },
  { 
    src: "videos/scrabble_bingo.webm", 
    title: "Bingo Swarm", 
    desc: "Notice the model slipping into a coma briefly when faced with a plethora of high-scoring bingo opportunities.\n<span class='highlight'>༼ʘ̚ل͜ʘ̚༽</span>"
  },
  { 
    src: "videos/scrabble_demo_invalid.webm", 
    title: "Bruh Moments", 
    desc: "Error-handling and invalid move detection under edge-case board states.\n<span class='highlight'>ლ(ಠ益ಠლ)</span>"
  },
  { 
    src: "videos/scrabble_oxy.webm", 
    title: "PUSHING THE LIMITS", 
    desc: "Stress test the model on extreme-scoring words and extreme board density.\n<span class='highlight'>ಥ_ಥ</span>"
  },
];

const gallery = document.getElementById('gallery');
videos.forEach(v => {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <h3>${v.title}</h3>
    <div class="video-row">
      <video controls preload="metadata" playsinline>
        <source src="${v.src}" type="video/webm">
        Your browser doesn't support WebM.
      </video>
      <p class="desc">${v.desc}</p>
    </div>
  `;
  gallery.appendChild(card);
});