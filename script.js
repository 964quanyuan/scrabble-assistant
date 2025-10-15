// simple static roster — you can also dynamically fetch a JSON index.
const videos = [
  { src: "videos/scrabble_demo_basic.webm", title: "Basic Demo" },
  { src: "videos/scrabble_bingo.webm", title: "Bingo Swarm" },
  { src: "videos/scrabble_demo_invalid.webm", title: "Bruh Moments" },
  { src: "videos/scrabble_oxy.webm", title: "PUSHING THE LIMITS" },
];

const gallery = document.getElementById('gallery');
videos.forEach(v => {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <h3>${v.title}</h3>
    <video controls preload="metadata" playsinline>
      <source src="${v.src}" type="video/webm">
      Your browser doesn't support WebM.
    </video>
  `;
  gallery.appendChild(card);
});