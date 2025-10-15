// simple static roster — you can also dynamically fetch a JSON index.
const videos = [
  { src: "videos/demo1_trimmed.webm", title: "Move Finder" },
  { src: "videos/demo2_trimmed.webm", title: "Rack Analyzer" },
  // add more
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
