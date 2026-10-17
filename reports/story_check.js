
(function(){
  // ---------- wave background ----------
  const canvas = document.getElementById('wave-canvas');
  const ctx = canvas.getContext('2d');
  let w, h, t = 0;
  function resize(){ w = canvas.width = canvas.offsetWidth; h = canvas.height = canvas.offsetHeight; }
  window.addEventListener('resize', resize); resize();
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function drawWaves(){
    ctx.clearRect(0,0,w,h);
    const layers = [
      {amp: 18, freq: 0.012, speed: 0.012, y: h*0.62, color: 'rgba(82,209,195,0.10)'},
      {amp: 26, freq: 0.008, speed: 0.008, y: h*0.72, color: 'rgba(217,167,92,0.08)'},
      {amp: 14, freq: 0.018, speed: 0.02,  y: h*0.84, color: 'rgba(132,160,189,0.10)'},
    ];
    layers.forEach(L => {
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let x = 0; x <= w; x += 6) {
        const y = L.y + Math.sin(x*L.freq + t*L.speed) * L.amp;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h); ctx.closePath();
      ctx.fillStyle = L.color; ctx.fill();
    });
    t += 1;
    if (!reduceMotion) requestAnimationFrame(drawWaves);
  }
  drawWaves();

  // ---------- counters ----------
  function animateCount(el){
    const target = parseFloat(el.dataset.count);
    const prefix = el.dataset.prefix || '';
    const suffix = el.dataset.suffix || '';
    const dur = 1600; const start = performance.now();
    function step(now){
      const p = Math.min(1, (now-start)/dur);
      const eased = 1 - Math.pow(1-p, 3);
      const v = Math.round(target * eased);
      el.textContent = prefix + v.toLocaleString() + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ---------- reveal + scenario triggers ----------
  const dowData = [
    {d:'Mon', v:6365129, wk:false},{d:'Tue', v:6621687, wk:false},{d:'Wed', v:5909602, wk:false},
    {d:'Thu', v:7881394, wk:false},{d:'Fri', v:27773156, wk:true},{d:'Sat', v:31689120, wk:true},
    {d:'Sun', v:4644800, wk:false},
  ];
  const dowChart = document.getElementById('dow-chart');
  const maxDow = Math.max(...dowData.map(d=>d.v));
  dowData.forEach(d => {
    const wrap = document.createElement('div');
    wrap.className = 'dow-bar-wrap' + (d.wk ? ' weekend' : '');
    wrap.innerHTML = `<div class="dow-val">$${(d.v/1e6).toFixed(1)}M</div><div class="dow-bar" data-h="${(d.v/maxDow*100).toFixed(1)}"></div><div class="dow-label">${d.d}</div>`;
    dowChart.appendChild(wrap);
  });

  const scenarios = [
    {n:'Payday ×2', ok:true}, {n:'Holiday shopping ×3', ok:false}, {n:'Bond shock +200bp', ok:true},
    {n:'Equity crash -35%', ok:true}, {n:'Haircut doubling', ok:true}, {n:'SVB-style run', ok:false},
  ];
  const pills = document.getElementById('scenario-pills');
  scenarios.forEach(s => {
    const p = document.createElement('span');
    p.className = 'pill ' + (s.ok ? 'ok' : 'fail');
    p.textContent = (s.ok ? '✓ ' : '✗ ') + s.n;
    pills.appendChild(p);
  });

  const scoreData = [
    {cls:'allcash', name:'Never sweep', val:41330803},
    {cls:'naive', name:'Fixed $38M cushion', val:20377549},
    {cls:'ballast', name:'Ballast (adaptive)', val:20349462},
    {cls:'oracle', name:'Perfect foresight (theoretical)', val:19751669},
  ];
  const scoreRows = document.getElementById('score-rows');
  const maxScore = Math.max(...scoreData.map(s=>s.val));
  scoreData.forEach(s => {
    const row = document.createElement('div');
    row.className = 'score-row ' + s.cls;
    row.innerHTML = `<div class="meta"><span class="name">${s.name}</span><span class="val num">$${s.val.toLocaleString()}/yr</span></div>
      <div class="score-track"><div class="score-fill" data-w="${(s.val/maxScore*100).toFixed(1)}"></div></div>`;
    scoreRows.appendChild(row);
  });

  const stormTiers = document.querySelectorAll('.storm-tier');
  let stormPlayed = false;
  function playStorm(){
    if (stormPlayed) return; stormPlayed = true;
    stormTiers.forEach((tier, i) => {
      setTimeout(() => {
        tier.classList.add('active');
        if (i < stormTiers.length - 1) {
          setTimeout(() => tier.classList.add('draining'), 500);
        }
      }, i * 650);
    });
  }

  let clockPlayed = false;
  function playClock(){
    if (clockPlayed) return; clockPlayed = true;
    const hand = document.getElementById('clock-hand');
    const steps = document.querySelectorAll('.flow-step');
    const angles = [360*(17/24), 360*(22/24), 360*(32/24)]; // 17:00, 22:00, 08:00(+24)
    angles.forEach((ang, i) => {
      setTimeout(() => {
        hand.style.transform = `translateX(-50%) rotate(${ang}deg)`;
        steps.forEach(s => s.classList.remove('active'));
        steps[i].classList.add('active');
      }, i * 1500 + 300);
    });
  }

  let tanksPlayed = false;
  function playTanks(){
    if (tanksPlayed) return; tanksPlayed = true;
    document.getElementById('fill-cash').style.height = '38%';
    document.getElementById('fill-vault').style.height = '82%';
  }

  const bars = document.querySelectorAll('.dow-bar');
  let barsPlayed = false;
  function playBars(){
    if (barsPlayed) return; barsPlayed = true;
    bars.forEach(b => b.style.height = b.dataset.h + '%');
  }

  const fills = document.querySelectorAll('.score-fill');
  let scorePlayed = false;
  function playScore(){
    if (scorePlayed) return; scorePlayed = true;
    fills.forEach(f => f.style.width = f.dataset.w + '%');
  }

  const counters = document.querySelectorAll('[data-count]');
  let countersPlayed = false;
  function playCounters(){
    if (countersPlayed) return; countersPlayed = true;
    counters.forEach(animateCount);
  }

  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('in');
        if (e.target.closest('#storm')) playStorm();
        if (e.target.closest('#nightcycle')) playClock();
        if (e.target.closest('#problem')) playTanks();
        if (e.target.closest('#discovery')) playBars();
        if (e.target.closest('#scoreboard')) playScore();
      }
    });
  }, {threshold: 0.3});
  document.querySelectorAll('.reveal').forEach(el => io.observe(el));

  setTimeout(playCounters, 400);
})();
