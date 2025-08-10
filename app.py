import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Zuma by.Tara", layout="wide", initial_sidebar_state="collapsed")

# Hide Streamlit header and footer for a more app-like fullscreen feel
hide_streamlit_style = """
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .css-18e3th9 {padding-top: 0rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("<div style='display:flex;align-items:center;justify-content:space-between'><h2 style='margin:6px 0'>Zuma — by.Tara</h2><div style='font-size:12px;opacity:0.8'>Tap • Drag • Shoot</div></div>", unsafe_allow_html=True)

# Embed HTML/JS game (responsive and mobile/fullscreen friendly)
html_code = r"""
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Zuma by.Tara</title>
<style>
  :root{
    --bg1:#071428; --bg2:#0f1b33; --accent:#ffd166;
  }
  html,body{height:100%;margin:0;background:linear-gradient(180deg,var(--bg1),var(--bg2));-webkit-font-smoothing:antialiased;-webkit-touch-callout:none;-webkit-user-select:none;user-select:none;}
  .app{display:flex;flex-direction:column;align-items:center;padding:8px;gap:6px;}
  .topbar{width:100%;max-width:420px;display:flex;justify-content:space-between;align-items:center;color:#fff;padding:8px;border-radius:10px}
  .canvas-wrap{position:relative;width:100%;max-width:420px;height:85vh;display:flex;align-items:center;justify-content:center}
  canvas{border-radius:18px;display:block; touch-action:none; width:100%; height:100%; box-shadow: 0 18px 48px rgba(0,0,0,0.6); background: radial-gradient(circle at 40% 18%, rgba(255,255,255,0.02), transparent 12%), linear-gradient(180deg, rgba(255,255,255,0.01), transparent);}
  .hud{position:absolute;left:12px;top:10px;display:flex;gap:8px;align-items:center;color:#fff;font-weight:700}
  .btns{position:absolute;right:12px;top:10px;display:flex;gap:8px}
  .ghost{background:rgba(255,255,255,0.04);padding:6px 10px;border-radius:999px;font-size:13px}
  .brand{position:absolute;left:50%;transform:translateX(-50%);bottom:14px;padding:6px 10px;border-radius:999px;background:rgba(0,0,0,0.28);color:rgba(255,255,255,0.88);font-weight:700;font-size:13px}
  .fullscreen-hint{position:absolute;left:12px;bottom:12px;color:rgba(255,255,255,0.6);font-size:12px}
  .overlay{position:absolute;left:0;top:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;pointer-events:none}
  .banner{pointer-events:auto;background:linear-gradient(90deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02));padding:12px 18px;border-radius:14px;font-weight:800;font-size:20px;color:#fff;box-shadow:0 12px 36px rgba(0,0,0,0.5);opacity:0;transform:translateY(-6px);transition:all .45s ease}
  .banner.show{opacity:1;transform:translateY(0)}
  /* Fullscreen styling hints */
  @media (max-width:420px){
    body{padding:6px;}
    .canvas-wrap{height:92vh}
  }
</style>
</head>
<body>
<div class="app">
  <div class="topbar" style="max-width:420px;">
    <div style="display:flex;align-items:center;gap:8px"><div style="width:10px;height:10px;border-radius:50%;background:var(--accent)"></div><div style="font-weight:800">Zuma</div></div>
    <div style="opacity:0.85;font-size:13px">by.Tara</div>
  </div>

  <div class="canvas-wrap" id="wrap">
    <canvas id="game" width="420" height="780"></canvas>
    <div class="hud" id="hud"><div class="ghost">Score: <span id="score">0</span></div><div class="ghost" id="levelInfo">Level 1</div></div>
    <div class="btns"><button id="reset" style="background:var(--accent);border:0;padding:8px 10px;border-radius:10px;font-weight:700">Reset</button></div>
    <div class="brand">by.Tara</div>
    <div class="fullscreen-hint">Tip: Add to Home Screen for app-like fullscreen</div>
    <div class="overlay"><div id="banner" class="banner">Level 1</div></div>
  </div>
</div>

<script>
// Full polished JS (optimized, with audio and particles). Similar to previous examples but with branding and fullscreen-friendly tuning.
(()=>{
  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');
  const scoreEl = document.getElementById('score');
  const levelInfo = document.getElementById('levelInfo');
  const banner = document.getElementById('banner');
  const resetBtn = document.getElementById('reset');
  const wrap = document.getElementById('wrap');

  // Responsive fit, keep aspect and fill container
  function fitCanvas(){
    const rect = wrap.getBoundingClientRect();
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
  }
  window.addEventListener('resize', fitCanvas);
  fitCanvas();

  // Audio (WebAudio)
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  let audioCtx = null;
  function ensureAudio(){ if (!audioCtx) audioCtx = new AudioCtx(); return audioCtx; }
  function playShoot(){ try{ const c=ensureAudio(); const o=c.createOscillator(); const g=c.createGain(); o.type='sine'; o.frequency.value=820; g.gain.setValueAtTime(0.0001,c.currentTime); g.gain.exponentialRampToValueAtTime(0.12,c.currentTime+0.01); g.gain.exponentialRampToValueAtTime(0.001,c.currentTime+0.22); o.connect(g); g.connect(c.destination); o.start(); o.stop(c.currentTime+0.24);}catch(e){} }
  function playPop(){ try{ const c=ensureAudio(); const bufferSize=0.18*c.sampleRate; const buf=c.createBuffer(1,bufferSize,c.sampleRate); const data=buf.getChannelData(0); for(let i=0;i<data.length;i++) data[i]=(Math.random()*2-1)*Math.exp(-3*i/data.length); const src=c.createBufferSource(); const g=c.createGain(); src.buffer=buf; g.gain.setValueAtTime(0.18,c.currentTime); g.gain.exponentialRampToValueAtTime(0.001,c.currentTime+0.18); src.connect(g); g.connect(c.destination); src.start(); src.stop(c.currentTime+0.18);}catch(e){} }
  function playLevelUp(){ try{ const c=ensureAudio(); const o1=c.createOscillator(), o2=c.createOscillator(), g=c.createGain(); o1.type='triangle'; o2.type='sine'; o1.frequency.value=460; o2.frequency.value=640; g.gain.setValueAtTime(0.0001,c.currentTime); g.gain.exponentialRampToValueAtTime(0.16,c.currentTime+0.02); g.gain.exponentialRampToValueAtTime(0.001,c.currentTime+0.6); const mix=c.createGain(); o1.connect(mix); o2.connect(mix); mix.connect(g); g.connect(c.destination); o1.start(); o2.start(); o1.stop(c.currentTime+0.48); o2.stop(c.currentTime+0.48);}catch(e){} }

  // Game vars
  const colors = ['#ff6b6b','#ffd166','#6ef0b8','#6fb3ff','#c77dff'];
  const center = {x: canvas.width/2, y: canvas.height/2};
  const pathRadius = Math.min(canvas.width, canvas.height) * 0.38;
  const ballR = 18;
  const baseSpeed = 0.018;
  let chain = [], score=0, level=1, shooter={angle:-Math.PI/2, nextBall:colors[0]}, fired=null, particles=[], trails=[];
  let lastTime=0, spawnCounter=0, gameOver=false;

  const levelConfig = (lv) => ({ speed: baseSpeed + (lv-1)*0.004, spawnInterval: Math.max(420,900-(lv-1)*80), targetScore:50*lv });

  function randomColor(){ return colors[Math.floor(Math.random()*colors.length)]; }
  function posFromT(t){ const maxTurns=2.2; const theta=Math.PI*2*maxTurns*(t)+Math.PI/2; const r=pathRadius*t; return {x:center.x + r*Math.cos(theta), y:center.y + r*Math.sin(theta), theta}; }

  function updateHUD(){ scoreEl.textContent = score; const cfg = levelConfig(level); levelInfo.textContent = `Level ${level}`; }

  function reset(){ chain=[]; score=0; level=1; shooter.angle=-Math.PI/2; shooter.nextBall=randomColor(); particles=[]; trails=[]; spawnCounter=0; gameOver=false; for(let i=0;i<10;i++) chain.push({t:0.9 - i*0.05, color: colors[i%colors.length]}); updateHUD(); showBanner("Level " + level, 1200); }

  function showBanner(text, ms=1000){ banner.textContent = text; banner.classList.add('show'); setTimeout(()=>banner.classList.remove('show'), ms); playLevelUp(); }

  function spawnPop(x,y,color,count=12){ for(let i=0;i<count;i++){ const a=Math.random()*Math.PI*2; const s=1+Math.random()*3; particles.push({ x,y, vx:Math.cos(a)*s*(1+Math.random()), vy:Math.sin(a)*s*(1+Math.random()), life:600+Math.random()*300, age:0, color }); } try{ playPop(); }catch(e){} }
  function spawnTrail(x,y){ trails.push({x,y,r:3+Math.random()*3, alpha:0.9, life:220}); }

  function update(dt){
    if (gameOver) return;
    const cfg = levelConfig(level);
    for (let b of chain){ b.t -= cfg.speed * dt; if (b.t <= 0.06) gameOver = true; }
    spawnCounter += dt * 16.666;
    if (spawnCounter > cfg.spawnInterval / 16.666){ spawnCounter = 0; const lastT = chain.length ? chain[chain.length-1].t : 0.98; chain.push({t: lastT + 0.06 + Math.random()*0.03, color: randomColor()}); }
    if (fired){ fired.x += fired.vx * (dt); fired.y += fired.vy * (dt); spawnTrail(fired.x, fired.y); }
    for (let p of particles){ p.age += dt*16.666; p.x += p.vx * dt * 0.9; p.y += p.vy * dt * 0.9; p.vx *= 0.995; p.vy *= 0.995; }
    for (let i=particles.length-1;i>=0;i--){ if (particles[i].age > particles[i].life) particles.splice(i,1); }
    for (let i=trails.length-1;i>=0;i--){ trails[i].life -= dt*16.666; trails[i].alpha -= 0.018*dt; if (trails[i].life <= 0 || trails[i].alpha <= 0) trails.splice(i,1); }
    if (fired) handleCollision();
    if (score >= levelConfig(level).targetScore){ level++; updateHUD(); showBanner("Level " + level, 1200); for (let i=0;i<3;i++){ const last = chain.length ? chain[chain.length-1].t : 0.9; chain.push({t: last + 0.045*(i+1), color: randomColor()}); } }
  }

  function handleCollision(){
    if (!fired) return;
    for (let i=0;i<chain.length;i++){
      const b = chain[i]; const p = posFromT(Math.max(0.03,b.t)); const dx = fired.x - p.x; const dy = fired.y - p.y; const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist <= (ballR*1.9)) {
        const insertIndex = i; const insertT = Math.min(b.t + 0.02, 0.98);
        chain.splice(insertIndex, 0, {t: insertT, color: fired.color}); fired = null; runRemoval(insertIndex); return;
      }
    }
    if (fired.x < -60 || fired.x > canvas.width+60 || fired.y < -60 || fired.y > canvas.height+60) fired = null;
  }

  function runRemoval(insertIndex){
    const color = chain[insertIndex].color; let left = insertIndex, right = insertIndex;
    while (left-1 >= 0 && chain[left-1].color === color) left--; while (right+1 < chain.length && chain[right+1].color === color) right++;
    const count = right - left + 1;
    if (count >= 3){
      let avgX=0, avgY=0;
      for (let k=left;k<=right;k++){ const pos = posFromT(chain[k].t); avgX += pos.x; avgY += pos.y; }
      avgX /= count; avgY /= count;
      spawnPop(avgX, avgY, color, Math.min(28, 8 + count*4));
      score += count * 12; updateHUD();
      chain.splice(left, count);
      if (count >= 5){ const mid = Math.max(0, Math.min(chain.length-1, left)); chain.splice(mid, 0, {t: 0.5 + Math.random()*0.3, color: randomColor()}); }
      try{ playPop(); }catch(e){} 
    }
  }

  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const g = ctx.createRadialGradient(center.x, center.y - 80, 20, center.x, center.y, Math.max(canvas.width, canvas.height));
    g.addColorStop(0, 'rgba(255,255,255,0.02)'); g.addColorStop(1, 'rgba(0,0,0,0)'); ctx.fillStyle = g; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.lineWidth = 1.4; ctx.strokeStyle = 'rgba(255,255,255,0.02)'; ctx.beginPath();
    for (let tt=0; tt<=1.001; tt+=0.003){ const p = posFromT(tt); if (tt===0) ctx.moveTo(p.x,p.y); else ctx.lineTo(p.x,p.y); } ctx.stroke();
    for (let i = chain.length-1; i>=0; i--){ const b = chain[i]; const p = posFromT(Math.max(0.03, b.t)); const scale = 1 + (0.9 - Math.max(0.03, b.t))*0.15; drawBall(p.x, p.y, ballR*scale, b.color); }
    if (fired) drawBall(fired.x, fired.y, ballR, fired.color);
    for (let p of particles){ const lifeRatio = 1 - p.age / p.life; ctx.beginPath(); ctx.globalAlpha = Math.max(0, lifeRatio); ctx.fillStyle = p.color; ctx.arc(p.x, p.y, 3 + (1-lifeRatio)*5, 0, Math.PI*2); ctx.fill(); ctx.globalAlpha = 1; }
    for (let tr of trails){ ctx.beginPath(); ctx.globalAlpha = Math.max(0, tr.alpha); ctx.fillStyle = 'rgba(255,255,255,0.06)'; ctx.arc(tr.x, tr.y, tr.r, 0, Math.PI*2); ctx.fill(); ctx.globalAlpha = 1; }
    ctx.beginPath(); ctx.fillStyle = '#03060a'; ctx.arc(center.x, center.y, 30, 0, Math.PI*2); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.stroke();
  }

  function drawBall(x,y,r,color){ ctx.beginPath(); ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 14; ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); ctx.shadowBlur = 0; const lg = ctx.createRadialGradient(x - r*0.4, y - r*0.6, r*0.1, x, y, r); lg.addColorStop(0, 'rgba(255,255,255,0.45)'); lg.addColorStop(0.25, 'rgba(255,255,255,0.12)'); lg.addColorStop(1, 'rgba(255,255,255,0)'); ctx.beginPath(); ctx.fillStyle = lg; ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); ctx.lineWidth = 1.6; ctx.strokeStyle = 'rgba(0,0,0,0.28)'; ctx.stroke(); }

  let dragging=false, lastDown={x:0,y:0};
  function rotateToPointer(clientX, clientY){ const rect = canvas.getBoundingClientRect(); const cx = rect.left + center.x * (rect.width/canvas.width); const cy = rect.top + center.y * (rect.height/canvas.height); const dx = clientX - cx; const dy = clientY - cy; shooter.angle = Math.atan2(dy, dx) + Math.PI/2; }
  canvas.addEventListener('pointerdown', (e)=>{ dragging=true; lastDown.x=e.clientX; lastDown.y=e.clientY; rotateToPointer(e.clientX, e.clientY); try{ ensureAudio(); }catch(e){} });
  window.addEventListener('pointermove', (e)=>{ if(!dragging) return; rotateToPointer(e.clientX, e.clientY); });
  window.addEventListener('pointerup', (e)=>{ if(!dragging) return; const dx=e.clientX - lastDown.x; const dy=e.clientY - lastDown.y; if (Math.sqrt(dx*dx + dy*dy) < 12) doShoot(); dragging=false; });
  canvas.addEventListener('click', (e)=>{ rotateToPointer(e.clientX, e.clientY); doShoot(); try{ ensureAudio(); }catch(e){} });

  function doShoot(){ if (gameOver) return; if (fired) return; const speed=10; const startX = center.x + Math.cos(shooter.angle) * -70; const startY = center.y + Math.sin(shooter.angle) * -70; const vx=Math.cos(shooter.angle) * -speed; const vy=Math.sin(shooter.angle) * -speed; fired={x:startX,y:startY,vx,vy,color:shooter.nextBall}; shooter.nextBall=randomColor(); try{ playShoot(); }catch(e){} }

  resetBtn.addEventListener('click', ()=>{ reset(); });

  function loop(ts){ if (!lastTime) lastTime = ts; const dt = (ts - lastTime) / 16.666; lastTime = ts; update(dt); draw(); ctx.save(); ctx.resetTransform(); ctx.font='14px Inter,Arial'; ctx.fillStyle='rgba(255,255,255,0.95)'; ctx.textAlign='left'; ctx.fillText('Next:', 12, 24); drawBall(64,20,ballR*0.7,shooter.nextBall); ctx.restore(); if (!gameOver) requestAnimationFrame(loop); else { ctx.fillStyle='rgba(0,0,0,0.6)'; ctx.fillRect(30, canvas.height/2 - 60, canvas.width - 60, 120); ctx.fillStyle='#fff'; ctx.textAlign='center'; ctx.font='28px Inter,Arial'; ctx.fillText('Game Over', canvas.width/2, canvas.height/2 - 6); ctx.font='16px Inter,Arial'; ctx.fillText('Tekan Reset untuk main lagi', canvas.width/2, canvas.height/2 + 26); } }
  reset(); requestAnimationFrame(loop);

  // expose for debugging
  window.__zuma = {chain, spawnPop, reset};
})();
</script>
</body>
</html>
"""

components.html(html_code, height=920, scrolling=False)
