const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
const buttons = [...document.querySelectorAll('[data-engine]')];
const pointers = new Map(), keyboard = new Set();
const bindings = {KeyA:'left',KeyW:'main',KeyD:'right'};
let state = Lunar.start(), mode = '', last = 0, carry = 0;
function start(nextMode) {
  if (!mode || state.result) requestAnimationFrame(frame);
  state = Lunar.start(); mode = nextMode; carry = 0; last = 0;
  pointers.clear(); keyboard.clear(); document.querySelector('#debrief').textContent = '';
  document.querySelector('#result').textContent = mode === 'auto' ? 'Autopilot flying' : 'Manual flight';
}
document.querySelector('#manual').onclick = () => start('manual');
document.querySelector('#auto').onclick = () => start('auto');
for (const button of buttons) {
  button.onpointerdown = e => {
    if (!mode || state.result) start('manual');
    mode = 'manual'; document.querySelector('#result').textContent = 'Manual flight';
    button.setPointerCapture(e.pointerId); pointers.set(e.pointerId, button.dataset.engine);
  };
  button.onlostpointercapture = e => pointers.delete(e.pointerId);
}
window.onkeydown = e => {
  if (!bindings[e.code] || e.repeat) return;
  if (!mode || state.result) start('manual');
  mode = 'manual'; document.querySelector('#result').textContent = 'Manual flight';
  keyboard.add(bindings[e.code]);
};
window.onkeyup = e => keyboard.delete(bindings[e.code]);
window.onblur = () => { keyboard.clear(); pointers.clear(); };
function draw(keys) {
  const scale = Math.min(16, 290 / Math.max(16,state.h), 350 / Math.max(12,Math.abs(state.x)));
  const x = 400 + state.x * scale, y = 315 - state.h * scale;
  ctx.clearRect(0,0,800,350);
  ctx.fillStyle = '#a4bed0';
  for (let i = 1; i <= 65; i++) ctx.fillRect((i * 173) % 797, (i * i * 17) % 284, 2, 2);
  ctx.strokeStyle = '#7899ac';
  ctx.setLineDash([3,6]);
  ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x,315-scale*Lunar.terrain(state.x)); ctx.stroke();
  ctx.setLineDash([]);
  const surface = px => 315-scale*Lunar.terrain((px-400)/scale);
  ctx.beginPath(); ctx.moveTo(0,350);
  for(let px=0; px<=800; px+=2) ctx.lineTo(px,surface(px));
  ctx.lineTo(800,350); ctx.closePath();
  const soil = ctx.createLinearGradient(0,270,0,350);
  soil.addColorStop(0,'#8994a1'); soil.addColorStop(1,'#37414f');
  ctx.fillStyle=soil; ctx.fill();
  // Shallow crater shadows sit below the silhouette.
  for(let i=0; i<60; i++) {
    const px=400+((i*13.71)%56-28)*scale;
    ctx.fillStyle=i%2 ? '#505b69' : '#707b87';
    ctx.beginPath(); ctx.ellipse(px,surface(px)+8+i%19,(.14+i%4*.12)*scale,.12*scale,0,0,2*Math.PI); ctx.fill();
  }
  const halfPad=Lunar.pad*scale;
  ctx.fillStyle='#66e3bc'; ctx.fillRect(400-halfPad,315,2*halfPad,4);
  for(const edge of [400-halfPad,400+halfPad]) {
    ctx.fillRect(edge-1,307,2,8); ctx.fillRect(edge-3,305,6,3);
  }
  ctx.font='bold 12px monospace'; ctx.textAlign='center'; ctx.fillText('TARGET',400,337);
  ctx.save(); ctx.translate(x,y); ctx.rotate(state.angle); ctx.scale(scale,-scale);
  function polygon(points,color) {
    ctx.beginPath(); ctx.moveTo(...points[0]);
    for(const p of points.slice(1)) ctx.lineTo(...p);
    ctx.closePath(); ctx.fillStyle=color; ctx.fill();
  }
  if(state.fuel && !state.result) {
    if(keys.main) polygon([[-.16,-.32],[0,-1.3],[.16,-.32]],'#ffb65c');
    if(keys.left) polygon([[-.4,.43],[-1.25,.35],[-.4,.27]],'#9adfff');
    if(keys.right) polygon([[.4,.43],[1.25,.35],[.4,.27]],'#9adfff');
  }
  ctx.strokeStyle='#d9caa4'; ctx.lineWidth=.09;
  for(const [fx,fy] of Lunar.feet) {
    ctx.beginPath(); ctx.moveTo(fx*.45,-.15); ctx.lineTo(fx,fy);
    ctx.lineTo(fx*.78,fy); ctx.stroke();
  }
  polygon(Lunar.hull,'#d7b87a');
  polygon([[-.28,.37],[.26,.37],[.36,.11],[-.35,.11]],'#26455b');
  ctx.restore();
  for(const b of buttons) b.classList.toggle('active',!!keys[b.dataset.engine] && !state.result && state.fuel>0);
  document.querySelector('#telemetry').textContent =
    'Offset '+state.x.toFixed(1)+' m · Center above terrain '+(state.h-Lunar.terrain(state.x)).toFixed(1)+' m · Horizontal '+state.vx.toFixed(2)+' m/s · Vertical '+state.vy.toFixed(2)+' m/s · Tilt '+(state.angle*180/Math.PI).toFixed(0)+'° · Spin '+(state.spin*180/Math.PI).toFixed(0)+'°/s · Fuel '+state.fuel.toFixed(1);
}
function frame(time) {
  carry += last ? Math.min(.1,(time-last)/1000) : 0; last=time;
  let keys = {left:false,right:false,main:false};
  for(const k of [...keyboard,...pointers.values()]) keys[k]=true;
  while(carry >= 1/60) {
    if(mode==='auto') keys=Lunar.pilot(state);
    if(mode) Lunar.step(state,keys);
    carry -= 1/60;
  }
  if(mode==='auto') keys=Lunar.pilot(state);
  if(state.result && document.querySelector('#result').textContent !== state.result) {
    document.querySelector('#result').textContent = state.result;
    document.querySelector('#debrief').textContent = state.misses.length
      ? 'At touchdown: ' + state.misses.join(', ') + '.'
      : 'Both feet over the pad. Speed, tilt and rotation within limits.';
  }
  draw(keys);
  if (mode && !state.result) requestAnimationFrame(frame);
}
draw({});
