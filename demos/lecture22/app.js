'use strict';
const {World,Planner,RNG,ARROWS,actions,target,residual}=DP;
const $=id=>document.getElementById(id);
const lessons=[
 {name:'Policy evaluation',title:'What is a policy worth?',slides:'04—09',sub:'The slide’s 4 × 4 grid. Four equally likely actions. One costly step at a time.',headline:'Average over actions.',body:'A random walker does not choose the best direction. Evaluate its fixed policy by averaging all four action values. Watch the expected remaining journey length emerge from repeated local updates.',eq:'Vₖ₊₁(s) = ¼ Σₐ [−1 + γVₖ(s′)]',question:'At γ = 1, why does a value of −20 mean 20 expected steps, rather than the shortest distance?',note:'The two corner exits have value zero. Every transition from a nonterminal cell costs −1, including exit entry and collisions with the boundary. Updates use the entire previous value array. The random policy is proper in this finite grid, so γ = 1 converges here; the discounted contraction argument requires γ < 1.'},
 {name:'Policy iteration',title:'Evaluate. Improve. Repeat.',slides:'10—13',sub:'Separate learning how good a policy is from changing what it does.',headline:'Two distinct operations.',body:'First hold the policy fixed and evaluate it to a small Bellman residual. Then choose every action tied for the largest Q-value, equally likely. Arrows change only during the improvement step. Repeat until the policy stops changing.',eq:'Evaluate: V ← Tᵖⁱ V\nImprove: π ← greedy(V)',question:'Can the policy already be optimal while its values are still inaccurate?',note:'Starts from the uniform random policy. “One step” is one evaluation sweep or one complete improvement. Evaluation stops at residual < 10⁻⁷. Equal-probability greedy ties avoid presenting an arbitrary unique best move. This is numerical policy iteration, with warm-started evaluation.'},
 {name:'Value iteration',title:'Let good news travel.',slides:'14—19',sub:'A wave of useful information moves backward from the exits.',headline:'Maximize at every backup.',body:'Instead of finishing policy evaluation, take a greedy Bellman backup immediately. The first sweep knows only about the immediate cost. More sweeps reveal longer futures. Switch to in-place updates to see the effect of reusing new values within a sweep.',eq:'Vₖ₊₁(s) = maxₐ [−1 + γVₖ(s′)]',question:'Why can in-place updates move information farther than one cell per sweep?',note:'Synchronous updates reproduce slide 17 at γ = 1. In-place updates use row-major state order. Residual priority ranks all current state residuals and backs up the largest 1/8; it illustrates prioritized asynchronous updates, not an optimized predecessor-queue implementation. Sweep/batch counts are not equal work; compare state backups.'},
 {name:'Rescue mission',title:'Plan for the unexpected.',slides:'18—19 · APPLICATION',sub:'Follow the value gradient through the ridge openings toward the rescue destination.',headline:'The shortest route can be costly.',body:'Both planners start at zero on the same map. Compare their optimality residuals, max |T*V − V|. PI evaluates a policy before improving it. VI maximizes every sweep. A known stochastic model lets a planner account for outcomes that a shortest-path route ignores. Plan, launch a fleet, then add an obstacle or hazard. Every edit resets the values so you can watch a new solution emerge.',eq:'Q(s,a) = Σₛ′ P(s′|s,a)\n           [r + γV(s′)]',question:'When the robot may slip sideways, does the best route give dangerous terrain a wider berth?',note:'Illustrative rescue environment, not a physical robot simulator. Blue disk: launch; gold ring: destination; orange: hazard (−16 on entry); walls block movement. Other moves cost −1, goal entry gives +100 and terminates. Slip splits equally between perpendicular directions. γ defaults to .98. Rollout animates 12 independent seeded trials of the frozen policy, not a success-rate estimate. The large map has 4,000 states; the sparse transition model has at most three outcomes per action.'},
 {name:'Contraction',title:'Different starts, one rescue value function',slides:'20—22',sub:'Switch initial values or resample. Both planners start from the same random grid.',headline:'A unique optimal value for every rescue state.',body:'Each column solves the same discounted rescue mission. Each new sample changes the initial value at every active cell. Walls and the destination stay at zero.',eq:'‖T*U − T*V‖∞ ≤ γ ‖U − V‖∞',question:'Can two policies tie for the best action even when the optimal value is unique?',note:'Every synchronous VI sweep contracts the maximum distance between the two value grids by at least the factor γ in the bound. PI alternates policy evaluation and improvement. Its displayed gap need not satisfy the VI bound per tick. The first completed VI solve supplies a numerical reference for this map and its settings. Later samples report their largest cell-wise difference from that reference.'}
];
let section=0,world,planner,other,selected=5,playing=false,finishing=false,roll=null,lastTime=0,changedAt=0,layout=[],gapHistory=[],initialGap=40,statusText='';
const canvas=$('world'),ctx=canvas.getContext('2d');
$('tabs').innerHTML=lessons.map((x,i)=>`<button data-section="${i}"><span>0${i+1}</span>${x.name}</button>`).join('');
$('tabs').addEventListener('click',e=>{let b=e.target.closest('button');if(b)selectSection(+b.dataset.section);});
function stop(){playing=false;finishing=false;$('play').textContent='▶ Play';}
let inspectingPI=false;
let rescueVI=[], rescuePI=[];
let seed=42, freshSeed=101, reference=null, referenceKey='', priorError=NaN;
function distance(a,b){
 let maximum=0;
 for(let i=0;i<a.length;i++)maximum=Math.max(maximum,Math.abs(a[i]-b[i]));
 return maximum;
}
function initializeComparison(){
 const choice=$('seed-choice').value;
 seed=choice==='fresh'?freshSeed:Number(choice)||42;
 const key=[world.map,world.w,world.gamma,world.slip].join('/');
 if(key!==referenceKey){reference=null;referenceKey=key;}
 const random=new RNG(seed);
 for(let cell=0;cell<world.n;cell++)if(world.active(cell)){
   planner.v[cell]=other.v[cell]=-40+140*random.next();
 }
 for(const run of [other,planner])run.old.set(run.v);
 for(let cell=0;cell<world.n;cell++)if(world.active(cell))planner.policy[cell]=DP.greedy(world.qs(cell,planner.v));
 initialGap=residual(world,planner.v,null);
 gapHistory.push(initialGap);
 priorError=reference?distance(planner.v,reference):NaN;
}
function completed(){return planner.done&&(!other||other.done);}
function reset(){rescueVI=[];rescuePI=[];inspectingPI=false;stop();roll=null;const rescue=section>=3;let width=rescue?+$('size').value:4;world=new World(width,rescue?(width===80?50:16):4,{rescue,map:$('map').value,slip:rescue?+$('slip').value:0,gamma:+$('gamma').value});planner=new Planner(world,section===0?'eval':section===1?'pi':'vi');selected=world.start;other=rescue?new Planner(world,'pi'):null;gapHistory=[];if(section===4)initializeComparison();statusText='Animation started. Pause or One step to inspect an update.';changedAt=0;playing=true;lastTime=performance.now();$('play').textContent='Ⅱ Pause';render();}
function selectSection(i){const previous=section;section=i;document.querySelectorAll('#tabs button').forEach((b,j)=>b.classList.toggle('selected',i===j));const l=lessons[i];$('chapter').textContent=`EXPERIMENT 0${i+1} / SLIDES ${l.slides}`;$('title').textContent=l.title;$('subtitle').textContent=l.sub;$('big-number').textContent=`0${i+1}`;$('idea').innerHTML=`<p class="eyebrow">INSIDE THE ALGORITHM</p><h2>${l.headline}</h2><p>${l.body}</p><div class="equation">${l.eq.replaceAll('\n','<br>')}</div><div class="question"><p class="eyebrow">ASK THE ROOM</p><p>${l.question}</p></div><p class="note">${l.note}</p>`;
$('seed-wrap').hidden=$('sample-values').hidden=i!==4;$('high-key').hidden=$('low-key').hidden=i>=3;$('gamma').options[0].disabled=i>=3;$('gamma').value=i>=3?(previous>=3?$('gamma').value:'.98'):'1';for(const name of ['slip','size','map'])$(name+'-wrap').hidden=i<3;$('brush-wrap').hidden=i!==3;$('method-wrap').hidden=![2,3].includes(i);$('method').value='sync';$('brush').value='inspect';$('rollout').hidden=i===4;$('chart-label').textContent=i===4?'VI OPTIMALITY RESIDUAL · LOG SCALE':i===3?'VI AND PI OPTIMALITY RESIDUALS · LOG SCALE':'BELLMAN RESIDUAL · LOG SCALE';$('map-legend').textContent=i>=3?'■ Wall · ▧ Hazard · ◎ Destination · Values −40 (purple) → 0 (blue) → 50 (green) → 100 (yellow)':i===2?'Arrows: policy · Fixed color scale −4…0':i===4?'Shared fixed color scale −24…20':'Arrows: policy · Fixed color scale −24…0';reset();}
function step(refresh=true){roll=null;const phase=planner.phase;if(section===3){
 const mode=$('method').value;
 planner.step(mode);
 other.step(mode);
 rescueVI.push(residual(world,planner.v,null));
 rescuePI.push(residual(world,other.v,null));
 statusText='VI '+(planner.done?'converged':'maximizing')+' · PI '+other.phase+' · '+other.round+' improvement passes. One tick advances each active planner by one sweep, batch or improvement pass.';
}else if(section===4){
 other.step(); planner.step();
 gapHistory.push(planner.delta);
 if(planner.done&&!reference)reference=planner.v.slice();
 priorError=reference?distance(planner.v,reference):NaN;
 statusText=`Seed ${seed} · VI sweep ${planner.sweeps} · PI improvements ${other.round}. ${reference?'Reference saved for these settings.':'First converged VI run will supply the reference.'}`;

}
else{planner.step($('method').value);statusText=section===1?(phase==='improve'?`Improvement ${planner.round}: ${planner.changed} states changed policy. ${planner.done?'Policy is stable.':'Now evaluate this new policy.'}`:planner.phase==='improve'?'Evaluation converged. Next step performs greedy policy improvement.':`Evaluating the fixed policy. Residual ${fmt(planner.delta)}.`):`Completed ${$('method').value==='priority'?'priority batch':'sweep'} ${planner.sweeps}. ${planner.last.length.toLocaleString()} state backups this step.`;}
if(section===1&&planner.phase==='improve'){stop();statusText='Evaluation converged. Predict the new arrows, then press One step to improve the policy.';}
changedAt=performance.now();if(completed()){stop();statusText+=section===1?'':' Converged: Bellman residual < 10⁻⁷.';}if(refresh)render();}
function fmt(v){if(!Number.isFinite(v))return '—';if(Math.abs(v)<.00001&&v!==0)return v.toExponential(2);return v.toFixed(3);}
function metric(value,label){return `<div class="metric"><b>${value}</b><span>${label}</span></div>`;}
function render(){
const r=residual(world,planner.v,section<2?planner.policy:null);
$('metrics').innerHTML=section===4?metric(fmt(distance(other.v,planner.v)),'PI vs VI · max difference')+metric(fmt(gapHistory.at(-1)),'VI optimality residual')+metric(fmt(initialGap*world.gamma**planner.sweeps),'VI residual upper bound')+metric(fmt(priorError),'VI vs first solution · max difference'):metric(planner.sweeps,$('method').value==='priority'?'Priority batches':'Evaluation / VI sweeps')+metric(planner.backups.toLocaleString(),'State backups')+metric(fmt(r),'Bellman residual')+metric(section===1?planner.round:world.activeCount,section===1?'Policy improvements':'Nonterminal states');
if(section===3){
 const work=$('method').value==='priority'?'batches':'sweeps';
 $('metrics').innerHTML=metric(other.sweeps,'PI evaluation '+work+' · '+other.backups+' backups')+metric(fmt(residual(world,other.v,null)),'PI optimality residual')+metric(planner.sweeps,'VI '+work+' · '+planner.backups+' backups')+metric(fmt(r),'VI optimality residual');
}
$('rollout').textContent=section===3?'Animate VI fleet':'Animate policy';
$('finish').textContent=section===1?'Finish current phase':'Run to convergence';
$('status').textContent=statusText;$('step').disabled=completed();$('finish').disabled=$('step').disabled;$('play').disabled=$('step').disabled&&!roll;inspect();drawChart();draw(performance.now());}
function inspect(){const source=inspectingPI?other:planner;const evaluating=section<2||inspectingPI&&source.phase!=='improve';let s=selected,q=world.qs(s,source.v),mask=source.policy[s];
$('numbers').innerHTML=`<p class="eyebrow">${section>=3?(inspectingPI?'PI · ':'VI · '):''}NEXT BACKUP / CELL (${Math.floor(s/world.w)}, ${s%world.w})</p><h2>V(s) = ${fmt(source.v[s])}</h2>`+(world.active(s)?`<table class="calculation"><thead><tr><th>Action</th><th>Q(s,a)</th><th>${evaluating?'π(a|s)':'Greedy'}</th></tr></thead><tbody>${q.map((v,a)=>`<tr class="${mask&(1<<a)?'chosen':''}"><td>${ARROWS[a]}</td><td>${fmt(v)}</td><td>${evaluating?((mask&(1<<a))?1/actions(mask).length:0).toFixed(2):(mask&(1<<a))?'✓':'—'}</td></tr>`).join('')}</tbody></table><div class="equation">${evaluating?'Average under π':'Maximum Q'} = ${fmt(target(world,s,source.v,evaluating?source.policy:null))}</div><p class="note">These are predictions using the currently displayed values, not a record of the last update. Each Q averages over all transition outcomes.</p><p class="note">Up-action branches:<br>${world.model[s][0].map(t=>`${(t.p*100).toFixed(0)}% → (${Math.floor(t.s/world.w)},${t.s%world.w}), r=${t.r}, V′=${fmt(source.v[t.s])}`).join('<br>')}</p>`:'<p>This is a wall or terminal state. Its value stays zero; it receives no backup.</p>');

}
function fit(c){let rect=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;if(c.width!==Math.round(rect.width*dpr)||c.height!==Math.round(rect.height*dpr)){c.width=Math.round(rect.width*dpr);c.height=Math.round(rect.height*dpr);}let cctx=c.getContext('2d');cctx.setTransform(dpr,0,0,dpr,0,0);return [rect.width,rect.height];}
function draw(now){if(!world)return;const [W,H]=fit(canvas);ctx.clearRect(0,0,W,H);layout=[];
if(section>=3){
 drawGrid(other.v,20,34,W/2-30,H-50,'Policy iteration',now);
 drawGrid(planner.v,W/2+10,34,W/2-30,H-50,'Value iteration',now);
}else drawGrid(planner.v,12,12,W-24,H-24,'',now);
}
// A fixed numeric scale keeps colors comparable across maps, starts and iterations.
const rescueColors = Array.from({length:141}, (_,index)=>{
 const stops=[[0,65,29,99],[40,39,87,147],[90,53,173,128],[140,252,231,91]];
 const right=stops.findIndex(stop=>stop[0]>=index);
 const high=stops[right], low=stops[Math.max(0,right-1)];
 const fraction=(index-low[0])/Math.max(1,high[0]-low[0]);
 return 'rgb('+low.slice(1).map((v,i)=>Math.round(v+fraction*(high[i+1]-v))).join(',')+')';
});
function drawGrid(values,x,y,width,height,label,now,run=null){let size=Math.min(width/world.w,height/world.h),left=x+(width-size*world.w)/2,top=y+(height-size*world.h)/2;layout.push({left,top,size});if(label){ctx.fillStyle='#bdd0e0';ctx.font='12px sans-serif';ctx.textAlign='center';ctx.fillText(label,x+width/2,y-12);}
const lo=section>=3?-40:section===2?-4:-24,hi=section>=3?100:section===4?20:0;const owner=run||(other&&values===other.v?other:planner);let visiblePolicy=owner.policy;const old=owner.old;let blend=changedAt?Math.min(1,Math.max(0,(now-changedAt)/Math.min(180,+$('tempo').value*.8))):1;blend=blend*blend*(3-2*blend);
for(let s=0;s<world.n;s++){const c=s%world.w,r=Math.floor(s/world.w),px=left+c*size,py=top+r*size,t=Math.max(0,Math.min(1,((old[s]+blend*(values[s]-old[s]))-lo)/(hi-lo)));ctx.fillStyle=world.cells[s]===1?'#09111c':world.terminal(s)?'#384d4c':world.rescue?rescueColors[Math.round(140*t)]:`rgb(${Math.round(25+t*27)},${Math.round(52+t*103)},${Math.round(86+t*43)})`;ctx.fillRect(px+1,py+1,size-2,size-2);
if(world.cells[s]===2){ctx.fillStyle='rgba(255,139,56,.65)';ctx.fillRect(px+1,py+1,size-2,size-2);if(size>13){ctx.strokeStyle='#ffb76c';ctx.beginPath();ctx.moveTo(px+2,py+size-2);ctx.lineTo(px+size-2,py+2);ctx.stroke();}}

ctx.textAlign='center';ctx.textBaseline='middle';if(world.terminal(s)){ctx.fillStyle='#ffda91';ctx.font=`${Math.max(12,size*.4)}px sans-serif`;ctx.fillText('◎',px+size/2,py+size/2);}else if(world.cells[s]===1){if(size>25){ctx.fillStyle='#5e7289';ctx.font=`${size*.4}px sans-serif`;ctx.fillText('×',px+size/2,py+size/2);}}
else if(size>35){ctx.fillStyle='#f0f8ff';ctx.font=`500 ${Math.min(23,size*.24)}px sans-serif`;ctx.fillText(values[s].toFixed(1),px+size/2,py+size*.42);ctx.font=`${Math.min(18,size*.20)}px sans-serif`;ctx.fillStyle='#c3f9ed';ctx.fillText(actions(visiblePolicy[s]).map(a=>ARROWS[a]).join(''),px+size/2,py+size*.72);}
else if(size>9){let aa=actions(visiblePolicy[s]);if(aa.length===1){const d=DP.DIR[aa[0]],cx=px+size/2,cy=py+size/2;ctx.strokeStyle='#c1e9df';ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(cx-d[1]*size*.2,cy-d[0]*size*.2);ctx.lineTo(cx+d[1]*size*.2,cy+d[0]*size*.2);ctx.lineTo(cx+(d[1]-d[0])*size*.09,cy+(d[0]+d[1])*size*.09);ctx.stroke();}}
if(s===selected){ctx.strokeStyle='#ffd18b';ctx.lineWidth=2;ctx.strokeRect(px+1,py+1,size-2,size-2);}
}
if(world.rescue){let s=world.start;ctx.fillStyle='#78b7ff';ctx.beginPath();ctx.arc(left+(s%world.w+.5)*size,top+(Math.floor(s/world.w)+.5)*size,Math.max(3,size*.2),0,Math.PI*2);ctx.fill();}
if(roll&&values===planner.v){let f=Math.min(1,(now-roll.time)/Math.min(220,+$('tempo').value));roll.agents.forEach((a,i)=>{ctx.strokeStyle=`hsla(${160+i*9},80%,72%,.5)`;ctx.lineWidth=Math.max(1,size*.07);ctx.beginPath();a.path.forEach((s,j)=>{let xx=left+(s%world.w+.5)*size,yy=top+(Math.floor(s/world.w)+.5)*size;j?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);});ctx.stroke();const xx=(a.prev%world.w)*(1-f)+(a.s%world.w)*f,yy=Math.floor(a.prev/world.w)*(1-f)+Math.floor(a.s/world.w)*f;ctx.fillStyle=a.done?'#ffd18b':'#f5ffff';ctx.beginPath();ctx.arc(left+(xx+.5)*size,top+(yy+.5)*size,Math.max(3,size*.16),0,Math.PI*2);ctx.fill();});}
}
function drawChart(){const c=$('chart'),[W,H]=fit(c),g=c.getContext('2d');g.clearRect(0,0,W,H);let data=section===4?gapHistory:section===3?rescueVI:planner.history;if(!data.length){$('chart-note').textContent='Run a step to start the measured convergence trace.';return;}let plotted=section===3?data.concat(rescuePI):data;let logs=plotted.map(x=>Math.log10(Math.max(1e-10,x))),min=Math.min(-7,...logs),max=Math.max(1,...logs);const Y=v=>8+(max-Math.log10(Math.max(1e-10,v)))/(max-min)*(H-24),X=i=>32+i/Math.max(1,data.length-1)*(W-40);g.font='9px sans-serif';g.fillStyle='#91a7bc';g.textAlign='left';for(let k=Math.ceil(min);k<=max;k+=2){let yy=Y(10**k);g.fillText('1e'+k,0,yy+3);g.strokeStyle='#27384a';g.beginPath();g.moveTo(30,yy);g.lineTo(W,yy);g.stroke();}function line(arr,color,dash=[]){g.strokeStyle=color;g.lineWidth=2;g.setLineDash(dash);g.beginPath();arr.forEach((v,i)=>i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v)));g.stroke();g.setLineDash([]);}line(data,'#69ead1');if(section===3)line(rescuePI,'#ffd18b',[4,4]);if(section===4)line(data.map((_,i)=>initialGap*world.gamma**Math.min(i,planner.sweeps)),'#ffd18b',[4,4]);$('chart-note').textContent=section===3?'Teal: VI · Gold dashed: PI. Both show max |T*V − V| per paired tick. Finished planners hold their final value.':section===4?'Teal: measured residual · Gold dashed: γᵏ × initial residual. Horizontal axis: paired ticks; VI holds after convergence.':'Measured max |T V − V| after each sweep/batch. Smaller is better. Horizontal axis: completed sweeps/batches.';}
function startRollout(){stop();const rng=new RNG(42);roll={agents:Array.from({length:section===3?12:1},()=>({s:world.start,prev:world.start,path:[world.start],done:false,reward:0,risks:0})),rng,time:performance.now(),ticks:0};playing=true;$('play').disabled=false;$('play').textContent='Ⅱ Pause';statusText='Following the frozen current policy. Planning is paused.';lastTime=0;render();}
function rollStep(now){roll.time=now;roll.ticks++;for(const a of roll.agents){a.prev=a.s;if(a.done)continue;let as=actions(planner.policy[a.s]),t=world.sample(a.s,as[Math.floor(roll.rng.next()*as.length)],roll.rng);a.s=t.s;a.reward+=t.r;a.risks+=world.cells[a.s]===2?1:0;a.path.push(a.s);a.done=world.terminal(a.s);}let n=roll.agents.filter(a=>a.done).length;statusText=`Frozen policy · ${n}/${roll.agents.length} reached the goal · ${roll.ticks} moves · ${roll.agents.reduce((n,a)=>n+a.risks,0)} hazard entries · mean reward ${(roll.agents.reduce((n,a)=>n+a.reward,0)/roll.agents.length).toFixed(1)} (undiscounted).`;if(n===roll.agents.length||roll.ticks>=400){stop();if(n<roll.agents.length)statusText+=' Stopped at 400 moves; unfinished trials count as failures.';}render();}
canvas.addEventListener('click',e=>{let rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;for(const l of layout){let c=Math.floor((x-l.left)/l.size),r=Math.floor((y-l.top)/l.size);if(c<0||c>=world.w||r<0||r>=world.h)continue;let s=r*world.w+c;inspectingPI=section>=3&&l===layout[0];stop();if(section===3&&$('brush').value!=='inspect'&&!world.terminal(s)&&s!==world.start){let type=$('brush').value==='wall'?1:2;world.cells[s]=world.cells[s]===type?0:type;world.compile();planner=new Planner(world);other=new Planner(world,'pi');rescueVI=[];rescuePI=[];roll=null;statusText='Map edited. Values reset to zero; recompute the policy for the new model.';}else{statusText=`Inspecting cell (${r}, ${c}). Playback paused.`;$('numbers-tab').click();}selected=s;render();break;}});
$('seed-choice').addEventListener('change',reset);
$('sample-values').onclick=()=>{
 freshSeed=Math.floor(Math.random()*4294967296);
 $('seed-choice').value='fresh';
 reset();
};
for(const id of ['gamma','slip','size','method','map'])$(id).addEventListener('change',reset);
$('reset').onclick=reset;$('step').onclick=()=>{stop();step();};$('play').onclick=()=>{if(playing||finishing)stop();else{if(completed()&&!roll)return;playing=true;lastTime=0;$('play').textContent='Ⅱ Pause';}};
$('finish').onclick=()=>{if(section===1&&planner.phase==='improve'){stop();step();return;}roll=null;playing=false;finishing=true;$('play').textContent='Ⅱ Pause';statusText='Computing in short batches; Pause remains available.';render();};$('rollout').onclick=startRollout;
$('idea-tab').onclick=()=>{ $('idea').hidden=false;$('numbers').hidden=true;$('idea-tab').classList.add('selected');$('numbers-tab').classList.remove('selected');};$('numbers-tab').onclick=()=>{$('explanation-drawer').open=true;$('idea').hidden=true;$('numbers').hidden=false;$('numbers-tab').classList.add('selected');$('idea-tab').classList.remove('selected');};
$('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{statusText='Use your browser’s full-screen menu for this window.';render();}};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.key==='ArrowRight')selectSection((section+1)%5);if(e.key==='ArrowLeft')selectSection((section+4)%5);});
$('explanation-drawer').addEventListener('toggle',()=>{if($('explanation-drawer').open)stop();drawChart();});
window.addEventListener('resize',()=>{drawChart();draw(performance.now());});
function frame(now){if(finishing){let start=performance.now();do{step(false);}while(finishing&&performance.now()-start<12);render();}else if(playing&&now-lastTime>=+$('tempo').value){lastTime=now;if(roll)rollStep(now);else step();}if(playing||finishing||now-changedAt<700||(roll&&now-roll.time<250))draw(now);requestAnimationFrame(frame);}selectSection(0);requestAnimationFrame(frame);
