/* UI for the explicit forward/backward model in model.js. */
(() => {
  'use strict';
  const M = window.BackpropModel, $ = id => document.getElementById(id);
  const sup = ['⁰', '¹', '²', '³'], sub = ['₁', '₂'];
  const fmt = (v, digits = 4) => Math.abs(v) < 1e-12 ? '0' : Math.abs(v) < 0.0001 ? v.toExponential(2) : Number(v.toFixed(digits)).toString();
  const pname = p => p.type === 'b' ? `b${sup[p.l]}${sub[p.j]}` : `W${sup[p.l]}${sub[p.j]},${sub[p.i]}`;
  const aname = (l, j) => l === 0 ? `x${sub[j]}` : l === 3 ? 'y′' : `z${sup[l]}${sub[j]}`;
  const uname = (l, j) => `u${sup[l + 1]}${sub[j]}`;
  let state = M.preset(), result, selected = { type: 'w', l: 0, j: 0, i: 0 }, phase = 0, timer = null;
  let history = [M.evaluate(state).loss], undo = [];
  const titles = ['Forward', 'Loss', 'Output', 'Hidden 2', 'Hidden 1', 'Update'];
  const descriptions = [
    ['Cache the forward values', 'Each neuron forms u = Σ w × input + b, then applies an activation. Work left to right to compute y′ and L. These saved values are reused in the backward pass.'],
    ['Start at the loss', 'The derivative ∂L/∂L is 1. For squared loss, the first signal is ∂L/∂y′ = 2(y′ − y). No weights have changed.'],
    ['Differentiate the output neuron', 'Multiply the loss signal by the output sigmoid derivative. This gives δ³. Each output weight then receives δ³ times its saved input.'],
    ['Send gradients to hidden layer 2', 'Pass δ³ back through the output weights. Multiply by each hidden activation derivative to obtain δ², then by the saved input to differentiate W¹.'],
    ['Add paths at hidden layer 1', 'Each first-layer neuron feeds BOTH second-layer neurons. Add the two downstream contributions, multiply by its local activation derivative, then by its input. This is the chain rule on a branching graph.'],
    ['Use the gradients to update parameters', 'Backpropagation is complete. Gradient descent subtracts η times each gradient. Use “Apply 1 update” below to change all parameters together, then recompute the forward pass.']
  ];
  const svgNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, text) { const e = document.createElementNS(svgNS, tag); Object.entries(attrs || {}).forEach(([k,v]) => e.setAttribute(k,v)); if (text !== undefined) e.textContent = text; return e; }
  function buttonize(e, label, handler) { e.setAttribute('tabindex','0'); e.setAttribute('role','button'); e.setAttribute('aria-label',label); e.addEventListener('click',handler); e.addEventListener('keydown', event => {if(event.key==='Enter'||event.key===' '){event.preventDefault();handler();}}); }
  function stop() { clearInterval(timer); timer = null; $('play').textContent = '▶ Play walkthrough'; $('play').setAttribute('aria-pressed','false'); }
  function visibleLayer(l) { return phase >= 4 - l; }
  function focusParameter(p) { stop(); selected = p; phase = 4 - p.l; render(); }
  function position(l,j) { return {x:[65,295,525,750][l],y:l===3?181:j===0?103:266}; }
  function pathEdge(l,j,i) {
    if (phase < 2) return false;
    if (selected.type === 'node') {
      const sl = selected.l;
      return l >= sl && (l !== sl || i === selected.j);
    }
    if (l === selected.l) return selected.type === 'w' && j === selected.j && i === selected.i;
    return l > selected.l && (l !== selected.l + 1 || i === selected.j);
  }
  function drawNetwork() {
    const svg = $('network'); svg.querySelectorAll(':scope > :not(title):not(desc)').forEach(n=>n.remove());
    const defs=el('defs'); ['forward','back'].forEach((id,idx)=>{const marker=el('marker',{id,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});marker.append(el('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:idx?'#b54923':'#a6b5bf'}));defs.append(marker);});svg.append(defs);
    const labels=['Inputs','Hidden layer 1','Hidden layer 2','Output'];
    labels.forEach((name,l)=>{svg.append(el('text',{x:position(l,0).x,y:22,'text-anchor':'middle',class:'column-label'},name));svg.append(el('text',{x:position(l,0).x,y:41,'text-anchor':'middle',class:'column-sub'},l===0?'given':l===3?'sigmoid':state.activation));});
    for(let l=0;l<3;l++)state.W[l].forEach((row,j)=>row.forEach((w,i)=>{
      const a=position(l,i),b=position(l+1,j),dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy),ux=dx/len,uy=dy/len;
      const x1=a.x+39*ux,y1=a.y+39*uy,x2=b.x-41*ux,y2=b.y-41*uy;
      const p={type:'w',l,j,i},active=pathEdge(l,j,i),chosen=selected.type==='w'&&selected.l===l&&selected.j===j&&selected.i===i;
      const g=el('g',{class:'edge-group'}),coords={x1,y1,x2,y2};
      g.append(el('line',{...coords,class:`edge ${active?'path':''} ${chosen?'chosen':''}`,'marker-end':'url(#forward)'}));
      if(active)g.append(el('line',{x1:x2,y1:y2-5,x2:x1,y2:y1-5,stroke:'#b54923','stroke-width':1.5,'stroke-dasharray':'4 4','marker-end':'url(#back)'}));
      g.append(el('line',{...coords,class:'edge-hit'}));
      const t=l===2?0.43:i===j?0.51:(i===0?0.31:0.69);
      const label=el('text',{x:a.x+dx*t,y:a.y+dy*t-10,'text-anchor':'middle',class:`edge-label ${active||chosen?'active':''}`},`${pname(p)} ${fmt(w,2)}`);
      g.append(label);buttonize(g,`${pname(p)}, value ${fmt(w)}. Inspect chain rule.`,()=>focusParameter(p));svg.append(g);
    }));
    for(let l=0;l<4;l++)result.a[l].forEach((a,j)=>{
      const pos=position(l,j),active=selected.type==='node'?selected.l===l&&selected.j===j:selected.l+1===l&&selected.j===j;
      const g=el('g',{class:`node ${active?'active':''}`});g.append(el('circle',{cx:pos.x,cy:pos.y,r:38}));
      g.append(el('text',{x:pos.x,y:pos.y-7,'text-anchor':'middle',class:'node-name'},aname(l,j)));
      g.append(el('text',{x:pos.x,y:pos.y+15,'text-anchor':'middle',class:'node-value'},fmt(a,3)));
      if((l===3&&phase>=1)||(l<3&&phase>=4-l))g.append(el('text',{x:pos.x,y:pos.y+58,'text-anchor':'middle',class:'node-gradient'},`∂L/∂${aname(l,j)} = ${fmt(result.da[l][j],3)}`));
      buttonize(g,`${aname(l,j)}, value ${fmt(a)}. Inspect neuron.`,()=>{stop();selected={type:'node',l,j};phase=l===0?4:5-l;render();});svg.append(g);
    });
    svg.append(el('line',{x1:790,y1:181,x2:842,y2:181,class:'edge','marker-end':'url(#forward)'}));
    const loss=el('g',{class:'loss-node'});loss.append(el('rect',{x:847,y:145,width:80,height:72,rx:9}));loss.append(el('text',{x:887,y:171,'text-anchor':'middle',class:'node-name'},'L'));loss.append(el('text',{x:887,y:195,'text-anchor':'middle',class:'node-value'},fmt(result.loss,3)));svg.append(loss);
    svg.append(el('text',{x:887,y:240,'text-anchor':'middle',class:'column-sub'},`y = ${fmt(state.y)}`));
    if(phase>0){svg.append(el('line',{x1:887,y1:344,x2:phase<3?750:phase===3?525:295,y2:344,stroke:'#b54923','stroke-width':2,'marker-end':'url(#back)'}));svg.append(el('text',{x:575,y:374,'text-anchor':'middle',class:'flow-label'},'Backward: multiply local derivatives; add where paths branch.'));}
  }
  const box=(label,eq)=>`<div class="formula-box"><span class="label">${label}</span><div class="equation">${eq}</div></div>`;
  function upstream(l,j) {
    if(l===2)return box('Incoming gradient from squared loss',`∂L/∂y′ = 2(${fmt(result.prediction)} − ${fmt(state.y)})<br>= ${fmt(result.da[3][0])}`);
    const terms=state.W[l+1].map((row,k)=>`<div class="sum-term">via ${aname(l+2,k)}: δ${sup[l+2]}${sub[k]} × W${sup[l+1]}${sub[k]},${sub[j]}<br>${fmt(result.d[l+1][k])} × ${fmt(row[j])} = ${fmt(result.d[l+1][k]*row[j])}</div>`).join('');
    return `<div><h3>${state.W[l+1].length>1?'Add the downstream paths':'Follow the downstream path'}</h3>${terms}<p class="equation">∂L/∂${aname(l+1,j)} = <strong>${fmt(result.da[l+1][j])}</strong></p></div>`;
  }
  function inspect() {
    const p=selected;
    if(p.type==='node') {inspectNode(p);return;}
    const {l,j}=p,source=p.type==='b'?'1':aname(l,p.i),value=M.get(state,p),g=M.gradient(result,p),incoming=result.da[l+1][j],local=M.slope(result.u[l][j],result.a[l+1][j],l===2?'sigmoid':state.activation),input=p.type==='b'?1:result.a[l][p.i];
    $('inspect-title').textContent=pname(p);
    $('inspect-context').textContent=p.type==='b'?`Bias of ${aname(l+1,j)}. A bias has a constant input of 1.`:`${source} → ${aname(l+1,j)} · select any other connection to change the trace.`;
    let html=box('Forward computation',`${uname(l,j)} = Σ w × input + b = ${fmt(result.u[l][j])}<br>${aname(l+1,j)} = ${l===2?'sigmoid':state.activation}(${uname(l,j)}) = ${fmt(result.a[l+1][j])}`);
    if(visibleLayer(l)){
      html=`<details class="inspect-details"><summary>Saved forward values</summary>${html}</details>`;
      html+=upstream(l,j);
      html+=`<div><h3>Three factors, one derivative</h3><div class="factors"><div class="factor"><span>Incoming gradient · ∂L/∂${aname(l+1,j)}</span><strong>${fmt(incoming)}</strong><small>Sum every downstream contribution.</small></div><div class="factor"><span>Local activation slope · ∂${aname(l+1,j)}/∂${uname(l,j)}</span><strong>${fmt(local)}</strong><small>${l===2||state.activation==='sigmoid'?'σ(u)(1 − σ(u))':state.activation==='tanh'?'1 − tanh²(u)':'1 if u > 0; otherwise 0 (chosen at zero)'}</small></div><div class="factor"><span>Saved input · ∂${uname(l,j)}/∂${pname(p)}</span><strong>${fmt(input)}</strong><small>${source}${p.type==='b'?' · bias derivative is always 1':''}</small></div></div></div>`;
      html+=box('Chain rule',`∂L/∂${pname(p)}<br><span class="symbolic">= (∂L/∂${aname(l+1,j)}) × (∂${aname(l+1,j)}/∂${uname(l,j)}) × (∂${uname(l,j)}/∂${pname(p)})</span><br>= ${fmt(incoming)} × ${fmt(local)} × ${fmt(input)}<br><span class="result">= ${fmt(g,7)}</span>`);
      const n=M.numerical(state,p),diff=Math.abs(n.value-g);
      html+=`<details class="inspect-details"><summary>Verify with finite differences ${n.kink?'· kink':''}</summary><div class="formula-box"><span class="label">Check with a tiny perturbation</span><div class="equation">[L(w + ε) − L(w − ε)] / (2ε)</div><p>ε = 0.00001; all other parameters fixed.</p><div class="check ${n.kink?'warning':'pass'}">${n.kink?'ReLU kink: a centered difference may disagree with the chosen derivative.':`Numerical: ${fmt(n.value,7)}<br>Backprop: ${fmt(g,7)}<br>Absolute difference: ${diff.toExponential(1)}`}</div></div></details>`;
      if(phase===5)html+=box('Gradient-descent preview',`${pname(p)} ← w − η × ∂L/∂w<br>${fmt(value)} − ${state.rate} × (${fmt(g)})<br>= ${fmt(value-state.rate*g,6)}`);
    }else{
      html+=`<div class="formula-box"><h3>Ready to trace this weight?</h3><p>The forward pass saves the input and activation. Step through the backward pass to see how the loss depends on this parameter.</p><button id="trace-selected" class="primary">Trace its gradient ←</button></div>`;
      if(phase===1)html+=box('The backward seed',`∂L/∂L = 1<br>∂L/∂y′ = 2(y′ − y) = ${fmt(result.da[3][0])}`);
    }
    html+=`<div class="inspect-edit"><label>Change ${pname(p)}<input id="selected-value" aria-label="Value of selected parameter" type="number" min="-20" max="20" step="any" value="${value}"></label><button id="set-parameter">Set value</button></div>`;
    $('inspection').innerHTML=html;
    if($('trace-selected'))$('trace-selected').onclick=()=>focusParameter(selected);
    $('set-parameter').onclick=()=>{const input=$('selected-value');if(!input.reportValidity()||input.value==='')return;M.set(state,p,input.valueAsNumber);restartHistory();render();};
  }
  function inspectNode(p) {
    const {l,j}=p,name=aname(l,j);$('inspect-title').textContent=name;$('inspect-context').textContent=l===0?'An input is given, not trained. Its derivative measures sensitivity.':'A neuron combines inputs, then applies one activation.';
    if(l===0){$('inspection').innerHTML=box('Input value',`${name} = ${fmt(state.x[j])}`)+`<h3>Add both paths leaving this input</h3>`+state.W[0].map((row,k)=>box(`Via ${aname(1,k)}`,`${fmt(result.d[0][k])} × ${fmt(row[j])} = ${fmt(result.d[0][k]*row[j])}`)).join('')+box('Input sensitivity',`∂L/∂${name} = ${fmt(result.da[0][j],7)}`);return;}
    const layer=l-1,kind=l===3?'sigmoid':state.activation;
    const terms=state.W[layer][j].map((w,i)=>`${fmt(w)} × ${fmt(result.a[layer][i])}`).join(' + ');
    $('inspection').innerHTML=box('Weighted sum',`${uname(layer,j)} = ${terms} + (${fmt(state.b[layer][j])})<br>= ${fmt(result.u[layer][j])}`)+box('Activation',`${name} = ${kind}(${fmt(result.u[layer][j])})<br>= ${fmt(result.a[l][j])}`)+upstream(layer,j)+box('Local backward step',`δ${sup[l]}${sub[j]} = ∂L/∂${name} × ${kind}′(u)<br>= ${fmt(result.da[l][j])} × ${fmt(M.slope(result.u[layer][j],result.a[l][j],kind))}<br><span class="result">= ${fmt(result.d[layer][j],7)}</span>`)+`<p>This δ is reused for every incoming weight: multiply it by that weight’s saved input. The bias gradient equals δ.</p><button id="node-bias">Inspect this neuron’s bias</button>`;
    $('node-bias').onclick=()=>focusParameter({type:'b',l:layer,j});
  }
  function drawHistory() {
    const svg=$('loss-chart');svg.replaceChildren();const max=Math.max(...history,0.001)*1.12,n=history.length-1;
    for(let i=0;i<=3;i++){const y=145-i*40;svg.append(el('line',{x1:57,y1:y,x2:420,y2:y,stroke:'#dee4e3'}));svg.append(el('text',{x:49,y:y+4,'text-anchor':'end',class:'chart-text'},fmt(max*i/3,3)));}
    const points=history.map((v,i)=>`${57+(n?i/n:0)*363},${145-v/max*120}`);
    svg.append(el('polyline',{points:points.join(' '),fill:'none',stroke:'#087c83','stroke-width':2.5}));
    const [cx,cy]=points[points.length-1].split(',');svg.append(el('circle',{cx,cy,r:4,fill:'#087c83'}));
    svg.append(el('text',{x:57,y:165,class:'chart-text'},'0'));if(n)svg.append(el('text',{x:420,y:165,'text-anchor':'end',class:'chart-text'},String(n)));
    svg.append(el('text',{x:240,y:183,'text-anchor':'middle',class:'chart-text'},'Gradient descent updates'));svg.append(el('text',{x:8,y:15,class:'chart-text'},'Loss'));
    $('update-count').textContent=`${n} update${n===1?'':'s'}`;
  }
  function drawParameters() {
    $('parameter-tables').innerHTML=[0,1,2].map(l=>`<div><h3>${l===2?'Output layer':`Hidden layer ${l+1}`}</h3><table><thead><tr><th>Parameter</th><th>Value</th><th>∂L/∂parameter</th></tr></thead><tbody>${M.parameters(state).filter(p=>p.l===l).map(p=>`<tr><td><button data-param='${JSON.stringify(p)}'>${pname(p)}</button></td><td><input aria-label="${pname(p)} value" data-edit='${JSON.stringify(p)}' type="number" min="-20" max="20" step="any" value="${M.get(state,p)}"></td><td>${fmt(M.gradient(result,p),6)}</td></tr>`).join('')}</tbody></table></div>`).join('');
    document.querySelectorAll('[data-param]').forEach(e=>e.onclick=()=>focusParameter(JSON.parse(e.dataset.param)));
    document.querySelectorAll('[data-edit]').forEach(e=>e.onchange=()=>{if(!e.reportValidity()||e.value==='')return;M.set(state,JSON.parse(e.dataset.edit),e.valueAsNumber);restartHistory();render();});
  }
  function render() {
    result=M.evaluate(state);$('x1-value').textContent=fmt(state.x[0],2);$('x2-value').textContent=fmt(state.x[1],2);
    $('prediction').textContent=fmt(result.prediction,5);$('target-display').textContent=fmt(state.y);$('loss').textContent=fmt(result.loss,6);
    $('step-count').textContent=`STEP ${phase+1} OF 6 · ${phase===0?'FORWARD PASS':phase===5?'GRADIENT DESCENT':'BACKWARD PASS'}`;
    $('step-title').textContent=descriptions[phase][0];$('step-description').textContent=descriptions[phase][1];
    [...$('steps').children].forEach((e,i)=>{if(i===phase)e.setAttribute('aria-current','step');else e.removeAttribute('aria-current');});
    $('previous').disabled=phase===0;$('next').disabled=phase===5;$('undo').disabled=!undo.length;
    drawNetwork();inspect();drawHistory();drawParameters();
  }
  function setPhase(p) {phase=p;if(p===2)selected={type:'w',l:2,j:0,i:0};if(p===3)selected={type:'w',l:1,j:0,i:0};if(p===4)selected={type:'w',l:0,j:0,i:0};render();}
  function restartHistory(){stop();history=[M.evaluate(state).loss];undo=[];$('train-message').textContent='Values changed. A new loss history starts from this network.';}
  function syncControls(){ $('activation').value=state.activation;$('x1').value=state.x[0];$('x2').value=state.x[1];$('target').value=state.y;$('rate').value=String(state.rate); }
  function reset(name){stop();state=M.preset(name);phase=0;selected={type:'w',l:0,j:0,i:0};history=[M.evaluate(state).loss];undo=[];syncControls();$('train-message').textContent='No updates yet. Backpropagation computes gradients; gradient descent changes the parameters.';render();}
  function train(n){
    stop();const before=M.evaluate(state).loss;undo.push({state:M.clone(state),history:history.slice()});if(undo.length>50)undo.shift();
    for(let i=0;i<n;i++){const next=M.update(state);if(!Number.isFinite(M.evaluate(next).loss)){ $('train-message').textContent='Update stopped: non-finite value. Reduce the learning rate.';render();return;}state=next;history.push(M.evaluate(state).loss);}
    phase=0;const after=history[history.length-1];$('train-message').textContent=`Applied ${n} simultaneous update${n===1?'':'s'}. Loss: ${fmt(before,6)} → ${fmt(after,6)}. ${after>before?'Loss increased; try a smaller learning rate.':'The graph now shows the new forward pass.'}`;render();
  }
  titles.forEach((title,i)=>{const b=document.createElement('button');b.textContent=`${i+1}. ${title}`;b.onclick=()=>{stop();setPhase(i);};$('steps').append(b);});
  $('previous').onclick=()=>{stop();setPhase(Math.max(0,phase-1));};$('next').onclick=()=>{stop();setPhase(Math.min(5,phase+1));};
  $('play').onclick=()=>{if(timer){stop();return;}if(phase===5)setPhase(0);$('play').textContent='Ⅱ Pause';$('play').setAttribute('aria-pressed','true');timer=setInterval(()=>{setPhase(Math.min(5,phase+1));if(phase===5)stop();},2400);};
  $('preset').onchange=e=>reset(e.target.value);$('reset').onclick=()=>{$('preset').value='standard';reset('standard');};
  $('activation').onchange=e=>{state.activation=e.target.value;restartHistory();render();};
  ['x1','x2'].forEach((id,i)=>$(id).oninput=e=>{state.x[i]=Number(e.target.value);restartHistory();render();});
  $('target').onchange=e=>{if(!e.target.reportValidity()||e.target.value==='')return;state.y=e.target.valueAsNumber;restartHistory();render();};
  $('rate').onchange=e=>{state.rate=Number(e.target.value);render();};
  $('train-one').onclick=()=>train(1);$('train-many').onclick=()=>train(20);
  $('undo').onclick=()=>{if(!undo.length)return;stop();const snapshot=undo.pop();state=snapshot.state;history=snapshot.history;phase=0;syncControls();$('train-message').textContent='Restored the network and loss history before the last update action.';render();};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  reset('standard');
})();
