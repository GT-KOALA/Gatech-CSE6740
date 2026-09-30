/* Symbolic explanations with an explicit, progressive forward/backward walkthrough. */
(() => {
  'use strict';
  const M = window.BackpropModel, $ = id => document.getElementById(id);
  const sup = ['⁰', '¹', '²', '³'], sub = ['₁', '₂'];
  const fmt = (v, digits = 4) => Math.abs(v) < 1e-12 ? '0' : Math.abs(v) < 0.0001 ? v.toExponential(2) : Number(v.toFixed(digits)).toString();
  const pname = p => p.type === 'b' ? `b${sup[p.l]}${sub[p.j]}` : `W${sup[p.l]}${sub[p.j]},${sub[p.i]}`;
  const aname = (l, j) => l === 0 ? `x${sub[j]}` : l === 3 ? 'y′' : `z${sup[l]}${sub[j]}`;
  const uname = (l, j) => `u${sup[l + 1]}${sub[j]}`;
  let state = M.preset(), result, selected = { type: 'w', l: 0, j: 0, i: 0 };
  let phase = 0, forwardStage = 0, lossStage = 0, timer = null;
  let history = [M.evaluate(state).loss], undo = [];
  const titles = ['Forward', 'Loss', 'Output', 'Hidden 2', 'Hidden 1', 'Update'];
  const forwardLabels = ['Inputs', 'Hidden 1', 'Hidden 2', 'Prediction'];
  const forwardCopy = [
    ['Begin with the inputs', 'Only the inputs are known. Advance to compute hidden layer 1. Uncomputed neurons show “…”; blue arrows will show which values are being used.'],
    ['Compute hidden layer 1', 'Blue arrows carry the inputs into the first hidden layer. Each neuron takes a weighted sum, adds its bias, and applies the hidden activation. Save these outputs for the next layer.'],
    ['Compute hidden layer 2', 'The first hidden layer’s outputs become the next layer’s inputs. Repeat the weighted sum and activation, moving left to right.'],
    ['Compute the prediction', 'The output neuron combines the second hidden layer’s values and applies sigmoid to produce y′. The next step compares this prediction with the target to compute the loss.']
  ];
  const descriptions = [null, null,
    ['Differentiate the output neuron', 'Start with ∂L/∂y′ from the loss. Multiply by the output activation’s derivative, then by the derivative of the weighted sum with respect to the selected weight.'],
    ['Continue into hidden layer 2', 'Follow the orange arrows from the loss toward the selected weight. The incoming gradient already accounts for the output neuron; multiply it by the two local derivatives here.'],
    ['Add contributions from both branches', 'Each first-layer neuron affects both second-layer neurons. Add the gradients returning along those two branches, then multiply by the local activation and input derivatives.'],
    ['Use the gradients to change the weights', 'Backpropagation tells us how each parameter affects the loss. Gradient descent subtracts the learning rate times that gradient. Use “Apply 1 update” below to update all parameters together.']
  ];
  const svgNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, text) {
    const e = document.createElementNS(svgNS, tag);
    Object.entries(attrs || {}).forEach(([k,v]) => e.setAttribute(k,v));
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function buttonize(e, label, handler) {
    e.setAttribute('tabindex','0'); e.setAttribute('role','button'); e.setAttribute('aria-label',label);
    e.addEventListener('click',handler);
    e.addEventListener('keydown', event => { if(event.key==='Enter'||event.key===' '){event.preventDefault();handler();} });
  }
  function stop() { clearInterval(timer); timer = null; $('play').textContent = '▶ Play walkthrough'; $('play').setAttribute('aria-pressed','false'); }
  function focusParameter(p) { stop(); selected = p; phase = 4 - p.l; render(); }
  function position(l,j) { return {x:[70,295,520,740][l],y:l===3?182:j===0?105:267}; }
  function pathEdge(l,j,i) {
    if (phase < 2) return false;
    if (selected.type === 'node') return l >= selected.l && (l !== selected.l || i === selected.j);
    if (l === selected.l) return selected.type === 'w' && j === selected.j && i === selected.i;
    return l > selected.l && (l !== selected.l + 1 || i === selected.j);
  }
  function nodeOnPath(l,j) {
    if(phase===1)return lossStage===1&&l===3;
    if(phase<2)return false;
    const start=selected.type==='node'?selected.l:selected.l+1;
    return l>=start&&(l!==start||j===selected.j);
  }
  function drawNetwork() {
    const svg = $('network'); svg.querySelectorAll(':scope > :not(desc)').forEach(n=>n.remove());
    svg.dataset.phase = phase; svg.dataset.forwardStage = forwardStage; svg.dataset.lossStage = lossStage;
    const defs=el('defs');
    [['forward','#226fba'],['back','#b54923']].forEach(([id,color])=>{
      const marker=el('marker',{id,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});
      marker.append(el('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:color}));defs.append(marker);
    });svg.append(defs);
    ['Inputs','Hidden layer 1','Hidden layer 2','Prediction'].forEach((name,l)=>{
      svg.append(el('text',{x:position(l,0).x,y:22,'text-anchor':'middle',class:'column-label'},name));
      svg.append(el('text',{x:position(l,0).x,y:42,'text-anchor':'middle',class:'column-sub'},l===0?'given':l===3?'sigmoid':state.activation));
    });
    svg.append(el('text',{x:885,y:22,'text-anchor':'middle',class:'column-label'},'Loss'));
    for(let l=0;l<3;l++)state.W[l].forEach((row,j)=>row.forEach((w,i)=>{
      const a=position(l,i),b=position(l+1,j),dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy),ux=dx/len,uy=dy/len;
      const x1=a.x+39*ux,y1=a.y+39*uy,x2=b.x-41*ux,y2=b.y-41*uy;
      const p={type:'w',l,j,i},back=pathEdge(l,j,i),forward=phase===0&&forwardStage===l+1;
      const computed=phase>0||forwardStage>l+1;
      const chosen=phase>=2&&selected.type==='w'&&selected.l===l&&selected.j===j&&selected.i===i;
      const g=el('g',{class:'edge-group','data-layer':l});
      // Exactly one visible line per connection. Reverse its geometry for backpropagation.
      const coords=back?{x1:x2,y1:y2,x2:x1,y2:y1}:{x1,y1,x2,y2};
      const line=el('line',{...coords,class:`edge ${back?'backward':forward?'forward':computed?'computed':'pending'} ${chosen?'selected-edge':''}`});
      if(back||forward)line.setAttribute('marker-end',`url(#${back?'back':'forward'})`);
      g.append(line);g.append(el('line',{x1,y1,x2,y2,class:'edge-hit'}));
      const t=l===2?0.43:i===j?0.51:(i===0?0.31:0.69);
      g.append(el('text',{x:a.x+dx*t,y:a.y+dy*t-11,'text-anchor':'middle',class:`edge-label ${back?'active':''} ${phase===0&&forwardStage<=l?'uncomputed-label':''}`},pname(p)));
      buttonize(g,`${pname(p)}. Trace this weight’s gradient.`,()=>focusParameter(p));svg.append(g);
    }));
    for(let l=0;l<4;l++)result.a[l].forEach((a,j)=>{
      const pos=position(l,j),ready=phase>0||forwardStage>=l;
      const active=phase>=2&&(selected.type==='node'?selected.l===l&&selected.j===j:selected.l+1===l&&selected.j===j);
      const current=phase===0&&forwardStage===l;
      const g=el('g',{class:`node ${active?'active':''} ${current?'forward-current':''} ${ready?'':'uncomputed'}`,'data-layer':l,'data-neuron':j});
      g.append(el('circle',{cx:pos.x,cy:pos.y,r:38}));
      g.append(el('text',{x:pos.x,y:pos.y-7,'text-anchor':'middle',class:'node-name'},aname(l,j)));
      g.append(el('text',{x:pos.x,y:pos.y+15,'text-anchor':'middle',class:'node-value'},ready?fmt(a,3):'…'));
      if(nodeOnPath(l,j))
        g.append(el('text',{x:pos.x,y:pos.y+60,'text-anchor':'middle',class:'node-gradient'},`∂L/∂${aname(l,j)}`));
      buttonize(g,`${aname(l,j)}. Trace this neuron’s gradient.`,()=>{stop();selected={type:'node',l,j};phase=l===0?4:5-l;render();});svg.append(g);
    });
    const backwardLoss=phase>1||(phase===1&&lossStage===1),forwardLoss=phase===1&&lossStage===0;
    const lossLine=el('line',{...(backwardLoss?{x1:841,y1:182,x2:782,y2:182}:{x1:780,y1:182,x2:841,y2:182}),class:`edge ${backwardLoss?'backward':forwardLoss?'forward':'pending'}`});
    if(backwardLoss||forwardLoss)lossLine.setAttribute('marker-end',`url(#${backwardLoss?'back':'forward'})`);svg.append(lossLine);
    const loss=el('g',{class:`loss-node ${forwardLoss?'forward-current':''} ${backwardLoss?'active':''} ${phase===0?'uncomputed':''}`});
    loss.append(el('rect',{x:847,y:146,width:76,height:72,rx:9}));
    loss.append(el('text',{x:885,y:172,'text-anchor':'middle',class:'node-name'},'L'));
    loss.append(el('text',{x:885,y:196,'text-anchor':'middle',class:'node-value'},phase===0?'…':fmt(result.loss,3)));svg.append(loss);
    if(phase>0){
      svg.append(el('line',{x1:885,y1:266,x2:885,y2:222,class:`edge ${forwardLoss?'forward':'computed'}`,...(forwardLoss?{'marker-end':'url(#forward)'}:{})}));
      svg.append(el('text',{x:885,y:287,'text-anchor':'middle',class:'column-sub'},`Target y = ${fmt(state.y)}`));
    }
    const caption=phase===0?`Forward → ${forwardLabels[forwardStage]}${forwardStage===3?' computed; loss comes next':''}`:forwardLoss?'Compare prediction with target → compute L':backwardLoss&&phase===1?'← Send ∂L/∂y′ from the loss to the output':'← Backward: gradients flow toward the selected neuron or weight';
    svg.append(el('text',{x:470,y:368,'text-anchor':'middle',class:`flow-label ${phase===0||forwardLoss?'forward-caption':''}`},caption));
  }
  const box=(label,eq)=>`<div class="formula-box"><span class="label">${label}</span><div class="equation">${eq}</div></div>`;
  const derivative=(a,b)=>`<span class="fraction" role="math" aria-label="partial derivative of ${a} with respect to ${b}"><span aria-hidden="true">∂${a}</span><span aria-hidden="true">∂${b}</span></span>`;
  function factor(title,formula,explanation){return `<div class="factor"><span class="factor-title">${title}</span><div class="factor-formula">${formula}</div><p>${explanation}</p></div>`;}
  function product(lhs,terms){return `<div class="chain-product">${derivative('L',lhs)}<span>=</span><div class="chain-terms">${terms.join('<span class="multiply">×</span>')}</div></div>`;}
  function slopeExplanation(l,j){
    const a=aname(l+1,j),u=uname(l,j),kind=l===2?'sigmoid':state.activation;
    return kind==='sigmoid'?`For sigmoid, this is ${a}(1 − ${a}).`:
      kind==='tanh'?`For tanh, this is 1 − (${a})².`:`For ReLU, this is 1 when ${u} > 0 and 0 when ${u} < 0. At zero, we choose 0.`;
  }
  function upstream(l,j){
    const a=aname(l+1,j);
    if(l===2)return `<div class="upstream-explanation">${box('From the loss',`${derivative('L','y′')} = 2(y′ − y)`)}<p>This is the gradient arriving from the loss, before we differentiate the output neuron.</p></div>`;
    const terms=state.W[l+1].map((_,k)=>{
      const u=uname(l+1,k);
      return `<div class="branch-term"><span class="branch-label">Via ${aname(l+2,k)}</span><div>${derivative('L',u)}<span class="multiply">×</span>${derivative(u,a)}</div></div>`;
    });
    return `<div class="upstream-explanation"><h3>${terms.length>1?'The incoming gradient adds both paths':'Where the incoming gradient comes from'}</h3><div class="branch-sum">${derivative('L',a)}<span>=</span><div>${terms.join('<span class="branch-plus">+</span>')}</div></div><p>Each path contributes “gradient from the next neuron × connecting weight.” ${terms.length>1?'Add these contributions because both paths depend on the same neuron.':''}</p></div>`;
  }
  function inspectForward(){
    $('inspect-title').textContent=['Read the inputs','Compute hidden layer 1','Compute hidden layer 2','Compute the prediction'][forwardStage];
    $('inspect-context').textContent='Follow the blue arrows from left to right. Values appear only after their layer is computed.';
    const stages=[
      box('Given quantities','Input x; target y')+'<p>The network starts from x. The target y is held aside until we compare it with the prediction in step 2.</p>',
      box('Weighted sum','u¹ = W⁰x + b⁰')+box('Activation','z¹ = f(u¹)'),
      box('Weighted sum','u² = W¹z¹ + b¹')+box('Activation','z² = f(u²)'),
      box('Weighted sum','u³ = W²z² + b²')+box('Prediction','y′ = σ(u³)')
    ];
    $('inspection').innerHTML=stages[forwardStage]+`<p class="notation-note">W is a matrix of weights; b is a bias. u is the weighted sum before activation; z is a hidden layer’s output. f is the chosen hidden activation; σ is sigmoid.</p><p>Save the inputs and activations. The backward pass will reuse them.</p>`;
  }
  function inspectLoss(){
    $('inspect-title').textContent=lossStage===0?'Measure the error':'Differentiate the loss';
    $('inspect-context').textContent=lossStage===0?'Compare the prediction y′ with the target y.':'Ask: if the prediction changes a little, how does the loss change?';
    $('inspection').innerHTML=lossStage===0?
      box('Squared error','L = (y′ − y)²')+'<p>Subtract the target from the prediction, then square the difference. The loss is zero for a perfect prediction and grows as they move apart.</p><p>The blue arrows bring the prediction and target into L. Next, start the backward pass by differentiating this loss.</p>':
      box('First backward gradient',`${derivative('L','y′')} = 2(y′ − y)`)+`<p>This derivative is the signal sent from the loss to the output neuron along the orange arrow.</p><div class="loss-directions"><p><strong>Prediction below target:</strong> the gradient is negative. Increasing the prediction would reduce the loss.</p><p><strong>Prediction above target:</strong> the gradient is positive. Decreasing the prediction would reduce the loss.</p><p><strong>Prediction equals target:</strong> the gradient is zero.</p></div><p>The target stays fixed. We now use the chain rule to find how each weight affects this loss.</p>`;
  }
  function inspect(){
    if(phase===0){inspectForward();return;}
    if(phase===1){inspectLoss();return;}
    const p=selected;
    if(p.type==='node'){inspectNode(p);return;}
    const {l,j}=p,a=aname(l+1,j),u=uname(l,j),name=pname(p),source=p.type==='b'?'1':aname(l,p.i);
    $('inspect-title').textContent=`How ${name} affects L`;
    $('inspect-context').textContent=p.type==='b'?`This bias shifts ${u}, which changes ${a}, which changes the loss.`:`${name} changes ${u}, which changes ${a}, which changes the loss.`;
    const terms=[derivative('L',a),derivative(a,u),derivative(u,name)];
    let html=box('Chain rule',product(name,terms));
    html+=`<div class="factors">`+
      factor('Incoming gradient',terms[0],`How this neuron affects the loss through all downstream paths.`)+
      factor('Activation derivative',terms[1],`How the neuron responds to its weighted sum. ${slopeExplanation(l,j)}`)+
      factor(p.type==='b'?'Bias derivative':'Input derivative',`${terms[2]}<span class="factor-equals">= ${source}</span>`,p.type==='b'?`The bias enters ${u} additively, so its local derivative is 1.`:`How this weight changes the weighted sum. Its derivative is the saved input, ${source}.`)+`</div>`;
    html+=upstream(l,j);
    html+=`<p class="notation-note">${u}: weighted sum before activation.<br>${a}: output after activation.</p>`;
    if(phase===5)html+=box('Gradient descent',`${name} ← ${name} − η ${derivative('L',name)}`);
    $('inspection').innerHTML=html;
  }
  function inspectNode(p){
    const {l,j}=p,a=aname(l,j);
    $('inspect-title').textContent=`How ${a} affects L`;
    $('inspect-context').textContent=l===0?'An input is given, not trained. Its gradient describes how sensitive the loss is to that input.':'First gather the gradient from downstream; then pass it through this neuron’s activation.';
    if(l===0){$('inspection').innerHTML=upstream(-1,j);return;}
    const u=uname(l-1,j),terms=[derivative('L',a),derivative(a,u)];
    $('inspection').innerHTML=box('Chain rule at this neuron',product(u,terms))+`<div class="factors">`+
      factor('Incoming gradient',terms[0],`How the loss changes with this neuron’s output ${a}.`)+
      factor('Activation derivative',terms[1],`How the output changes with its weighted sum ${u}. ${slopeExplanation(l-1,j)}`)+`</div>`+
      upstream(l-1,j)+`<p class="notation-note">${u} is the weighted sum before activation. Reuse its gradient for each incoming weight by multiplying by that weight’s input.</p><button id="node-bias">Trace this neuron’s bias</button>`;
    $('node-bias').onclick=()=>focusParameter({type:'b',l:l-1,j});
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
  function drawProgress(){
    const container=$('pass-progress');container.replaceChildren();container.hidden=phase>1;
    const labels=phase===0?forwardLabels:['Measure error','Differentiate loss'];
    const current=phase===0?forwardStage:lossStage;
    labels.forEach((label,i)=>{
      const b=document.createElement('button');b.textContent=label;
      if(i===current)b.setAttribute('aria-current','step');
      if(i<current)b.classList.add('completed');
      b.onclick=()=>{stop();if(phase===0)forwardStage=i;else lossStage=i;render();$('pass-progress').children[i].focus({preventScroll:true});};
      container.append(b);
    });
  }
  function render(){
    result=M.evaluate(state);$('x1-value').textContent=fmt(state.x[0],2);$('x2-value').textContent=fmt(state.x[1],2);
    $('prediction').textContent=phase===0&&forwardStage<3?'…':fmt(result.prediction,5);
    $('target-display').textContent=fmt(state.y);$('loss').textContent=phase===0?'…':fmt(result.loss,6);
    let copy=descriptions[phase];
    if(phase===0)copy=forwardCopy[forwardStage];
    if(phase===1)copy=lossStage===0?
      ['Measure how far the prediction is from the target','The prediction and target enter the loss together: L = (y′ − y)². This completes the forward calculation. Next we ask how changing the prediction would change this loss.']:
      ['Turn the error into a backward signal','Differentiate L with respect to the prediction: ∂L/∂y′ = 2(y′ − y). The orange arrow sends this gradient to the output neuron. It tells us which direction of change in y′ would increase the loss.'];
    const pass=phase===0?'FORWARD PASS':phase===1&&lossStage===0?'COMPUTE THE LOSS':phase===5?'GRADIENT DESCENT':'BACKWARD PASS';
    $('inspector-label').textContent=phase===0?'02 / FORWARD COMPUTATION':phase===1?'02 / UNDERSTAND THE LOSS':'02 / FOLLOW THE CHAIN RULE';
    $('step-count').textContent=`STEP ${phase+1} OF 6 · ${pass}`;
    $('step-title').textContent=copy[0];$('step-description').textContent=copy[1];
    [...$('steps').children].forEach((e,i)=>{if(i===phase)e.setAttribute('aria-current','step');else e.removeAttribute('aria-current');});
    $('previous').disabled=phase===0&&forwardStage===0;$('next').disabled=phase===5;
    $('next').textContent=phase===0?['Compute hidden 1 →','Compute hidden 2 →','Compute prediction →','Compute loss →'][forwardStage]:
      phase===1?(lossStage===0?'Differentiate loss ←':'Backpropagate through output ←'):phase===5?'Complete':'Next step ←';
    $('undo').disabled=!undo.length;$('replay-forward').hidden=phase!==0;
    $('graph-mode').textContent=phase===0?'BLUE → COMPUTE VALUES':phase===1&&lossStage===0?'BLUE → COMPUTE LOSS':'ORANGE ← TRACE GRADIENTS';
    drawProgress();drawNetwork();inspect();drawHistory();drawParameters();
  }
  function setPhase(p){
    phase=p;
    if(p===0)forwardStage=0;
    if(p===1)lossStage=0;
    if(p===2)selected={type:'w',l:2,j:0,i:0};
    if(p===3)selected={type:'w',l:1,j:0,i:0};
    if(p===4)selected={type:'w',l:0,j:0,i:0};
    render();
  }
  function advance(){
    if(phase===0&&forwardStage<3){forwardStage++;render();return;}
    if(phase===1&&lossStage===0){lossStage=1;render();return;}
    setPhase(Math.min(5,phase+1));
  }
  function retreat(){
    if(phase===0){forwardStage=Math.max(0,forwardStage-1);render();return;}
    if(phase===1&&lossStage===1){lossStage=0;render();return;}
    const previous=phase-1;setPhase(previous);
    if(previous===0){forwardStage=3;render();}
    if(previous===1){lossStage=1;render();}
  }
  function restartHistory(){stop();history=[M.evaluate(state).loss];undo=[];$('train-message').textContent='Values changed. A new loss history starts from this network.';}
  function syncControls(){ $('activation').value=state.activation;$('x1').value=state.x[0];$('x2').value=state.x[1];$('target').value=state.y;$('rate').value=String(state.rate); }
  function reset(name){stop();state=M.preset(name);phase=0;forwardStage=0;lossStage=0;selected={type:'w',l:0,j:0,i:0};history=[M.evaluate(state).loss];undo=[];syncControls();$('train-message').textContent='No updates yet. Backpropagation computes gradients; gradient descent changes the parameters.';render();}
  function train(n){
    stop();const before=M.evaluate(state).loss;undo.push({state:M.clone(state),history:history.slice()});if(undo.length>50)undo.shift();
    for(let i=0;i<n;i++){
      const next=M.update(state);
      if(!Number.isFinite(M.evaluate(next).loss)){$('train-message').textContent='Update stopped: non-finite value. Reduce the learning rate.';render();return;}
      state=next;history.push(M.evaluate(state).loss);
    }
    phase=1;lossStage=0;forwardStage=3;
    const after=history[history.length-1];$('train-message').textContent=`Applied ${n} simultaneous update${n===1?'':'s'}. Loss: ${fmt(before,6)} → ${fmt(after,6)}. ${after>before?'Loss increased; try a smaller learning rate.':'The graph shows the updated prediction and loss.'}`;render();
  }
  titles.forEach((title,i)=>{const b=document.createElement('button');b.textContent=`${i+1}. ${title}`;b.onclick=()=>{stop();setPhase(i);};$('steps').append(b);});
  $('previous').onclick=()=>{stop();retreat();};$('next').onclick=()=>{stop();advance();};
  function play(forwardOnly=false){
    stop();
    if(forwardOnly||phase===5)setPhase(0);
    $('play').textContent='Ⅱ Pause';$('play').setAttribute('aria-pressed','true');
    timer=setInterval(()=>{advance();if(phase===5||(forwardOnly&&forwardStage===3))stop();},2000);
  }
  $('play').onclick=()=>{if(timer)stop();else play();};$('replay-forward').onclick=()=>play(true);
  $('preset').onchange=e=>reset(e.target.value);$('reset').onclick=()=>{$('preset').value='standard';reset('standard');};
  $('activation').onchange=e=>{state.activation=e.target.value;restartHistory();render();};
  ['x1','x2'].forEach((id,i)=>$(id).oninput=e=>{state.x[i]=Number(e.target.value);restartHistory();render();});
  $('target').onchange=e=>{if(!e.target.reportValidity()||e.target.value==='')return;state.y=e.target.valueAsNumber;restartHistory();render();};
  $('rate').onchange=e=>{state.rate=Number(e.target.value);render();};
  $('train-one').onclick=()=>train(1);$('train-many').onclick=()=>train(20);
  $('undo').onclick=()=>{if(!undo.length)return;stop();const snapshot=undo.pop();state=snapshot.state;history=snapshot.history;phase=1;lossStage=0;syncControls();$('train-message').textContent='Restored the network and loss history before the last update action.';render();};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  reset('standard');
})();
