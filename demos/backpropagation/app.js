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
  let phase = 0, forwardStage = 0, lossStage = 0, updateStage = 0, timer = null;
  const MAX_UPDATES = 500;
  let showValues = false;
  let history = [M.evaluate(state).loss], undo = [];
  const titles = ['Forward', 'Loss', 'Backward', 'Update'];
  const mainPhases = [0, 1, 2, 5];
  const mainStep = () => phase===0 ? 0 : phase===1&&lossStage===0 ? 1 : phase<5 ? 2 : 3;
  const forwardLabels = ['Whole pass', 'Hidden 1', 'Hidden 2', 'Prediction'];
  const svgNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, text) {
    const e = document.createElementNS(svgNS, tag);
    Object.entries(attrs || {}).forEach(([k,v]) => e.setAttribute(k,v));
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function paramSymbol(p){
    return `<msubsup><mi>${p.type==='b'?'b':'W'}</mi><mn>${p.j+1}${p.type==='b'?'':','+(p.i+1)}</mn><mn>${p.l}</mn></msubsup>`;
  }
  function nodeSymbol(l,j){
    return l===0?`<msub><mi>x</mi><mn>${j+1}</mn></msub>`:l===3?'<mi>y</mi><mo>′</mo>':`<msubsup><mi>z</mi><mn>${j+1}</mn><mn>${l}</mn></msubsup>`;
  }
  function mathLabel(attrs,content,width=120){
    const gradient=attrs.class==='node-gradient';
    const f=el('foreignObject',{x:attrs.x-width/2,y:attrs.y-(gradient?16:21),width,height:gradient?42:32,'pointer-events':'none',class:'math-label'});
    const div=document.createElementNS('http://www.w3.org/1999/xhtml','div');
    div.setAttribute('class',`svg-math ${attrs.class||''}`);div.innerHTML=math(content);f.append(div);return f;
  }
  function buttonize(e, label, handler) {
    e.setAttribute('tabindex','0'); e.setAttribute('role','button'); e.setAttribute('aria-label',label);
    e.addEventListener('click',handler);
    e.addEventListener('keydown', event => { if(event.key==='Enter'||event.key===' '){event.preventDefault();handler();} });
  }
  function stop() { clearInterval(timer); timer = null; $('play').textContent = '▶ Play'; $('play').setAttribute('aria-pressed','false'); }
  function focusParameter(p) { stop(); selected = p; phase = 4 - p.l; $('weights-dialog').close(); render(); if(matchMedia('(max-width:800px)').matches)setPanel(true); }
  function position(l,j) { return {x:[70,295,520,740][l],y:l===3?182:j===0?105:267}; }
  function pathEdge(l,j,i) {
    if (phase < 2 || phase===5) return false;
    if (selected.type === 'node') return l >= selected.l && (l !== selected.l || i === selected.j);
    if (l === selected.l) return selected.type === 'w' && j === selected.j && i === selected.i;
    return l > selected.l && (l !== selected.l + 1 || i === selected.j);
  }
  function nodeOnPath(l,j) {
    if(phase===1)return lossStage===1&&l===3;
    if(phase<2||phase===5)return false;
    const start=selected.type==='node'?selected.l:selected.l+1;
    return l>=start&&(l!==start||j===selected.j);
  }
  function drawNetwork() {
    const svg = $('network'); svg.querySelectorAll(':scope > :not(desc)').forEach(n=>n.remove());
    svg.dataset.phase = phase; svg.dataset.forwardStage = forwardStage; svg.dataset.lossStage = lossStage; svg.dataset.updateStage=updateStage;
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
      const p={type:'w',l,j,i},back=pathEdge(l,j,i),forward=phase===0&&(forwardStage===0||forwardStage>=l+1);
      const computed=phase>0||forwardStage>l+1;
      const chosen=phase>=2&&phase<5&&selected.type==='w'&&selected.l===l&&selected.j===j&&selected.i===i;
      const g=el('g',{class:'edge-group','data-layer':l});
      // Exactly one visible line per connection. Reverse its geometry for backpropagation.
      const coords=back?{x1:x2,y1:y2,x2:x1,y2:y1}:{x1,y1,x2,y2};
      const line=el('line',{...coords,class:`edge ${phase===5&&updateStage===0?'update-edge':back?'backward':forward?'forward':computed?'computed':'pending'} ${chosen?'selected-edge':''}`});
      if(back||forward)line.setAttribute('marker-end',`url(#${back?'back':'forward'})`);
      g.append(line);g.append(el('line',{x1,y1,x2,y2,class:'edge-hit'}));
      const t=l===2?0.43:i===j?0.51:(i===0?0.31:0.69);
      g.append(mathLabel({x:a.x+dx*t,y:a.y+dy*t-11,class:`edge-label ${back?'active':''} ${phase===0&&forwardStage<=l?'uncomputed-label':''}`},paramSymbol(p)));
      buttonize(g,`${pname(p)}. Trace this weight’s gradient.`,()=>focusParameter(p));svg.append(g);
    }));
    for(let l=0;l<4;l++)result.a[l].forEach((a,j)=>{
      const pos=position(l,j),ready=phase>0||forwardStage===0||forwardStage>=l;
      const active=phase>=2&&phase<5&&(selected.type==='node'?selected.l===l&&selected.j===j:selected.l+1===l&&selected.j===j);
      const current=phase===0&&(forwardStage===0||forwardStage===l);
      const forwardReady=phase===0&&ready;
      const g=el('g',{class:`node ${active?'active':''} ${current?'forward-current':''} ${forwardReady?'forward-ready':''} ${phase===5&&updateStage===1&&l>0?'update-node':''} ${ready?'':'uncomputed'}`,'data-layer':l,'data-neuron':j});
      g.append(el('circle',{cx:pos.x,cy:pos.y,r:38}));
      g.append(mathLabel({x:pos.x,y:pos.y-7,class:'node-name'},nodeSymbol(l,j),76));
      g.append(el('text',{x:pos.x,y:pos.y+15,'text-anchor':'middle',class:'node-value'},ready?(showValues?fmt(a,3):(l===0?'input':l===3?'output':'f(s)')):'…'));
      if(nodeOnPath(l,j))
        g.append(mathLabel({x:pos.x,y:pos.y+60,class:'node-gradient'},`<mfrac><mrow><mo>∂</mo><mi>L</mi></mrow><mrow><mo>∂</mo>${nodeSymbol(l,j)}</mrow></mfrac>`));
      buttonize(g,`${aname(l,j)}. Trace this neuron’s gradient.`,()=>{stop();selected={type:'node',l,j};phase=l===0?4:5-l;render();if(matchMedia('(max-width:800px)').matches)setPanel(true);});svg.append(g);
    });
    const backwardLoss=(phase>1&&phase<5)||(phase===1&&lossStage===1),forwardLoss=phase===1&&lossStage===0;
    const lossLine=el('line',{...(backwardLoss?{x1:841,y1:182,x2:782,y2:182}:{x1:780,y1:182,x2:841,y2:182}),class:`edge ${backwardLoss?'backward':forwardLoss?'forward':'pending'}`});
    if(backwardLoss||forwardLoss)lossLine.setAttribute('marker-end',`url(#${backwardLoss?'back':'forward'})`);svg.append(lossLine);
    const loss=el('g',{class:`loss-node ${forwardLoss?'forward-current':''} ${backwardLoss?'active':''} ${phase===0?'uncomputed':''}`});
    loss.append(el('rect',{x:847,y:146,width:76,height:72,rx:9}));
    loss.append(el('text',{x:885,y:172,'text-anchor':'middle',class:'node-name'},'L'));
    loss.append(el('text',{x:885,y:196,'text-anchor':'middle',class:'node-value'},phase===0?'…':showValues?fmt(result.loss,3):'error'));svg.append(loss);
    if(phase>0){
      svg.append(el('line',{x1:885,y1:266,x2:885,y2:222,class:`edge ${forwardLoss?'forward':'computed'}`,...(forwardLoss?{'marker-end':'url(#forward)'}:{})}));
      svg.append(el('text',{x:885,y:287,'text-anchor':'middle',class:'column-sub'},showValues?`Target y = ${fmt(state.y)}`:'Target'));
    }
    const caption=phase===5?(updateStage===0?'Update every weight using its gradient':'Update every bias using its gradient'):phase===0?forwardStage===0?'Forward → inputs → hidden layers → prediction':`Forward → ${forwardLabels[forwardStage]}${forwardStage===3?' computed; loss comes next':''}`:forwardLoss?'Compare prediction with target → compute L':backwardLoss&&phase===1?'← Send ∂L/∂y′ from the loss to the output':'← Backward: gradients flow toward the selected neuron or weight';
    svg.append(el('text',{x:470,y:368,'text-anchor':'middle',class:`flow-label ${phase===5?'update-caption':phase===0||forwardLoss?'forward-caption':''}`},caption));
  }
  const math = content => `<math xmlns="http://www.w3.org/1998/Math/MathML">${content}</math>`;
  const mi = value => `<mi>${value}</mi>`;
  const frac = (a,b) => `<mfrac><mrow><mo>∂</mo>${mi(a)}</mrow><mrow><mo>∂</mo>${mi(b)}</mrow></mfrac>`;
  const derivative = (a,b) => math(frac(a,b));
  const box=(label,eq)=>`<div class="formula-box"><span class="label">${label}</span><div class="equation">${eq}</div></div>`;
  function concept(title, explanation){return `<div class="concept"><h3>${title}</h3><p>${explanation}</p></div>`;}
  function inspectForward(){
    $('inspect-title').textContent='Forward';
    $('inspection').innerHTML=`<div class="concept-route"><span>Inputs</span><b>↓</b><span>Weighted sum</span><b>↓</b><span>Activation</span><b>↓</b><span>Prediction</span></div>`+
      box('Combine',math('<mi>s</mi><mo>=</mo><mo>∑</mo><mi>w</mi><mi>x</mi><mo>+</mo><mi>b</mi>'))+
      box('Transform',math('<mi>a</mi><mo>=</mo><mi>f</mi><mo>(</mo><mi>s</mi><mo>)</mo>'))+
      '<p class="notation-note">x: input · w: weight · b: bias<br>s: weighted sum · a: neuron output.</p>';
  }
  function inspectLoss(){
    $('inspect-title').textContent=lossStage===0?'Compare with the target':'Loss gradient';
    $('inspection').innerHTML=lossStage===0?
      concept('Measure the mismatch','A closer prediction gives a smaller loss. A perfect match gives zero loss.')+concept('Prepare to work backward','Next, ask how changing the prediction would change the loss.'):
      box('Output gradient · arrives from the loss',derivative('L','y′'))+concept('Read the direction','Its sign tells us whether increasing the prediction would increase or decrease the loss.')+concept('Pass it backward','Multiply this gradient by the output neuron’s local derivative.');
  }
  function inspectBackward(){
    const p=selected;
    const inputNode=p.type==='node'&&p.l===0;
    $('inspect-title').textContent=inputNode?'Gradient at the input':'Backward through a neuron';
    if(inputNode){
      $('inspection').innerHTML=box('Input gradient · collected from both paths',math(frac('L','x')+'<mo>=</mo><mo>∑</mo>'+frac('L','s')+'<mo>×</mo><mi>w</mi>'))+
        concept('Add the returning contributions','Each downstream neuron sends back its weighted-sum gradient times its connecting weight.')+
        '<p class="notation-note">x: selected input · s: a downstream weighted sum · w: connecting weight.</p>';return;
    }
    const bias=p.type==='b';
    $('inspection').innerHTML=
      `<div class="gradient-card incoming"><div><strong>Output gradient</strong><small>Incoming from downstream</small></div>${derivative('L','a')}</div>`+
      box('Chain rule · through the activation',math(frac('L','s')+'<mo>=</mo>'+frac('L','a')+'<mo>×</mo>'+frac('a','s')))+
      `<p class="local-slope">${derivative('a','s')} is the local slope of the activation.</p>`+
      `<div class="gradient-card outgoing"><div><strong>Input gradient</strong><small>Contribution along this path</small></div>${math(frac('L','s')+'<mo>×</mo><mi>w</mi>')}</div>`+
      `<div class="gradient-card parameter"><div><strong>${bias?'Bias':'Weight'} gradient</strong><small>Saved for the update</small></div>${math(frac('L',bias?'b':'w')+'<mo>=</mo>'+frac('L','s')+(bias?'':'<mo>×</mo><mi>x</mi>'))}</div>`+
      '<p class="branch-note">Add contributions from all downstream neurons to get the total input gradient.</p>'+
      '<p class="notation-note">Forward: x → weighted sum s → output a.<br>w: one incoming weight · L: loss.</p>';
    if(p.type==='node'){
      $('inspection').insertAdjacentHTML('beforeend','<button id="node-bias" class="trace-bias">Trace bias</button>');
      $('node-bias').onclick=()=>focusParameter({type:'b',l:p.l-1,j:p.j});
    }
  }
  function inspectUpdate(){
    const name=updateStage===0?'w':'b';
    $('inspect-title').textContent=updateStage===0?'Update the weights':'Update the biases';
    $('inspection').innerHTML=box('Gradient descent',math(`<msub><mi>${name}</mi><mtext>new</mtext></msub><mo>=</mo><mi>${name}</mi><mo>−</mo><mi>η</mi>`+frac('L',name)))+
      '<p>η is the learning rate. Use the buttons below to update all weights and biases together.</p>';
  }
  function inspect(){
    if(phase===0)inspectForward();else if(phase===1)inspectLoss();else if(phase===5)inspectUpdate();else inspectBackward();
  }
  function drawHistory(){
    const svg=$('loss-chart'), width=Math.max(200,svg.clientWidth), height=Math.max(55,svg.clientHeight);
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.replaceChildren();
    const yMax=0.5;
    svg.dataset.xMax=MAX_UPDATES;svg.dataset.yMax=yMax;
    const left=38,right=width-12,top=8,bottom=height-29;
    for(const value of [0,0.25,yMax]){
      const y=bottom-value/yMax*(bottom-top);
      svg.append(el('line',{x1:left,y1:y,x2:right,y2:y,stroke:'#dee4e3'}));
      svg.append(el('text',{x:left-7,y:y+3,'text-anchor':'end',class:'chart-text y-tick'},String(value)));
    }
    const points=history.map((v,i)=>`${left+i/MAX_UPDATES*(right-left)},${bottom-v/yMax*(bottom-top)}`);
    const defs=el('defs'),clip=el('clipPath',{id:'loss-plot-bounds'});
    clip.append(el('rect',{x:left-3,y:top,width:right-left+6,height:bottom-top+3}));defs.append(clip);svg.append(defs);
    const curve=el('g',{'clip-path':'url(#loss-plot-bounds)'});
    curve.append(el('polyline',{points:points.join(' '),fill:'none',stroke:'#087c83','stroke-width':2.5}));
    const [cx,cy]=points[points.length-1].split(',');curve.append(el('circle',{cx,cy,r:3,fill:'#087c83'}));svg.append(curve);
    const ticks=width<450?[0,250,500]:[0,100,200,300,400,500];
    for(const n of ticks)svg.append(el('text',{x:left+n/MAX_UPDATES*(right-left),y:height-15,'text-anchor':'middle',class:'chart-text x-tick'},String(n)));
    svg.append(el('text',{x:(left+right)/2,y:height-2,'text-anchor':'middle',class:'chart-text'},'Updates'));
    $('update-count').textContent=`${history.length-1} / ${MAX_UPDATES} updates`;
  }
  function drawParameters() {
    $('parameter-tables').innerHTML=[0,1,2].map(l=>`<div><h3>${l===2?'Output layer':`Hidden layer ${l+1}`}</h3><table><thead><tr><th>Parameter</th><th>Value</th><th>∂L/∂parameter</th></tr></thead><tbody>${M.parameters(state).filter(p=>p.l===l).map(p=>`<tr><td><button data-param='${JSON.stringify(p)}'>${math(paramSymbol(p))}</button></td><td><input aria-label="${pname(p)} value" data-edit='${JSON.stringify(p)}' type="number" min="-20" max="20" step="any" value="${M.get(state,p)}"></td><td>${fmt(M.gradient(result,p),6)}</td></tr>`).join('')}</tbody></table></div>`).join('');
    document.querySelectorAll('[data-param]').forEach(e=>e.onclick=()=>focusParameter(JSON.parse(e.dataset.param)));
    document.querySelectorAll('[data-edit]').forEach(e=>e.onchange=()=>{if(!e.reportValidity()||e.value==='')return;M.set(state,JSON.parse(e.dataset.edit),e.valueAsNumber);restartHistory();render();});
  }
  function drawProgress(){
    const container=$('pass-progress');container.replaceChildren();
    const step=mainStep();
    const labels=step===0?forwardLabels:step===1?['Measure error']:step===2?['Loss gradient','Output','Hidden 2','Hidden 1']:['Weights','Biases'];
    const current=step===0?forwardStage:step===1?0:step===2?(phase===1?0:phase-1):updateStage;
    labels.forEach((label,i)=>{
      const b=document.createElement('button');b.textContent=label;
      if(i===current)b.setAttribute('aria-current','step');
      if(i<current)b.classList.add('completed');
      b.onclick=()=>{
        stop();
        if(step===0)forwardStage=i;
        else if(step===1)lossStage=0;
        else if(step===2){
          phase=i+1;
          if(i===0)lossStage=1;else selected={type:'w',l:4-phase,j:0,i:0};
        }else updateStage=i;
        render();$('pass-progress').children[i].focus({preventScroll:true});
      };
      container.append(b);
    });
  }
  function render(){
    result=M.evaluate(state);
    $('metrics').hidden=false;$('metrics').dataset.visible=String(showValues);$('metrics').setAttribute('aria-hidden',String(!showValues));
    $('show-values').checked=showValues;$('x1-value').textContent=fmt(state.x[0],2);$('x2-value').textContent=fmt(state.x[1],2);
    $('prediction').textContent=phase===0&&forwardStage>0&&forwardStage<3?'…':fmt(result.prediction,5);
    $('target-display').textContent=fmt(state.y);$('loss').textContent=phase===0?'…':fmt(result.loss,6);
    [...$('steps').children].forEach((e,i)=>{if(i===mainStep())e.setAttribute('aria-current','step');else e.removeAttribute('aria-current');});
    $('previous').disabled=phase===0&&forwardStage===0;$('next').disabled=phase===5&&updateStage===1;
    $('next').textContent=phase===0?['Inspect first layer →','Compute hidden 2 →','Compute prediction →','Compute loss →'][forwardStage]:
      phase===1?(lossStage===0?'Start backward ←':'Through output ←'):phase===4?'Review update →':phase===5?(updateStage===0?'Review biases →':'Complete'):'Next layer ←';
    const remaining=MAX_UPDATES-history.length+1;
    $('train-one').disabled=remaining===0;$('train-many').disabled=remaining===0;
    const batch=Math.max(1,Math.min(20,remaining));$('train-many').textContent=`Apply ${batch} update${batch===1?'':'s'}`;
    $('loss-current').textContent=`Loss ${fmt(result.loss,5)}${remaining===0?' · reset to start again':''}`;
    $('undo').disabled=!undo.length;$('replay-forward').disabled=phase!==0;
    $('graph-mode').textContent=phase===0?'BLUE → COMPUTE VALUES':phase===1&&lossStage===0?'BLUE → COMPUTE LOSS':phase===5?'TEAL · UPDATE PARAMETERS':'ORANGE ← TRACE GRADIENTS';
    drawProgress();drawNetwork();inspect();drawHistory();drawParameters();
  }
  function setPhase(p){
    phase=p;
    if(p===0)forwardStage=0;
    if(p===1)lossStage=0;
    if(p>=2&&p<=4)selected={type:'w',l:4-p,j:0,i:0};
    if(p===5)updateStage=0;
    render();
  }
  function advance(){
    if(phase===0&&forwardStage<3){forwardStage++;render();return;}
    if(phase===1&&lossStage===0){lossStage=1;render();return;}
    if(phase===5){updateStage=1;render();return;}
    setPhase(phase+1);
  }
  function retreat(){
    if(phase===0){forwardStage=Math.max(0,forwardStage-1);render();return;}
    if(phase===1&&lossStage===1){lossStage=0;render();return;}
    if(phase===5&&updateStage===1){updateStage=0;render();return;}
    const previous=phase-1;setPhase(previous);
    if(previous===0){forwardStage=3;render();}
    if(previous===1){lossStage=1;render();}
  }
  function restartHistory(){stop();history=[M.evaluate(state).loss];undo=[];$('train-message').textContent='Values changed. A new loss history starts from this network.';}
  function syncControls(){ $('activation').value=state.activation;$('x1').value=state.x[0];$('x2').value=state.x[1];$('target').value=state.y;$('rate').value=String(state.rate); }
  function reset(name){stop();state=M.preset(name);phase=0;forwardStage=0;lossStage=0;updateStage=0;selected={type:'w',l:0,j:0,i:0};history=[M.evaluate(state).loss];undo=[];syncControls();$('train-message').textContent='No updates yet. Backpropagation computes gradients; gradient descent changes the parameters.';render();}
  function train(n){
    stop();n=Math.min(n,MAX_UPDATES-history.length+1);if(n<=0)return;const before=M.evaluate(state).loss;undo.push({state:M.clone(state),history:history.slice()});if(undo.length>50)undo.shift();
    for(let i=0;i<n;i++){
      const next=M.update(state);
      if(!Number.isFinite(M.evaluate(next).loss)){$('train-message').textContent='Update stopped: non-finite value. Reduce the learning rate.';render();return;}
      state=next;history.push(M.evaluate(state).loss);
    }
    phase=5;forwardStage=3;
    const after=history[history.length-1];$('train-message').textContent=`Applied ${n} simultaneous update${n===1?'':'s'}. Loss: ${fmt(before,6)} → ${fmt(after,6)}. ${after>before?'Loss increased; try a smaller learning rate.':'The graph shows the updated prediction and loss.'}`;render();
  }
  titles.forEach((title,i)=>{const b=document.createElement('button');b.textContent=`${i+1}. ${title}`;b.onclick=()=>{stop();if(i===2){phase=1;lossStage=1;render();}else setPhase(mainPhases[i]);};$('steps').append(b);});
  $('previous').onclick=()=>{stop();retreat();};$('next').onclick=()=>{stop();advance();};
  function play(forwardOnly=false){
    stop();
    if(forwardOnly||(phase===5&&updateStage===1))setPhase(0);
    $('play').textContent='Ⅱ Pause';$('play').setAttribute('aria-pressed','true');
    timer=setInterval(()=>{advance();if((phase===5&&updateStage===1)||(forwardOnly&&forwardStage===3))stop();},2000);
  }
  $('play').onclick=()=>{if(timer)stop();else play();};$('replay-forward').onclick=()=>play(true);
  $('show-values').onchange=e=>{showValues=e.target.checked;render();};
  $('preset').onchange=e=>reset(e.target.value);$('reset').onclick=()=>{$('preset').value='standard';reset('standard');};
  $('activation').onchange=e=>{state.activation=e.target.value;restartHistory();render();};
  ['x1','x2'].forEach((id,i)=>$(id).oninput=e=>{state.x[i]=Number(e.target.value);restartHistory();render();});
  $('target').onchange=e=>{if(!e.target.reportValidity()||e.target.value==='')return;state.y=e.target.valueAsNumber;restartHistory();render();};
  $('rate').onchange=e=>{state.rate=Number(e.target.value);render();};
  $('train-one').onclick=()=>train(1);$('train-many').onclick=()=>train(20);
  $('undo').onclick=()=>{if(!undo.length)return;stop();const snapshot=undo.pop();state=snapshot.state;history=snapshot.history;phase=5;syncControls();$('train-message').textContent='Restored the network and loss history before the last update action.';render();};
  function setPanel(open){
    document.querySelector('.workspace').classList.toggle('show-explanation',open);
    $('panel-toggle').setAttribute('aria-expanded',String(open));
  }
  $('panel-toggle').onclick=()=>setPanel(true);$('close-panel').onclick=()=>setPanel(false);
  $('edit-weights').onclick=()=>{stop();$('weights-dialog').showModal();};
  $('close-weights').onclick=()=>$('weights-dialog').close();
  $('fullscreen').onclick=async()=>{
    try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}
    catch(e){$('train-message').textContent='Full screen is unavailable in this browser.';}
  };
  document.addEventListener('fullscreenchange',()=>{$('fullscreen').textContent=document.fullscreenElement?'Exit full screen':'Full screen';});
  new ResizeObserver(()=>drawHistory()).observe(document.querySelector('.loss-history'));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  reset('standard');
})();
