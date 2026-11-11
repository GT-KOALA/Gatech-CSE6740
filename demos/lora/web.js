'use strict';
const D = window.LORA_DATA, $ = id => document.getElementById(id);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const x0 = [1, -.5, .75, 0, -1, .25, .5, -.75];
let section = 0, frame = 0, playing = false, touring = false, timer = null, activeView = 'walk';
let recordedFrame = 0, recordedPlaying = false, recordedTimer = null, referenceShown = false;
let renderedSection = -1;
const titles = ['One input. Two paths. One adapted output.', 'Why use a low-rank bottleneck?', 'Backpropagation through both routes.', 'Where are the Transformer parameters?', 'Training loss, memory, and compute.'];
const notes = [
 ['Keep W. Learn a detour.', 'A token’s features x travel along two paths. W produces the original features. A and B produce a correction. Add the two results to get the adapted output.', 'z = A x\nΔy = s B z\ny = W x + Δy\ns = α/r', 'A and B are sequential within the learned branch. That entire branch runs in parallel with W. A full fine-tune would update W itself.', 'Every session starts automatically. Pause or step to inspect an operation.'],
 ['The bottleneck is the parameter budget.', 'Instead of learning an independent change for every entry of W, learn two thin matrices. Rank r is the number of intermediate features connecting A to B.', 'Dense update: d_out × d_in\nA: r × d_in\nB: d_out × r\nLoRA: r(d_in + d_out)', 'Change rank to see the budget grow. This controls the maximum rank of the update, not the rank of the original W. Higher rank gives more flexibility at greater training cost.', 'At the actual query width 1,536 and rank 8: 2,359,296 dense weights versus 24,576 factor weights—96× fewer.'],
 ['Frozen weights still transmit gradients.', 'First watch only the forward pass. Then the output gradient splits at the addition: one part goes through W and the other through B then A. Add both contributions at x.', 's = α/r\ng = ∂L/∂y\ng_x,frozen = Wᵀg\ng_x,LoRA = sAᵀBᵀg\n∂L/∂x = g_x,frozen + g_x,LoRA', 'W has no parameter-gradient buffer or optimizer update. Its multiplication is still needed in backward to send gradients to earlier trainable layers. A and B also receive parameter gradients.', 'Freezing W removes its parameter update; it does not remove the backward route through W.'],
 ['Count parameters before choosing targets.', 'Each row below is one projection in one Qwen decoder block. Attention has query, key, value, and output projections; the MLP has gate, up, and down projections.', 'Actual run: Q + V × 28 blocks\nRank = 8\nTrainable factors = 1,089,536', 'Counts come from the pinned backbone tensor header. Query/key/value biases and normalization weights are listed too. Embeddings and the tied output head share one parameter tensor.', 'Large MLPs dominate block parameters. That explains training cost, but parameter count alone does not identify the best LoRA targets for task quality.'],
 ['Separate memory savings from compute savings.', 'The curve is recorded validation loss. The resource comparison is an analytical model: FP32 parameters and gradients, Adam’s two FP32 moment buffers, and a linear-layer FLOP estimate.', 'Full state ≈ 16P bytes\nLoRA state ≈ 4P + 16L bytes\nP: backbone parameters\nL: adapter parameters', 'State memory excludes activations, temporary tensors, and framework overhead. Compute includes decoder linear layers only; it excludes attention products, the output head, loss, optimizer arithmetic, and checkpoint recomputation.', 'LoRA greatly reduces gradient and optimizer-state storage. Compute falls less because frozen projections still run in forward and backward.']
];
function fmt(v) { return Math.abs(v)<.0001&&v!==0?v.toExponential(1):v.toFixed(4); }
function color(v,max) { const a=.18+.62*Math.min(1,Math.abs(v)/(max||1));return v<0?`rgba(190,107,151,${a})`:`rgba(74,156,194,${a})`; }
function projection() {return D.projections.find(p=>p.layer===Number($('layer').value)&&p.projection===$('projection').value);}
function calculate() {const p=projection(),x=x0.map(v=>v*Number($('gain').value)),z=p.A.map(r=>r.reduce((s,v,i)=>s+v*x[i],0)),delta=p.B.map(r=>2*r.reduce((s,v,i)=>s+v*z[i],0));return {p,x,z,delta};}
// Standard LoRA notation: A compresses and B expands, matching the exported PEFT fields.
const toyA=[[1,0,1],[0,2,-1]],toyB=[[1,0],[0,1],[1,-1]],toyW=[[1,0,0],[0,1,0],[0,0,1]];
const matvec=(m,x)=>m.map(r=>r.reduce((s,v,i)=>s+v*x[i],0));
const addvec=(a,b)=>a.map((v,i)=>v+b[i]);
function toyMath() {const x=[2,1,-1].map(v=>v*Number($('gain').value)),z=matvec(toyA,x),delta=matvec(toyB,z).map(v=>2*v),y=addvec(x,delta);return {x,z,delta,y};}
function toyTraining() {
 const t=toyMath(),target=[5,5,-3],g=t.y.map((v,i)=>v-target[i]),btg=[0,1].map(j=>toyB.reduce((s,r,i)=>s+r[j]*g[i],0));
 const gxFrozen=g.slice(),gxLora=[0,1,2].map(i=>2*toyA.reduce((sum,row,j)=>sum+row[i]*btg[j],0)),gxTotal=addvec(gxFrozen,gxLora);
 const dB=g.map(v=>t.z.map(z=>2*v*z)),dA=btg.map(v=>t.x.map(x=>2*v*x)),eta=.01;
 const nextA=toyA.map((r,i)=>r.map((v,j)=>v-eta*dA[i][j])),nextB=toyB.map((r,i)=>r.map((v,j)=>v-eta*dB[i][j]));
 const nextY=addvec(t.x,matvec(nextB,matvec(nextA,t.x)).map(v=>2*v));
 const loss=y=>.5*y.reduce((s,v,i)=>s+(v-target[i])**2,0);
 return {...t,target,g,gxFrozen,gxLora,gxTotal,dA,dB,nextA,nextB,nextY,before:loss(t.y),after:loss(nextY)};
}
function short(v) {return Number.isInteger(v)?String(v):v.toFixed(2);}
function vtext(v) {return `[${v.map(short).join(', ')}]`;}
function tiles(values,ready=true,cls='') {return `<div class="toy-vector ${cls}">${values.map(v=>`<span>${ready?short(v):'·'}</span>`).join('')}</div>`;}
function toyMatrix(m,active=-1,columns=false) {return `<div class="toy-matrix" style="--cols:${m[0].length}">${m.map((r,i)=>r.map((v,j)=>`<span class="${(columns?j:i)===active?'selected':''}">${short(v)}</span>`).join('')).join('')}</div>`;}
const defs='<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#79c7ff"/></marker><marker id="pink-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#de9caf"/></marker></defs>';
function box(x,y,w,h,label,small,on=false) {return `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" class="${on?'on':''}"/><text x="${x+w/2}" y="${y+h/2-2}">${label}</text><text class="small" x="${x+w/2}" y="${y+h/2+17}">${small}</text></g>`;}
function wire(d,on=false,gradient=false) {
 const path=`<path class="wire ${on?'energized':''} ${gradient?'gradient':''}" style="marker-end:url(#${gradient?'pink-arrow':'arrow'})" d="${d}"/>`;
 const dot=on&&(playing||touring)&&!reducedMotion?`<circle r="5" fill="${gradient?'#de9caf':'#e9bd75'}"><animateMotion dur=".7s" repeatCount="indefinite" path="${d}"/></circle>`:'';
 return path+dot;
}
function svg(content,label) {return `<svg viewBox="0 0 780 260" role="img" aria-label="${label}">${defs}${content}</svg>`;}
function banner(steps) {return `<div class="banner" role="status">${steps[frame]}</div>`;}
function render() {
 const p=projection(),t=toyMath(),n=notes[section],last=titles.length-1;
 $('section').value=String(section);$('chapter').textContent=`INSIDE LoRA / ${String(section+1).padStart(2,'0')}`;$('title').textContent=titles[section];
 ['insight','explanation','equation','detail','takeaway'].forEach((id,i)=>$(id).textContent=n[i]);
 $('shape').textContent=section===1?`Real projection budget · ${p.projection}`:section===3?'Pinned Qwen · 1.54B unique parameters':section===last?'Recorded loss · estimated resources':'Teaching example · 3 features · rank 2 · s = α/r (example: 2)';
 $('projection-label').hidden=section!==1;$('gain-label').hidden=[1,3,last].includes(section);$('rank-label').hidden=section!==1;$('cost-token-label').hidden=section!==last;
 $('gain-value').textContent=`${$('gain').value}×`;$('layer').disabled=true;$('projection').disabled=![1,3].includes(section);$('gain').disabled=[1,3,last].includes(section);
 $('progress').value=frame;$('progress-label').textContent=frame===8?'Complete':`${frame+1} of 9`;$('back').disabled=section===0;$('next').textContent=section===last?'Explore recorded demos →':'Next section →';
 $('step-count').textContent=`${section+1} / ${titles.length}`;$('operation').disabled=frame===8;$('play').textContent=(playing||touring)?'Pause':frame===8?'Replay session':'Resume';$('tour').textContent=touring?'Stop auto tour':'Auto tour';document.body.classList.toggle('computing',playing||touring);
 let content='',caption='';
 if(section===0){
  const steps=['Start with one token’s feature vector x. W is already pretrained.','The same x is sent to the frozen and learned branches.','W computes the original projection Wx. Its weights stay fixed.','A reads x and writes a short vector z.','B receives z and expands it to the original output width.','Multiply the correction by s = α/r.','Add the correction to Wx, coordinate by coordinate.','The adapted output continues into the Transformer.','Full tuning changes W. LoRA changes only the small detour A and B.'];
  content=svg(box(15,92,100,75,'x',vtext(t.x),frame>=1)+wire('M115 130 H165 V55 H220',frame===1||frame===2)+wire('M165 130 V202 H220',frame===1||frame===3)+box(225,18,315,74,'W · x','Frozen pretrained projection',frame>=2)+box(225,165,125,74,'A','3 → 2 features',frame>=3)+wire('M350 202 H400',frame===4)+box(405,165,125,74,'B','2 → 3 features',frame>=4)+wire('M530 202 H590 V130 H635',frame===5||frame===6)+wire('M540 55 H590 V130',frame===6)+box(640,92,125,75,'+ → y',frame>=6?vtext(t.y):'Add both paths',frame>=6)+'<text class="large" x="580" y="240">× s</text>','Follow x through frozen W and the A then B branch')+banner(steps)+`<div class="forward-values"><div><small>Frozen output Wx</small>${tiles(t.x,frame>=2)}</div><b>+</b><div><small>Learned correction sBAx</small>${tiles(t.delta,frame>=5)}</div><b>=</b><div><small>Adapted output y</small>${tiles(t.y,frame>=6)}</div></div>`;
  caption='One illustrative feature vector, not token probabilities. Scale s = α/r; numerical example uses s = 2. The worked toy uses W = identity; the real adapter has 1,536 input features and rank 8.';
 }else if(section===1){content=rankScene(p);caption='Exact parameter counts for the selected real projection. The rank selector explores a hypothetical budget; it does not change trained weights.';
 }else if(section===2){content=backpropScene();caption='Forward only first; backward only second. Scale s = α/r; numerical example uses s = 2 and squared loss. Real training used completion-token cross-entropy.';
 }else if(section===3){content=placementScene();caption='Counts audited from the pinned Qwen tensor header. Softmax, attention products, residual addition, and RoPE have no learned parameters here.';
 }else{content=trainingScene();caption='Recorded SST-2 loss; resource figures are estimates, not measured GPU memory or runtime. Select tokens per forward/backward to change the compute estimate.';}
 const entering=renderedSection!==section;renderedSection=section;
 $('stage').innerHTML=`<div class="scene ${entering?'enter':''}"><div class="operation-track" aria-hidden="true">${Array.from({length:9},(_,i)=>`<span class="${i<frame?'done':i===frame?'current':''}"></span>`).join('')}</div>${content}</div>`;
 $('progress').setAttribute('aria-valuetext',`Operation ${frame+1} of 9`);$('caption').textContent=caption;scheduleFit();
 $('walk-provenance').innerHTML=section===3?'Sources: <a href="https://arxiv.org/html/2106.09685#S7.SS1" target="_blank" rel="noreferrer">LoRA paper §7.1</a> · <a href="https://huggingface.co/docs/peft/en/package_reference/lora" target="_blank" rel="noreferrer">PEFT targets</a>. Target choice is task-dependent.':section===last?'Loss: recorded SST-2 history. Resource figures: derived estimates. <a href="https://huggingface.co/docs/transformers/v4.37.0/en/model_memory_anatomy" target="_blank" rel="noreferrer">Memory and training-operation assumptions</a>.':'Explicit teaching example: W = I, 3 input/output features, rank 2. The real Qwen adapter uses rank 8; its parameter counts are labeled separately.';
 if(section===last)$('stage').querySelectorAll('.chart-dot').forEach(dot=>{const show=()=>$('chart-note').textContent=`Step ${dot.dataset.step}: validation loss ${Number(dot.dataset.loss).toFixed(5)}`;dot.onmouseenter=dot.onfocus=show;});
}

function compact(n) {return n>=1e9?(n/1e9).toFixed(2)+'B':n>=1e6?(n/1e6).toFixed(2)+'M':n.toLocaleString();}
function rankScene(p) {
 const rank=Number($('rank').value)||8,din=p.A_shape[1],dout=p.B_shape[0],full=din*dout,a=rank*din,b=rank*dout,total=a+b;
 const steps=['Compare two ways to compute an update for the same column vector x. W stays frozen in LoRA.','DENSE · Start at the input column vector x on the right.','DENSE · Apply ΔW to x to produce the output correction. ΔW has '+full.toLocaleString()+' trainable values.','LoRA · Start at the same input column vector x on the right.','LoRA · Apply A to x. Ax has shape '+rank+' × 1: the rank bottleneck.','LoRA · Apply B to Ax to return to '+dout+' output features.','Train A and B: '+a.toLocaleString()+' + '+b.toLocaleString()+' = '+total.toLocaleString()+' weights.','Only '+(100*total/full).toFixed(2)+'% as many trainable values; the same x enters both comparisons.','Both routes produce an output-sized correction. LoRA scales BAx by s = α/r before adding it to Wx.'];
 const grid=(rows,cols,on)=>`<div class="budget-grid ${on?'budget-on':''}" style="--cols:${cols}">${Array.from({length:rows*cols},()=>'<i></i>').join('')}</div>`;
 const operand=(label,shape,kind,rows,cols,on)=>`<div class="rank-operand ${kind} ${on?'operand-active':''}"><b>${label}</b><div class="operand-picture">${grid(rows,cols,on)}</div><small>${shape}</small></div>`;
 const input=on=>operand('x',`${din} × 1`,'rank-input',12,1,on);
 return `<div class="rank-comparison"><div class="strip"><div class="strip-title">Option 1 · independent dense update</div><div class="rank-product" aria-label="Dense update matrix times input column vector">${operand('ΔW',`${dout} × ${din}`,'rank-dense',12,12,frame===2)}<span class="multiply">·</span>${input(frame===1)}</div><div class="rank-equation">ΔW x → ${dout} × 1 correction</div><strong class="param-number">${full.toLocaleString()}</strong><small>Trainable weights · input x is not counted</small></div><div class="strip"><div class="strip-title">Option 2 · train two thin factors</div><div class="rank-product" aria-label="B times A times input column vector; apply A first, then B">${operand('B',`${dout} × ${rank}`,'rank-b',12,2,frame===5)}<span class="multiply">·</span>${operand('A',`${rank} × ${din}`,'rank-a',2,12,frame===4)}<span class="multiply">·</span>${input(frame===3)}</div><div class="rank-equation">B (A x) → ${dout} × 1 correction</div><div class="rank-order">Apply right to left: first Ax (${rank} × 1), then B(Ax)</div><strong class="param-number">${total.toLocaleString()}</strong><small>${(full/total).toFixed(1)}× fewer trainable weights · scale correction by s = α/r</small></div></div><div class="subtle">Dimensions are rows × columns. Tiles are schematic; labels give exact sizes. Both options use the same input x.</div>`+banner(steps);
}

function backpropScene() {
 const g=toyTraining(),backward=frame>=4;
 const steps=['FORWARD · read x = '+vtext(g.x)+'. Backward arrows are hidden.','FORWARD · x enters frozen W and trainable A.','FORWARD · A → B → scale, then add Wx to get y = '+vtext(g.y)+'.','FORWARD · compare with target '+vtext(g.target)+'; loss = '+short(g.before)+'.','BACKWARD · g = y − target = '+vtext(g.g)+' arrives at the addition and splits.','BACKWARD · frozen route computes Wᵀg; learned route computes sBᵀg.','BACKWARD · Aᵀ maps the learned gradient back to x; the frozen route also reaches x.','ADD AT x · '+vtext(g.gxFrozen)+' + '+vtext(g.gxLora)+' = '+vtext(g.gxTotal)+'. Earlier layers receive this sum.','UPDATE · change only A and B. With η = 0.01, loss '+short(g.before)+' → '+short(g.after)+'. W stays fixed.'];
 let routes='';
 if(!backward)routes=wire('M110 125 H160 V45 H200',frame===1)+wire('M160 125 V205 H200',frame===1)+wire('M320 205 H375',frame===2)+wire('M495 205 H520 V125 H540',frame===2)+wire('M460 45 H520 V125',frame===2)+wire('M640 125 H670',frame===3);
 else routes=wire('M675 112 H645',frame===4,true)+wire('M545 106 H505 V45 H465',frame===5,true)+wire('M545 145 H520 V205 H500',frame===5,true)+wire('M380 205 H325',frame===6,true)+wire('M205 45 H140 V106 H110',frame===6||frame===7,true)+wire('M205 205 H140 V145 H110',frame===6||frame===7,true);
 return `<div class="phase-label ${backward?'backward':''}">${backward?'② BACKWARD: split the error, then add both input gradients':'① FORWARD: compute features and loss'}</div>`+svg(box(10,93,100,65,'x',frame>=7?'Receives sum':vtext(g.x),frame===1||frame>=7)+box(205,12,255,65,'W frozen',backward?'Compute Wᵀg · no ∇W update':'Original projection',frame===5)+box(205,173,115,65,'A',frame>=6?'Receives ∇A':'Trainable',frame>=6)+box(380,173,115,65,'B',frame>=5?'Receives ∇B':'Trainable',frame>=5)+box(545,93,95,65,'+ → y',vtext(g.y),frame>=2)+box(675,93,95,65,'Loss',frame>=3?short(g.before):'Target',frame>=3)+routes,'Separate forward and backward phases through frozen and learned routes')+`<div class="input-gradients"><div class="strip"><div class="strip-title">Through frozen W · Wᵀg</div>${tiles(g.gxFrozen,frame>=5)}<small>Backward computation remains; W is not updated.</small></div><b>+</b><div class="strip"><div class="strip-title">Through B and A · sAᵀBᵀg</div>${tiles(g.gxLora,frame>=6)}<small>Also compute parameter gradients for A and B.</small></div><b>=</b><div class="strip"><div class="strip-title">Total ∂L/∂x</div>${tiles(g.gxTotal,frame>=7)}<small>Pass this sum to earlier layers.</small></div></div>`+banner(steps);
}
function placementScene() {
 const i=D.inventory,steps=['One block contains '+i.per_block.toLocaleString()+' parameters. The decoder repeats it '+i.blocks+' times.','Q: '+i.groups[0].parameters.toLocaleString()+' parameters; the real adapter adds '+i.groups[0].rank8_parameters.toLocaleString()+'.','K has fewer outputs because Qwen uses grouped-query attention.','V: '+i.groups[2].parameters.toLocaleString()+' parameters; the real adapter adds '+i.groups[2].rank8_parameters.toLocaleString()+'.','O mixes attention heads; it has the same weight count as Q.','Each gate/up/down MLP projection has 13,762,560 weights—much larger than an attention projection.','Q + V across all 28 blocks adds 1,089,536 adapter weights.','Embeddings and the output head share '+i.embedding_parameters.toLocaleString()+' weights, counted once.','Large parameter counts explain cost pressure. They do not prove which target set improves task quality most.'];
 const diagram=svg(box(10,90,95,65,'Token x','Features',false)+wire('M105 122 H145 V38 H195',frame===1)+wire('M145 122 H195',frame===2)+wire('M145 122 V210 H195',frame===3)+box(200,7,120,60,'Q · LoRA',compact(i.groups[0].parameters),frame===1||frame===6)+box(200,92,120,60,'K',compact(i.groups[1].parameters),frame===2)+box(200,177,120,60,'V · LoRA',compact(i.groups[2].parameters),frame===3||frame===6)+wire('M320 38 H350 V120 H375',frame===1)+wire('M320 122 H375',frame===2)+wire('M320 210 H350 V140 H375',frame===3)+box(380,90,120,70,'Attention','0 extra parameters',frame===3)+wire('M500 125 H530',frame===4)+box(535,90,100,70,'O',compact(i.groups[3].parameters),frame===4)+wire('M635 125 H665',frame===5)+box(670,90,100,70,'MLP',compact(i.groups.slice(4).reduce((s,g)=>s+g.parameters,0)),frame===5),'Transformer component sizes from the actual backbone');
 return diagram+`<div class="target-legend"><span class="target-core">● Q / V: common compact choice · used here</span><span class="target-wide">● K / O / MLP: also targeted in all-linear LoRA</span></div><table class="parameter-table"><thead><tr><th>One decoder block</th><th>Weight shape</th><th>Weight parameters</th><th>Bias</th><th>Rank-8 A+B</th></tr></thead><tbody>${i.groups.map((g,j)=>`<tr class="${g.adapted?'target-core-row':'target-wide-row'} ${((frame<=4&&frame===j+1)||(frame===5&&j>=4)||(frame===6&&g.adapted))?'current-component':''}"><td>${g.module} <span class="target-tag ${g.adapted?'target-core':'target-wide'}">${g.adapted?'Q/V · used here':'all-linear'}</span></td><td>${g.shape.join(' × ')}</td><td>${g.weights.toLocaleString()}</td><td>${g.bias.toLocaleString()}</td><td>${g.rank8_parameters.toLocaleString()}</td></tr>`).join('')}<tr><td>Two RMSNorms</td><td>2 × 1,536</td><td>${i.norms_per_block.toLocaleString()}</td><td>0</td><td>—</td></tr></tbody></table><div class="inventory-foot">${i.blocks} blocks + shared embedding/head (${compact(i.embedding_parameters)}) + final norm (${i.final_norm.toLocaleString()}) = <b>${i.parameters.toLocaleString()} unique parameters</b>. <a href="https://huggingface.co/docs/peft/en/package_reference/lora" target="_blank" rel="noopener">Targeting guide ↗</a></div>`+banner(steps);
}
function trainingBudget(tokens) {
 const p=D.inventory.parameters,l=D.parameters,d=D.inventory.decoder_linear_weights;
 return {tokens,full:{weights:4*p,gradients:4*p,adam:8*p,state:16*p,flops:6*tokens*d},lora:{weights:4*(p+l),gradients:4*l,adam:8*l,state:4*p+16*l,flops:4*tokens*d+6*tokens*l}};
}
function bytes(n) {return n>=2**30?(n/2**30).toFixed(2)+' GiB':(n/2**20).toFixed(2)+' MiB';}
function trainingScene() {
 const cost=trainingBudget(Number($('cost-tokens').value)||1024);
 return `<div class="training-budget-layout"><div class="loss-panel">${trainingCurve()}</div><div class="resource-panel"><div class="strip-title">FP32 Adam state · analytical estimate</div><table class="resource-table"><thead><tr><th>Stored per iteration</th><th>Full tuning</th><th>LoRA Q+V</th></tr></thead><tbody>${[['Weights','weights'],['Parameter gradients','gradients'],['Adam moments','adam'],['Total state','state']].map(([label,key])=>`<tr><td>${label}</td><td>${bytes(cost.full[key])}</td><td>${bytes(cost.lora[key])}</td></tr>`).join('')}</tbody></table><div class="memory-bars"><div><span>Full</span><i style="width:100%"></i></div><div><span>LoRA</span><i style="width:${100*cost.lora.state/cost.full.state}%"></i></div></div><div class="strip-title">Decoder linear FLOPs / forward + backward</div><div class="compute-costs"><span>Full: <b>${(cost.full.flops/1e12).toFixed(2)} TFLOPs</b></span><span>LoRA: <b>${(cost.lora.flops/1e12).toFixed(2)} TFLOPs</b></span></div><p class="subtle">${cost.tokens.toLocaleString()} token positions · ~${(100*(1-cost.lora.flops/cost.full.flops)).toFixed(1)}% fewer linear-layer FLOPs.<br>Full ≈ 6TD; LoRA ≈ 4TD + 6TL. Frozen W still needs its forward and input-gradient products.</p><p class="resource-warning">Memory excludes activations/workspaces. FLOPs exclude attention products, output head, loss, optimizer and recomputation. These are not GPU measurements.</p></div></div>`;
}
function trainingCurve() {
 const rows=D.training.log_history.filter(r=>r.eval_loss!==undefined), visible=Math.ceil(frame/8*rows.length);
 const x=s=>65+(s-50)/450*635, y=l=>235-(l-.06)/.07*195;
 return `<div class="metric-grid"><div class="metric"><strong>500</strong><small>Optimizer steps</small></div><div class="metric"><strong>8,000</strong><small>Training examples</small></div><div class="metric"><strong>0.07359</strong><small>Final validation loss</small></div></div><svg class="chart" viewBox="0 0 760 285" role="img" aria-label="Validation loss across 500 SST-2 optimizer steps">${[.06,.08,.10,.12].map(v=>`<line class="grid" x1="65" x2="700" y1="${y(v)}" y2="${y(v)}"/><text x="23" y="${y(v)+4}">${v.toFixed(2)}</text>`).join('')}<path d="${rows.slice(0,visible).map((r,i)=>`${i?'L':'M'} ${x(r.step)} ${y(r.eval_loss)}`).join(' ')}" fill="none" stroke="#91dcc1" stroke-width="2.5"/>${rows.slice(0,visible).map(r=>`<circle class="chart-dot" tabindex="0" cx="${x(r.step)}" cy="${y(r.eval_loss)}" r="5" data-step="${r.step}" data-loss="${r.eval_loss}"><title>Step ${r.step}: ${r.eval_loss.toFixed(5)}</title></circle>`).join('')}${rows.map(r=>`<text x="${x(r.step)}" y="258">${r.step}</text>`).join('')}<text x="380" y="280">Optimizer step</text></svg><div class="banner" id="chart-note">${frame?'Lowest recorded loss: 0.06940 at step 350.':'Play to reveal recorded validation points.'}</div>`;
}
function stop() { clearTimeout(timer); playing=false; touring=false; document.body.classList.remove('computing'); }
function schedule() { clearTimeout(timer); if(!playing&&!touring)return; timer=setTimeout(()=>{if(frame<8)frame++;else if(touring&&section<titles.length-1){section++;frame=0;}else{stop();render();return;}render();schedule();},frame===8?2400:Number($('tempo').value)); }
function go(next) { stop(); section=Math.max(0,Math.min(titles.length-1,next));frame=0;playing=true;schedule();render(); }
function togglePlay() { if(playing||touring)stop();else{if(frame===8)frame=0;playing=true;schedule();}render(); }
function stopRecorded() { clearTimeout(recordedTimer);recordedPlaying=false; }
function view(name) {
 stop();stopRecorded();activeView=name;
 for(const id of ['walk','recorded']){$(`${id}-view`).hidden=id!==name;$(`${id}-tab`).classList.toggle('active',id===name);$(`${id}-tab`).setAttribute('aria-pressed',String(id===name));}
 document.querySelector('footer').hidden=name!=='walk';$('section').hidden=name!=='walk';$('section').disabled=name!=='walk';
 if(name==='walk'){playing=true;if(frame===8)frame=0;schedule();render();}if(name==='recorded')renderRecorded();
 scheduleFit();history.replaceState(null,'',name==='walk'?'#walk':name==='recorded'?'#results':'#compare');
}
// Accumulate every revealed word in place; the prefix never drops out of the DOM.
function words(text) { return text.match(/\S+\s*/g)||[]; }
function writeWords(id,text,reveal) {
 const all=words(text),end=Math.min(all.length,reveal),target=$(id);
 const follow=target.scrollHeight-target.scrollTop-target.clientHeight<48;
 if((target.__source!==text&&!text.startsWith((target.__source||'').trimEnd()))||end<(target.__count||0)||!target.__count){
  target.replaceChildren();target.__count=0;target.__source=text;
 }
 target.querySelectorAll('.word-now').forEach(span=>span.classList.remove('word-now'));
 if(!end){const span=document.createElement('span');span.className='waiting';span.textContent='Press Play to reveal this saved output.';target.replaceChildren(span);return;}
 if(!target.__count)target.replaceChildren();
 for(let i=0;i<Math.min(target.__count,end);i++)target.children[i].textContent=all[i];
 for(let i=target.__count;i<end;i++){const span=document.createElement('span');span.textContent=all[i];if(i===end-1)span.className='word-now';target.append(span);}
 target.__count=end;target.__source=text;
 target.classList.toggle('long-output',all.length>90);
 if(follow)target.scrollTop=target.scrollHeight;
}
function experimentLabel(run) { return run.label|| (run.source.includes('-style-')?'Pirate · style run':'Pirate · first run'); }
const experiments = [...(D.recorded_runs||[]),...(D.saved_demos||[])];
function currentExperiment() { return experiments[Number($('recorded-run').value)]; }
function currentRecordedRow() { return currentExperiment().result.examples[Number($('recorded-example').value)]; }
function maxWords(row) { return Math.max(words(row.base).length,words(row.lora).length); }
function renderRecorded() {
 const run=currentExperiment();if(!run)return;const row=currentRecordedRow(),result=run.result,max=maxWords(row);
 $('recorded-question').textContent=row.text;$('recorded-label').textContent=`LoRA / ${result.metadata.task||'SST-2'} adapter`;
 for(const name of ['base','lora']){writeWords(`recorded-${name}`,row[name],recordedFrame);$(`recorded-${name}-count`).textContent=`${Math.min(recordedFrame,words(row[name]).length)} / ${words(row[name]).length} words`;$(`recorded-${name}-node`).classList.toggle('active',recordedPlaying);}
 $('recorded-step').disabled=recordedFrame>=max;
 $('recorded-progress').setAttribute('aria-valuetext',`${recordedFrame} of ${max} words`);
 $('recorded-progress').max=max;$('recorded-progress').value=recordedFrame;$('recorded-play').textContent=recordedPlaying?'Pause replay':recordedFrame>=max?'Replay saved outputs':'Play saved outputs';
 $('recorded-page').textContent='Full context stays visible · no text is discarded';
 $('recorded-source').textContent=`${run.source} · Saved output reveal, not live decoding · ${result.max_new_tokens||result.metadata.max_new_tokens||'recorded'} new-token cap`;
 $('recorded-source').title=`SHA-256: ${run.sha256}`;
 const metrics=result.metrics,style=result.style_metrics,isStyle=run.source.includes('-style-');
 $('recorded-insight').textContent=metrics?(isStyle?'The voice changed. Check the facts.':'Better reference prediction. Little style.'):'A story prompt, a sentiment adapter.';
 $('recorded-observation').textContent=metrics?(row.text==='Why does a prism separate colors?'&&isStyle?'The LoRA answer invokes diffraction and incorrect path lengths. The reference describes wavelength-dependent refraction. The base also misstates the presence of wavelengths.':isStyle?'Pirate markers occur in all 12 saved answers. Reference NLL is worse than Base. Style and factual quality need separate checks.':'Reference NLL improves by 21.5%, but none of the 12 saved answers contain pirate markers.'):'These are genuine generations with the available SST-2 adapter. Raw story continuation is outside its training task. Compare where the two continuations agree or diverge.';
 $('recorded-metrics').replaceChildren();
 if(metrics){for(const [label,value]of [['Base NLL',metrics.base.nll.toFixed(3)],['LoRA NLL',metrics.lora.nll.toFixed(3)],['Base / LoRA perplexity',`${metrics.base.perplexity.toFixed(2)} / ${metrics.lora.perplexity.toFixed(2)}`],['Pirate markers · base / LoRA',`${100*style.base.marker_rate}% / ${100*style.lora.marker_rate}%`]]){const div=document.createElement('div');div.className='score-row';const span=document.createElement('span');span.textContent=label;const b=document.createElement('strong');b.textContent=value;div.append(span,b);$('recorded-metrics').append(div);}}
 $('recorded-note').textContent=metrics?'Loss: 48 validation examples / 2,700 response tokens. Style: 12 saved generations. Marker rate is a phrasing heuristic, not factual accuracy.':'The adapter was not trained on stories. Changing decoding settings does not make it a story adapter.';
 $('recorded-reference-toggle').disabled=!row.reference;$('recorded-reference-toggle').textContent=referenceShown?'Hide reference':'Show reference';
 $('recorded-reference').hidden=!referenceShown||!row.reference;$('recorded-reference').textContent=row.reference||'';
 // While a reference is shown, give it the sidebar space used by metrics and notes.
 $('recorded-metrics').hidden=referenceShown;$('recorded-note').hidden=referenceShown;$('recorded-observation').hidden=referenceShown;
 scheduleFit();$('recorded-provenance').textContent=metrics?'Pinned Qwen revision; topic-disjoint curriculum. Both models get the same raw question. This is saved evaluation replay; use Live generation for the installed pirate-style adapter.':`Adapter: ${result.metadata.task||'sst2'} · greedy decoding · same backbone and exact prefix for both outputs.`;
}
function selectRecordedRun() {
 stopRecorded();const old=$('recorded-question').textContent, rows=currentExperiment().result.examples;
 $('recorded-example').replaceChildren(...rows.map((r,i)=>new Option(r.text.slice(0,65),String(i))));
 let i=rows.findIndex(r=>r.text===old);if(i<0)i=Math.max(0,rows.findIndex(r=>r.text==='What is caching?'));
 $('recorded-example').value=String(i);resetRecorded();
}
function resetRecorded() { stopRecorded();recordedFrame=0;referenceShown=false;renderRecorded(); }
function recordedTick() { if(!recordedPlaying)return;const max=maxWords(currentRecordedRow());if(recordedFrame>=max){stopRecorded();renderRecorded();return;}recordedFrame++;renderRecorded();recordedTimer=setTimeout(recordedTick,Number($('recorded-tempo').value)); }
function setupRecorded() {
 $('recorded-run').replaceChildren(...experiments.map((r,i)=>new Option(experimentLabel(r),String(i))));
 $('recorded-run').value=String(Math.max(0,experiments.findIndex(r=>r.source.includes('-style-'))));
 $('recorded-run').onchange=selectRecordedRun;$('recorded-example').onchange=resetRecorded;
 $('recorded-next').onclick=()=>{$('recorded-example').value=String((Number($('recorded-example').value)+1)%currentExperiment().result.examples.length);resetRecorded();};
 $('recorded-play').onclick=()=>{if(recordedPlaying)stopRecorded();else{if(recordedFrame>=maxWords(currentRecordedRow())){recordedFrame=0;}recordedPlaying=true;recordedTick();}renderRecorded();};
 $('recorded-step').onclick=()=>{stopRecorded();recordedFrame=Math.min(maxWords(currentRecordedRow()),recordedFrame+1);renderRecorded();};
 $('recorded-replay').onclick=resetRecorded;
 $('recorded-progress').oninput=()=>{stopRecorded();recordedFrame=Number($('recorded-progress').value);renderRecorded();};
 $('recorded-complete').onclick=()=>{stopRecorded();recordedFrame=maxWords(currentRecordedRow());renderRecorded();};
 $('recorded-reference-toggle').onclick=()=>{referenceShown=!referenceShown;const notes=$('recorded-reference').closest?.('details');if(notes&&referenceShown)notes.open=true;$('recorded-reference-toggle').setAttribute('aria-expanded',String(referenceShown));renderRecorded();};selectRecordedRun();
}
// Match the neighboring Transformer Lab: fit the scene and notes to the viewport.
let fitRequest=null;
function scheduleFit() {
 if(typeof requestAnimationFrame==='undefined')return;
 cancelAnimationFrame(fitRequest);fitRequest=requestAnimationFrame(()=>{
  if(activeView==='walk'){
   const stage=$('stage'),scene=stage.querySelector('.scene');
   if(scene&&stage.clientHeight&&stage.clientWidth){
    if(window.innerWidth<=680){scene.style.transform='none';scene.style.width='100%';}else{
    scene.style.transform='none';scene.style.width=stage.clientWidth+'px';
    const scale=Math.min(1,stage.clientHeight/Math.max(1,scene.scrollHeight),stage.clientWidth/Math.max(1,scene.scrollWidth));
    scene.style.transform=`scale(${scale})`;
    }
   }
  }

 });
}
if(typeof ResizeObserver!=='undefined')new ResizeObserver(scheduleFit).observe($('stage'));
// Wire the classroom navigation and playback controls.
for(let i=0;i<28;i++)$('layer').add(new Option(String(i),String(i)));
for(let i=0;i<titles.length;i++)$('section').add(new Option(`${String(i+1).padStart(2,'0')} · ${titles[i]}`,String(i)));
$('section').onchange=()=>go(Number($('section').value));
for(const id of ['layer','projection','gain','rank','cost-tokens'])$(id).oninput=()=>{stop();frame=0;playing=true;schedule();render();};
$('progress').oninput=()=>{stop();frame=Number($('progress').value);render();};$('tempo').onchange=schedule;
$('back').onclick=()=>go(section-1);$('next').onclick=()=>section===titles.length-1?view('recorded'):go(section+1);
$('play').onclick=togglePlay;$('operation').onclick=()=>{stop();frame=Math.min(8,frame+1);render();};
$('replay').onclick=()=>{stop();frame=0;if(!reducedMotion){playing=true;schedule();}render();};
$('tour').onclick=()=>{if(touring)stop();else{stop();if(section===titles.length-1){section=0;frame=0;}touring=true;schedule();}render();};
for(const name of ['walk','recorded'])$(`${name}-tab`).onclick=()=>view(name);
document.addEventListener('fullscreenchange',()=>{$('fullscreen').textContent=document.fullscreenElement?'Exit full screen':'Full screen';scheduleFit();});
$('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch(e){$('fullscreen').textContent='Use browser full screen';}};
document.addEventListener('keydown',e=>{if(e.target.closest('input,textarea,select,button,[contenteditable="true"]'))return;if(activeView==='walk'){if(e.key==='ArrowRight')go(section+1);if(e.key==='ArrowLeft')go(section-1);if(e.code==='Space'){e.preventDefault();togglePlay();}}else if(activeView==='recorded'&&e.code==='Space'){e.preventDefault();$('recorded-play').click();}});
document.querySelectorAll?.('.demo-notes').forEach(notes=>notes.addEventListener('toggle',scheduleFit));
window.addEventListener('resize',()=>{scheduleFit();if(activeView==='recorded')renderRecorded();});
setupRecorded();playing=true;schedule();render();if(location.hash==='#results')view('recorded');
