/* Pure numerical core. Browser and Node; sparse, known-model dynamic programming. */
(function(root){
'use strict';
const DIR=[[-1,0],[0,1],[1,0],[0,-1]], ARROWS=['↑','→','↓','←'];
class RNG { constructor(seed=42){this.x=seed>>>0;} next(){this.x=(1664525*this.x+1013904223)>>>0; return this.x/4294967296;} }
class World {
 constructor(w=4,h=4,{rescue=false,slip=0,gamma=1,map="depot"}={}){this.map=map;this.w=w;this.h=h;this.n=w*h;this.gamma=gamma;this.slip=slip;this.rescue=rescue;this.cells=new Uint8Array(this.n);this.start=rescue?(h-2)*w+1:5;this.goals=new Set(rescue?[w+ w-2]:[0,this.n-1]);this.build();}
 build(){if(this.rescue){for(let r=0;r<this.h;r++)for(let c=0;c<this.w;c++){let s=r*this.w+c; const x=c/(this.w-1), y=r/(this.h-1);
 if(this.map==='terraces'){
   // Ascend three terraces, crossing the lower ridge on the right and the upper ridge on the left.
   const lowerRidge=r===Math.floor(this.h*.66);
   const upperRidge=r===Math.floor(this.h*.33);
   if((lowerRidge && (x<.70 || x>.88)) || (upperRidge && (x<.12 || x>.30)))this.cells[s]=1;
   else if(y>.38 && y<.57 && x>.48 && x<.59)this.cells[s]=2;
 }else if(this.map==='river'){
   // A hazardous river has two safe crossings of different widths.
   if(x>.42&&x<.62&&Math.abs(y-.24)>.035&&Math.abs(y-.79)>.10)this.cells[s]=2;
 }else if(this.map==='switchbacks'){
   const first=c===Math.floor(this.w*.33),second=c===Math.floor(this.w*.66);
   if((first&&y>.24)||(second&&y<.72))this.cells[s]=1;
   else if((x>.34&&x<.49&&y<.2)||(x>.67&&x<.81&&y>.8))this.cells[s]=2;
 }else if(this.map==='courtyard'){
   if(x>.33&&x<.67&&y>.27&&y<.72)this.cells[s]=1;
   else if(x>.20&&x<.80&&y>.12&&y<.27)this.cells[s]=2;
 }else {if(c===Math.floor(this.w*.45)&&r>1&&r<this.h-2)this.cells[s]=1; if(c>this.w*.52&&c<this.w*.78&&r>=1&&r<this.h*.73)this.cells[s]=2;} } } this.compile();}
 terminal(s){return this.goals.has(s);}
 active(s){return !this.terminal(s)&&this.cells[s]!==1;}
 move(s,a){let r=Math.floor(s/this.w)+DIR[a][0],c=s%this.w+DIR[a][1];if(r<0||r>=this.h||c<0||c>=this.w||this.cells[r*this.w+c]===1)return s;return r*this.w+c;}
 transitions(s,a){if(!this.active(s))return [{s,p:1,r:0}];return [[a,1-this.slip],[(a+1)%4,this.slip/2],[(a+3)%4,this.slip/2]].filter(x=>x[1]>0).map(([d,p])=>{let t=this.move(s,d);return {s:t,p,r:this.rescue?(this.terminal(t)?100:this.cells[t]===2?-16:-1):-1};});}
 compile(){this.model=Array.from({length:this.n},(_,s)=>Array.from({length:4},(_,a)=>this.transitions(s,a)));this.activeCount=0;for(let s=0;s<this.n;s++)if(this.active(s))this.activeCount++;}
 q(s,a,v){return this.model[s][a].reduce((x,t)=>x+t.p*(t.r+this.gamma*v[t.s]),0);}
 qs(s,v){return [0,1,2,3].map(a=>this.q(s,a,v));}
 sample(s,a,rng){let u=rng.next(),sum=0,ts=this.model[s][a];for(const t of ts){sum+=t.p;if(u<sum)return t;}return ts.at(-1);}
}
function greedy(q){let m=Math.max(...q);return q.map(x=>Math.abs(x-m)<1e-8?1:0).reduce((mask,x,a)=>mask|(x<<a),0);}
function actions(mask){return [0,1,2,3].filter(a=>mask&(1<<a));}
function target(world,s,v,policy){if(!world.active(s))return 0;let q=world.qs(s,v);if(!policy)return Math.max(...q);let as=actions(policy[s]);return as.reduce((x,a)=>x+q[a],0)/as.length;}
function residual(world,v,policy){let d=0;for(let s=0;s<world.n;s++)if(world.active(s))d=Math.max(d,Math.abs(target(world,s,v,policy)-v[s]));return d;}
class Planner {
 constructor(world,kind='vi'){this.world=world;this.kind=kind;this.v=new Float64Array(world.n);this.old=this.v.slice();this.policy=new Uint8Array(world.n).fill(15);this.sweeps=0;this.backups=0;this.round=0;this.phase=kind==='pi'?'evaluate':kind==='eval'?'evaluate':'maximize';this.delta=0;this.done=false;this.changed=0;this.history=[];this.last=[];}
 step(mode='sync'){
 if(this.done)return;
 const w=this.world;this.old=this.v.slice();this.last=[];
 if(this.kind==='pi'&&this.phase==='improve'){
 this.changed=0;for(let s=0;s<w.n;s++)if(w.active(s)){let p=greedy(w.qs(s,this.v));if(p!==this.policy[s]){this.changed++;this.last.push(s);}this.policy[s]=p;}
 this.round++;this.done=this.changed===0;this.phase=this.done?'stable':'evaluate';return;
 }
 const pol=this.kind==='eval'||this.kind==='pi'?this.policy:null;
 if(mode==='priority'){
 // Residual-priority asynchronous batch: rank current residuals, then update top 1/8.
 let ranked=[];for(let s=0;s<w.n;s++)if(w.active(s))ranked.push([Math.abs(target(w,s,this.v,pol)-this.v[s]),s]);ranked.sort((a,b)=>b[0]-a[0]);
 for(const [d,s] of ranked.slice(0,Math.max(1,Math.ceil(w.activeCount/8)))){if(d<1e-10)continue;this.v[s]=target(w,s,this.v,pol);this.last.push(s);this.backups++;}
 }else{for(let s=0;s<w.n;s++)if(w.active(s)){this.v[s]=target(w,s,mode==='sync'?this.old:this.v,pol);this.last.push(s);this.backups++;}}
 this.sweeps++;this.delta=residual(w,this.v,pol);this.history.push(this.delta);
 if(!pol)for(let s=0;s<w.n;s++)if(w.active(s))this.policy[s]=greedy(w.qs(s,this.v));
 if(this.delta<1e-7){if(this.kind==='pi')this.phase='improve';else {this.done=true;this.phase='converged';}}
 }
}
const api={World,Planner,RNG,DIR,ARROWS,greedy,actions,target,residual};
if(typeof module!=='undefined')module.exports=api;root.DP=api;
})(typeof globalThis!=='undefined'?globalThis:this);
