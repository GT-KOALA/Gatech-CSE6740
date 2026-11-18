/* Discretized continuous-state, discounted optimal control with bilinear interpolation. */
(function(root){'use strict';
const TAU=2*Math.PI;
function wrap(x){return ((x+Math.PI)%TAU+TAU)%TAU-Math.PI;}
class Pendulum {
 constructor(na=121,nv=81){this.na=na;this.nv=nv;this.n=na*nv;this.dt=.05;this.gamma=.99;this.torques=[-2,0,2];this.v=new Float64Array(this.n);this.next=new Float64Array(this.n);this.iterations=0;this.residual=Infinity;this.history=[];this.done=false;this.build();}
 dynamics(theta,omega,u){let w=Math.max(-8,Math.min(8,omega+(15*Math.sin(theta)+3*u)*this.dt));return {theta:wrap(theta+w*this.dt),omega:w};}
 cost(theta,omega,u){return theta*theta+.1*omega*omega+.001*u*u;}
 weights(theta,omega){let x=(wrap(theta)+Math.PI)/TAU*this.na,y=(Math.max(-8,Math.min(8,omega))+8)/16*(this.nv-1),i=Math.floor(x)%this.na,j=Math.min(this.nv-2,Math.floor(y)),a=x-Math.floor(x),b=y-j;return {ids:[j*this.na+i,j*this.na+(i+1)%this.na,(j+1)*this.na+i,(j+1)*this.na+(i+1)%this.na],ws:[(1-a)*(1-b),a*(1-b),(1-a)*b,a*b]};}
 interpolate(theta,omega){const {ids,ws}=this.weights(theta,omega);return ids.reduce((z,s,k)=>z+ws[k]*this.v[s],0);}
 build(){this.ids=new Int32Array(this.n*12);this.ws=new Float64Array(this.n*12);this.rewards=new Float64Array(this.n*3);for(let j=0;j<this.nv;j++)for(let i=0;i<this.na;i++){let s=j*this.na+i,theta=-Math.PI+TAU*i/this.na,omega=-8+16*j/(this.nv-1);for(let a=0;a<3;a++){let u=this.torques[a],t=this.dynamics(theta,omega,u),b=this.weights(t.theta,t.omega),base=(s*3+a)*4;this.rewards[s*3+a]=-this.cost(theta,omega,u);for(let k=0;k<4;k++){this.ids[base+k]=b.ids[k];this.ws[base+k]=b.ws[k];}}}}
 sweep(){if(this.done)return;let d=0;for(let s=0;s<this.n;s++){let best=-Infinity;for(let a=0;a<3;a++){let base=(s*3+a)*4,sum=0;for(let k=0;k<4;k++)sum+=this.ws[base+k]*this.v[this.ids[base+k]];best=Math.max(best,this.rewards[s*3+a]+this.gamma*sum);}this.next[s]=best;d=Math.max(d,Math.abs(best-this.v[s]));}let old=this.v;this.v=this.next;this.next=old;this.iterations++;this.residual=d;this.history.push(d);this.done=d<1e-5;}
 action(theta,omega){let best=-Infinity,u=0;for(const torque of this.torques){let t=this.dynamics(theta,omega,torque),q=-this.cost(theta,omega,torque)+this.gamma*this.interpolate(t.theta,t.omega);if(q>best){best=q;u=torque;}}return u;}
}
const api={Pendulum,wrap};if(typeof module!=='undefined')module.exports=api;root.Swing=api;
})(typeof globalThis!=='undefined'?globalThis:this);
