/* Run with: node scripts/tests/backprop-model.test.cjs */
const assert = require('node:assert/strict');
const M = require('../../demos/backpropagation/model.js');
let checks = 0;
for (const activation of ['sigmoid', 'tanh', 'relu']) {
  for (let sample = 0; sample < 12; sample++) {
    const s = M.preset(); s.activation = activation; s.x = [0.23 + sample * 0.11, -0.71 + sample * 0.13]; s.y = sample % 2;
    M.parameters(s).forEach((p, i) => M.set(s,p,Math.sin(i * 3.1 + sample * 1.7) * 1.3));
    const r = M.evaluate(s);
    for (const p of M.parameters(s)) {
      const n = M.numerical(s, p);
      if (!n.kink) { assert.ok(Math.abs(n.value - M.gradient(r,p)) < 2e-7, `${activation} sample ${sample}: ${JSON.stringify(p)}`); checks++; }
    }
    for (let i=0;i<2;i++) {
      const plus=M.clone(s),minus=M.clone(s);plus.x[i]+=1e-5;minus.x[i]-=1e-5;
      assert.ok(Math.abs((M.evaluate(plus).loss-M.evaluate(minus).loss)/2e-5-r.da[0][i])<2e-7);
    }
  }
}
const s=M.preset(), r=M.evaluate(s), next=M.update(s);
assert.ok(M.evaluate(next).loss<r.loss,'Default update lowers loss');
for(const p of M.parameters(s))assert.equal(M.get(next,p),M.get(s,p)-s.rate*M.gradient(r,p),'Simultaneous update');
assert.equal(r.da[3][0],2*(r.prediction-s.y),'Squared loss sign and factor');
assert.equal(M.parameters(s).length,15);
const zero=M.preset();zero.x[0]=0;const zr=M.evaluate(zero);assert.ok(zr.gW[0][0][0]===0);assert.ok(zr.gW[0][1][0]===0);
const dead=M.evaluate(M.preset('relu'));assert.equal(dead.a[1][0],0);assert.ok(dead.gb[0][0]===0);
const sat=M.evaluate(M.preset('saturation'));assert.ok(Math.abs(sat.gW[0][0][0])<Math.abs(r.gW[0][0][0])/100);
const kink=M.preset('relu');kink.x=[0,0];kink.b[0][0]=0;assert.ok(M.numerical(kink,{type:'b',l:0,j:0}).kink);
const branches=r.d[1][0]*s.W[1][0][0]+r.d[1][1]*s.W[1][1][0];assert.equal(r.da[1][0],branches);
let trained=M.preset();for(let i=0;i<100;i++)trained=M.update(trained);assert.ok(M.evaluate(trained).loss<r.loss/10);
assert.deepEqual(s,M.preset(),'Updates do not mutate source state');
console.log(`Passed: ${checks} finite-difference parameter checks, input sensitivities, branch accumulation, simultaneous updates, loss reduction, zero input, saturation, and ReLU cases.`);
