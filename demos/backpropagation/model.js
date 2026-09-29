/* A 2 → 2 → 2 → 1 network. W[l][j][i] connects source i to destination j.
   No libraries or automatic differentiation: the backward pass is explicit. */
(function (root) {
  'use strict';
  const clone = x => JSON.parse(JSON.stringify(x));
  const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
  function activation(x, kind) { return kind === 'tanh' ? Math.tanh(x) : kind === 'relu' ? Math.max(0, x) : sigmoid(x); }
  function slope(x, a, kind) { return kind === 'tanh' ? 1 - a * a : kind === 'relu' ? +(x > 0) : a * (1 - a); }
  function preset(name = 'standard') {
    const s = { x: [0.8, 0.4], y: 1, activation: 'sigmoid', rate: 0.5,
      W: [[[0.6, -0.4], [0.3, 0.8]], [[0.7, -0.5], [-0.3, 0.9]], [[0.8, -0.6]]],
      b: [[0.1, -0.1], [0.05, 0.1], [0.05]] };
    if (name === 'saturation') { s.W[0] = [[6, 6], [-6, -6]]; s.b[0] = [3, -3]; }
    if (name === 'relu') { s.activation = 'relu'; s.b[0] = [-1, 0.1]; }
    return s;
  }
  function evaluate(s) {
    const a = [s.x.slice()], u = [], d = [], gW = [], gb = [], da = [];
    for (let l = 0; l < 3; l++) {
      u[l] = s.W[l].map((row, j) => row.reduce((v, w, i) => v + w * a[l][i], s.b[l][j]));
      a[l + 1] = u[l].map(v => activation(v, l === 2 ? 'sigmoid' : s.activation));
    }
    const prediction = a[3][0], loss = (prediction - s.y) ** 2;
    da[3] = [2 * (prediction - s.y)];
    for (let l = 2; l >= 0; l--) {
      d[l] = u[l].map((v, j) => da[l + 1][j] * slope(v, a[l + 1][j], l === 2 ? 'sigmoid' : s.activation));
      gW[l] = s.W[l].map((row, j) => row.map((w, i) => d[l][j] * a[l][i]));
      gb[l] = d[l].slice();
      da[l] = a[l].map((v, i) => s.W[l].reduce((sum, row, j) => sum + row[i] * d[l][j], 0));
    }
    return { a, u, d, da, gW, gb, prediction, loss };
  }
  function gradient(r, p) { return p.type === 'b' ? r.gb[p.l][p.j] : r.gW[p.l][p.j][p.i]; }
  function get(s, p) { return p.type === 'b' ? s.b[p.l][p.j] : s.W[p.l][p.j][p.i]; }
  function set(s, p, value) { if (p.type === 'b') s.b[p.l][p.j] = value; else s.W[p.l][p.j][p.i] = value; }
  function parameters(s) {
    return s.W.flatMap((rows, l) => rows.flatMap((row, j) => [...row.map((_, i) => ({ type: 'w', l, j, i })), { type: 'b', l, j }]));
  }
  function numerical(s, p, epsilon = 1e-5) {
    const plus = clone(s), minus = clone(s), v = get(s, p);
    set(plus, p, v + epsilon); set(minus, p, v - epsilon);
    const rp = evaluate(plus), rm = evaluate(minus), r = evaluate(s);
    const kink = s.activation === 'relu' && r.u.slice(0, 2).some((row, l) => row.some((z, j) => z === 0 || (rp.u[l][j] > 0) !== (rm.u[l][j] > 0)));
    return { value: (rp.loss - rm.loss) / (2 * epsilon), plus: rp.loss, minus: rm.loss, kink };
  }
  function update(s) {
    const next = clone(s), r = evaluate(s);
    parameters(s).forEach(p => set(next, p, get(s, p) - s.rate * gradient(r, p)));
    return next;
  }
  const api = { clone, sigmoid, activation, slope, preset, evaluate, gradient, get, set, parameters, numerical, update };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BackpropModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
