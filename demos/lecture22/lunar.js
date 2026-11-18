/* Planar rigid body in vacuum. Positive attitude tips the nose to the right. */
(function () {
  const dt = 1 / 60, gravity = 1.62, pad = 2;
  const feet = [[-.8,-.7],[.8,-.7]];
  const hull = [[-.52,0],[-.36,.55],[.36,.55],[.52,0],[.42,-.3],[-.42,-.3]];
  const contacts = [...feet, ...hull];
  const start = () => ({x:-8, h:16, vx:2, vy:-1, angle:0, spin:0, fuel:16, result:'', misses:[]});
  function terrain(x) {
    const distance = Math.max(0, Math.abs(x) - pad);
    return Math.min(1, distance / 2) * (1.2 + .9*Math.sin(x*.48) + .55*Math.cos(x*1.18) + .2*Math.sin(x*2.8));
  }
  function point(s, [x,y]) {
    const c = Math.cos(s.angle), t = Math.sin(s.angle);
    return [s.x + c*x + t*y, s.h - t*x + c*y];
  }
  function step(s, keys) {
    if (s.result) return;
    const main = Number(!!keys.main), side = Number(!!keys.left) - Number(!!keys.right);
    const burn = main + .3*(Number(!!keys.left) + Number(!!keys.right));
    const power = burn ? Math.min(1, s.fuel/(dt*burn)) : 0;
    s.fuel = Math.max(0, s.fuel - power*dt*burn);
    // Side jets act above the center of mass. The main engine acts through it.
    const c = Math.cos(s.angle), t = Math.sin(s.angle);
    s.vx += (4*main*t + side*.9*c)*power*dt;
    s.vy += ((4*main*c - side*.9*t)*power - gravity)*dt;
    s.spin += side*2.4*power*dt;
    s.angle = Math.atan2(Math.sin(s.angle+s.spin*dt), Math.cos(s.angle+s.spin*dt));
    s.x += s.vx*dt;
    s.h += s.vy*dt;
    const clearance = Math.min(...contacts.map(p => {
      const [x,y] = point(s,p);
      return y-terrain(x);
    }));
    if (clearance > 0) return;
    s.h -= clearance;
    if (feet.some(p => Math.abs(point(s,p)[0]) > pad)) s.misses.push('outside the pad');
    if (Math.abs(s.vx) > .6) s.misses.push('too much sideways drift');
    if (Math.abs(s.vy) > 1) s.misses.push('vertical speed too high');
    if (Math.abs(s.angle) > Math.PI/15) s.misses.push('too tilted');
    if (Math.abs(s.spin) > Math.PI/12) s.misses.push('rotating too fast');
    s.result = s.misses.length ? 'Missed landing' : 'Soft landing';
  }
  function pilot(s) {
    const limit = (v,bound) => Math.max(-bound, Math.min(bound,v));
    const desiredVx = limit(-.5*s.x, 2);
    const desiredAngle = limit(.6*(desiredVx-s.vx), .46);
    const steering = 3*(desiredAngle-s.angle) - 2*s.spin;
    const descent = -Math.min(2, Math.max(.4, .6*(s.h-.7)));
    return {left:steering>.04, right:steering<-.04, main:s.vy<descent};
  }
  const api = {start, step, pilot, terrain, feet, hull, pad};
  if (typeof module !== 'undefined') module.exports = api;
  else window.Lunar = api;
})();
