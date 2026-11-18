/* Fit a complete teaching composition without clipping controls or adding scrollbars. */
(function(){
'use strict';
function geometry(width,height){const scale=Math.min(1,width/1280,height/820);return {scale,width:width/scale,height:height/scale};}
if(typeof module!=='undefined')module.exports={geometry};
if(typeof document==='undefined')return;
const shell=document.getElementById('presentation-shell');
function fitDrawers(){for(const details of document.querySelectorAll('.explanation-drawer[open]')){const content=details.querySelector('.drawer-grid');content.style.transform='none';const available=details.clientHeight-details.querySelector('summary').offsetHeight-32;const scale=Math.min(1,available/Math.max(1,content.scrollHeight));content.style.transform=`scale(${scale})`;}}
function fit(){const g=geometry(window.innerWidth,window.innerHeight);shell.style.width=g.width+'px';shell.style.height=g.height+'px';shell.style.transform=`scale(${g.scale})`;fitDrawers();}
window.addEventListener('resize',fit);
for(const details of document.querySelectorAll('.explanation-drawer'))details.addEventListener('toggle',()=>{fitDrawers();window.dispatchEvent(new Event('resize'));});
new ResizeObserver(fitDrawers).observe(shell);
fit();requestAnimationFrame(()=>window.dispatchEvent(new Event('resize')));
})();
