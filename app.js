// Simple neural network visualizer + trainer
// Keeps everything in vanilla JS. Works for small networks.

// ---------- Utilities ----------
const $ = id => document.getElementById(id);
function randn() { // basic gaussian
  let u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}
function randInt(min, max){ return Math.floor(Math.random() * (max - min + 1)) + min; }
function clamp(n,min,max){return Math.max(min,Math.min(max,n))}

// ---------- Activation fns ----------
const Activations = {
  None: {
    f: x => x,
    df: x => 1
  },
  Sigmoid: {
    f: x => 1/(1+Math.exp(-x)),
    df: x => {
      const s = 1/(1+Math.exp(-x)); return s*(1-s);
    }
  },
  ReLU: {
    f: x => x>0?x:0,
    df: x => x>0?1:0
  }
}

// ---------- Neural Network ----------
class SimpleNN {
  constructor(sizes, activationName='Sigmoid'){
    this.sizes = sizes.slice();
    this.L = sizes.length;
    this.activationName = activationName;
    this.activation = Activations[activationName] || Activations.Sigmoid;
    this.initRandom();
  }
  initRandom(){
    this.weights = [];
    this.biases = [];
    for(let l=0;l<this.L-1;l++){
      const rows = this.sizes[l+1];
      const cols = this.sizes[l];
      // initialize weights and biases as integers between -10 and 10, then scale to smaller floats
      const scale = 0.1;
      const w = Array.from({length:rows},()=>Array.from({length:cols},()=>randInt(-10,10) * scale));
      const b = Array.from({length:rows},()=>randInt(-10,10) * scale);
      this.weights.push(w);
      this.biases.push(b);
    }
  }
  forward(input){
    const a = [input.slice()];
    const z = [];
    for(let l=0;l<this.L-1;l++){
      const W = this.weights[l];
      const b = this.biases[l];
      const prev = a[a.length-1];
      const zcur = W.map((row,i)=>{
        let s = b[i]||0;
        for(let j=0;j<row.length;j++) s += row[j]*prev[j];
        return s;
      });
      const acur = zcur.map(v => this.activation.f(v));
      z.push(zcur); a.push(acur);
    }
    return {a,z};
  }
  computeLoss(output, target){
    // MSE
    let sum=0; for(let i=0;i<output.length;i++){const d=output[i]-target[i];sum+=d*d} return sum/output.length;
  }
  backprop(x, y){
    const {a,z} = this.forward(x);
    const nabla_w = this.weights.map(W=>W.map(row=>row.map(()=>0)));
    const nabla_b = this.biases.map(b=>b.map(()=>0));
    // delta for output layer
    const Lidx = this.L-2;
    const out = a[a.length-1];
    const delta = out.map((o,i)=> (o - y[i]) * this.activation.df(z[Lidx][i]));
    nabla_b[Lidx]=delta.slice();
    for(let i=0;i<delta.length;i++){
      for(let j=0;j<a[a.length-2].length;j++){
        nabla_w[Lidx][i][j] = delta[i]*a[a.length-2][j];
      }
    }
    // propagate back
    let nextDelta = delta;
    for(let l=this.L-3;l>=0;l--){
      const Wnext = this.weights[l+1];
      const zcur = z[l];
      const curDelta = Array.from({length:this.sizes[l+1]},()=>0);
      for(let i=0;i<this.sizes[l+1];i++){
        let s=0;
        for(let k=0;k<nextDelta.length;k++) s += Wnext[k][i]*nextDelta[k];
        curDelta[i] = s * this.activation.df(zcur[i]);
      }
      nabla_b[l]=curDelta.slice();
      for(let i=0;i<curDelta.length;i++){
        for(let j=0;j<this.sizes[l];j++){
          nabla_w[l][i][j] = curDelta[i]*a[l][j];
        }
      }
      nextDelta = curDelta;
    }
    return {nabla_w,nabla_b,loss:this.computeLoss(out,y)};
  }
  applyGradients(nw, nb, lr=0.1){
    for(let l=0;l<this.weights.length;l++){
      for(let i=0;i<this.weights[l].length;i++){
        for(let j=0;j<this.weights[l][i].length;j++){
          this.weights[l][i][j] -= lr * nw[l][i][j];
        }
      }
      for(let i=0;i<this.biases[l].length;i++) this.biases[l][i] -= lr * nb[l][i];
    }
  }
}

// ---------- App state & UI ----------
const state = {
  net: null,
  running: false,
  lossHistory: [],
  steps: 0,
  selectedEdge: null,
};

function parseSizes(){
  const inp = parseInt($('input-size').value||'2');
  const out = parseInt($('output-size').value||'1');
  const hiddenTxt = $('hidden-sizes').value.trim();
  const hidden = hiddenTxt?hiddenTxt.split(',').map(s=>parseInt(s.trim())).filter(n=>!isNaN(n) && n>0):[];
  return [inp,...hidden,out];
}

function buildNetwork(){
  const sizes = parseSizes();
  const act = $('activation').value;
  state.net = new SimpleNN(sizes, act);
  state.lossHistory = [];
  state.steps = 0;
  drawNetwork();
  drawLoss();
  updateInfo();
}

function initRandom(){ buildNetwork(); }

// ---------- Training loop ----------
let rafHandle = null;
function trainStep(iterations=1){
  if(!state.net) return;
  const lr = parseFloat($('learning-rate').value) || 0.1;
  const inputs = ($('inputs').value||'').split(',').map(s=>parseFloat(s.trim())).filter(n=>!isNaN(n));
  // enforce x0 as constant 1 for bias term if available
  if(state.net && state.net.sizes[0] >= 1){
    if(inputs.length === 0) inputs = [1];
    else inputs[0] = 1;
  }
  const targets = ($('targets').value||'').split(',').map(s=>parseFloat(s.trim())).filter(n=>!isNaN(n));
  if(inputs.length !== state.net.sizes[0] || targets.length !== state.net.sizes[state.net.sizes.length-1]){
    console.warn('Input/target size mismatch');
  }
  for(let it=0;it<iterations;it++){
    const {nabla_w,nabla_b,loss} = state.net.backprop(inputs, targets);
    state.net.applyGradients(nabla_w,nabla_b,lr);
    state.lossHistory.push(loss);
    state.steps++;
  }
  drawNetwork(); drawLoss(); updateInfo();
}

function startTraining(){
  if(state.running) return;
  state.running = true;
  const per = parseInt($('iters').value)||1;
  function loop(){
    if(!state.running) return;
    trainStep(per);
    const delay = parseInt($('viz-delay').value) || 0;
    if(delay > 0){
      // use timeout to slow down visuals
      rafHandle = { type: 'timeout', id: setTimeout(loop, delay) };
    } else {
      rafHandle = { type: 'raf', id: requestAnimationFrame(loop) };
    }
  }
  loop();
}
function pauseTraining(){
  state.running = false;
  if(rafHandle){
    if(rafHandle.type === 'raf') cancelAnimationFrame(rafHandle.id);
    else if(rafHandle.type === 'timeout') clearTimeout(rafHandle.id);
    rafHandle = null;
  }
}

// ---------- Visualization ----------
const svg = $('network-viz');
function clearSVG(){ while(svg.firstChild) svg.removeChild(svg.firstChild); }

function drawNetwork(){
  clearSVG(); if(!state.net) return;
  const sizes = state.net.sizes;
  const W = svg.clientWidth, H = svg.clientHeight;
  const layerCount = sizes.length;
  const layerX = i => 60 + i*( (W-120) / Math.max(1,layerCount-1) );
  const nodePositions = [];
  for(let l=0;l<layerCount;l++){
    const n = sizes[l];
    nodePositions[l] = [];
    const y0 = 30; const y1 = H-30;
    for(let i=0;i<n;i++){
      const x = layerX(l);
      const y = y0 + (i+0.5)*( (y1-y0)/n );
      nodePositions[l].push({x,y});
    }
  }
  // draw edges
  for(let l=0;l<layerCount-1;l++){
    const Wl = state.net.weights[l];
    for(let i=0;i<Wl.length;i++){
      for(let j=0;j<Wl[i].length;j++){
        const p1 = nodePositions[l][j];
        const p2 = nodePositions[l+1][i];
        const weight = Wl[i][j];
        const line = document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('x1',p1.x); line.setAttribute('y1',p1.y);
        line.setAttribute('x2',p2.x); line.setAttribute('y2',p2.y);
        const mag = Math.abs(weight);
        const width = clamp( Math.log1p(mag+0.001)*3 + 0.5, 0.4, 8);
        line.setAttribute('stroke-width', width);
        const color = weight>=0 ? '#60a5fa' : '#fb7185';
        line.setAttribute('stroke', color);
        line.classList.add('edge');
        line.dataset.layer = l; line.dataset.i = i; line.dataset.j = j;
        line.addEventListener('click', onEdgeClick);
        svg.appendChild(line);
        // optional small label
        const mx = (p1.x+p2.x)/2, my=(p1.y+p2.y)/2;
        const t = document.createElementNS('http://www.w3.org/2000/svg','text');
        t.setAttribute('x',mx+6); t.setAttribute('y',my-2); t.setAttribute('class','label');
        t.textContent = weight.toFixed(2);
        svg.appendChild(t);
      }
    }
  }
  // draw nodes
  // compute activations (if inputs available)
  let inputVals = ($('inputs').value||'').split(',').map(s=>parseFloat(s.trim())).filter(n=>!isNaN(n));
  // enforce x0 as constant 1 for visualization/bias
  if(sizes[0] >= 1){
    if(inputVals.length === 0) inputVals = [1];
    else inputVals[0] = 1;
  }
  let forwardA = null;
  if(inputVals.length === sizes[0]){
    forwardA = state.net.forward(inputVals).a;
  }
  const targetVals = ($('targets').value||'').split(',').map(s=>parseFloat(s.trim())).filter(n=>!isNaN(n));

  const r = 20; // node radius
  for(let l=0;l<layerCount;l++){
    for(let i=0;i<sizes[l];i++){
      const p = nodePositions[l][i];
      const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
      c.setAttribute('cx',p.x); c.setAttribute('cy',p.y); c.setAttribute('r',r);
      c.setAttribute('class','node'); if(l==layerCount-1) c.classList.add('out');
      svg.appendChild(c);

      // show value inside the node
      let val = '-';
      if(l===0){
        val = (inputVals[i] !== undefined) ? inputVals[i] : '-';
      } else if(forwardA){
        val = forwardA[l][i];
      }
      const fmt = v => (typeof v === 'number' ? v.toFixed(2) : v);
      const tv = document.createElementNS('http://www.w3.org/2000/svg','text');
      tv.setAttribute('x', p.x);
      tv.setAttribute('y', p.y + 4);
      tv.setAttribute('class','node-text');
      tv.textContent = fmt(val);
      svg.appendChild(tv);

      // for output layer show target nearby
      if(l === layerCount-1 && targetVals[i] !== undefined){
        const tt = document.createElementNS('http://www.w3.org/2000/svg','text');
        tt.setAttribute('x', p.x + r + 12);
        tt.setAttribute('y', p.y + 4);
        tt.setAttribute('class','target-label');
        tt.textContent = `t:${targetVals[i].toFixed(2)}`;
        svg.appendChild(tt);
      }
    }
  }
  // add small index labels for inputs and outputs
  for(let i=0;i<sizes[0];i++){
    const p = nodePositions[0][i];
    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x', p.x - r - 8);
    t.setAttribute('y', p.y + 4);
    t.setAttribute('class','label');
    t.textContent = `x${i}`;
    svg.appendChild(t);
  }
  for(let i=0;i<sizes[layerCount-1];i++){
    const p = nodePositions[layerCount-1][i];
    const t = document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x', p.x + r + 36);
    t.setAttribute('y', p.y + 4);
    t.setAttribute('class','label');
    t.textContent = `o${i}`;
    svg.appendChild(t);
  }
}

function onEdgeClick(ev){
  const l = parseInt(ev.target.dataset.layer), i = parseInt(ev.target.dataset.i), j = parseInt(ev.target.dataset.j);
  state.selectedEdge = {l,i,j};
  inspectSelected();
}

function inspectSelected(){
  const out = $('weights-inspector');
  if(!state.selectedEdge){ out.textContent = 'Click an edge to inspect weights'; return; }
  const {l,i,j} = state.selectedEdge; const w = state.net.weights[l][i][j];
  let s = `Layer ${l} -> ${l+1}  (i=${i}, j=${j})\nweight: ${w.toFixed(6)}\n\nRow weights to node ${i}:\n`;
  s += state.net.weights[l][i].map((v,idx)=>`  w[${idx}] = ${v.toFixed(6)}`).join('\n');
  s += `\n\nbias: ${state.net.biases[l][i].toFixed(6)}`;
  out.textContent = s;
}

// ---------- Loss plotting ----------
const lossCanvas = $('loss-plot'); const lossCtx = lossCanvas.getContext('2d');
function drawLoss(){
  const w = lossCanvas.width, h = lossCanvas.height; lossCtx.clearRect(0,0,w,h);
  const data = state.lossHistory.slice(-200);
  if(data.length===0) return;
  const max = Math.max(...data); const min = Math.min(...data);
  lossCtx.strokeStyle = '#60a5fa'; lossCtx.lineWidth=2; lossCtx.beginPath();
  data.forEach((v,idx)=>{
    const x = idx/(data.length-1)*(w-10)+5;
    const y = h - 5 - ( (v - min) / (max - min + 1e-9) )*(h-10);
    if(idx===0) lossCtx.moveTo(x,y); else lossCtx.lineTo(x,y);
  });
  lossCtx.stroke();
  lossCtx.fillStyle='#9ca3af'; lossCtx.font='12px monospace';
  lossCtx.fillText(`loss: ${data[data.length-1].toFixed(6)}`, 8, 14);
}

function updateInfo(){ $('loss').textContent = 'Loss: ' + (state.lossHistory.length?state.lossHistory[state.lossHistory.length-1].toFixed(6):'-');
  $('step-count').textContent = 'Steps: ' + state.steps; inspectSelected(); }

// ---------- Wire UI ----------
$('init-random').addEventListener('click', ()=>{ initRandom(); });
$('reset').addEventListener('click', ()=>{ buildNetwork(); });
$('start').addEventListener('click', ()=>{ startTraining(); });
$('pause').addEventListener('click', ()=>{ pauseTraining(); });
$('step').addEventListener('click', ()=>{ trainStep(parseInt($('iters').value)||1); });
$('activation').addEventListener('change', ()=>{ if(state.net){ state.net.activationName=$('activation').value; state.net.activation = Activations[state.net.activationName]; drawNetwork(); } });

// initialize
buildNetwork();
