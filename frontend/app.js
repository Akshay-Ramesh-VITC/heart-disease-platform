// Simple frontend behavior: switch tabs, collect form, call /api/predict, render charts
const tabs = document.querySelectorAll('.tab-btn');
const tabPanels = document.querySelectorAll('.tab');
tabs.forEach(btn=>btn.addEventListener('click',()=>{
  tabs.forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const tab=btn.dataset.tab;
  tabPanels.forEach(p=>p.id===tab? p.classList.remove('hidden') : p.classList.add('hidden'));
}));

const form = document.getElementById('patient-form');
const results = document.getElementById('results');
const probEl = document.getElementById('prob');
const modalityList = document.getElementById('modality-list');
const historyList = document.getElementById('history-list');

let gaugeChart=null, importanceChart=null;

function collectFormData(){
  const fd = new FormData(form);
  const obj = {};
  for(const [k,v] of fd.entries()) obj[k]= isNaN(v)? v : Number(v);
  return obj;
}

async function callPredict(payload){
  try{
    const res = await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if(!res.ok) throw new Error('No backend');
    return await res.json();
  }catch(e){
    // Fallback: fake response for demo
    console.warn('Backend not available, using demo response');
    return demoResponse(payload);
  }
}

function demoResponse(payload){
  const prob = Math.min(0.95, Math.abs((payload.age-50)*0.004 + (payload.systolic_bp-120)*0.002 + (payload.ldl-100)*0.003 + (payload.potassium||4)*0.05 + Math.random()*0.08));
  return {
    probability: Number((prob).toFixed(3)),
    modalities: {cardiovascular: prob*0.4, metabolic: prob*0.35, labs: prob*0.15, demographics: prob*0.1},
    feature_importance: [{name:'systolic_bp',value:Math.abs(payload.systolic_bp-120)||10},{name:'ldl',value:Math.abs(payload.ldl-100)||8},{name:'age',value:Math.abs(payload.age-50)||6}]
  };
}

function renderGauge(prob){
  const pct = Math.round(prob*100);
  probEl.textContent = pct;
  const ctx = document.getElementById('gaugeChart').getContext('2d');
  const data = {datasets:[{data:[pct,100-pct],backgroundColor:['#ff6b6b','#0b1220'],hoverOffset:4}]};
  if(gaugeChart) gaugeChart.destroy();
  gaugeChart = new Chart(ctx,{type:'doughnut',data,options:{cutout:'80%',plugins:{legend:{display:false}}}});
}

function renderModalities(mods){
  modalityList.innerHTML='';
  Object.entries(mods).forEach(([k,v])=>{
    const li = document.createElement('li');
    li.textContent = `${k}: ${(v*100).toFixed(1)}%`;
    modalityList.appendChild(li);
  });
}

function renderImportance(items){
  const ctx = document.getElementById('importanceChart').getContext('2d');
  const labels = items.map(i=>i.name);
  const values = items.map(i=>i.value);
  if(importanceChart) importanceChart.destroy();
  importanceChart = new Chart(ctx,{type:'bar',data:{labels, datasets:[{label:'Importance',data:values,backgroundColor:'#66b2ff'}]},options:{indexAxis:'y'}});
}

function saveToHistory(payload, resp){
  const rec = {time:new Date().toISOString(),payload,resp};
  const existing = JSON.parse(localStorage.getItem('hd_history')||'[]');
  existing.unshift(rec);
  localStorage.setItem('hd_history', JSON.stringify(existing.slice(0,20)));
  renderHistory();
}

function renderHistory(){
  const existing = JSON.parse(localStorage.getItem('hd_history')||'[]');
  historyList.innerHTML='';
  existing.forEach(r=>{
    const li=document.createElement('li');
    li.textContent = `${new Date(r.time).toLocaleString()} — ${Math.round(r.resp.probability*100)}%`;
    historyList.appendChild(li);
  });
}

form.addEventListener('submit',async (ev)=>{
  ev.preventDefault();
  const data = collectFormData();
  results.classList.remove('hidden');
  const resp = await callPredict(data);
  renderGauge(resp.probability);
  renderModalities(resp.modalities||{});
  renderImportance(resp.feature_importance||[]);
  saveToHistory(data,resp);
});

document.getElementById('save-local').addEventListener('click',()=>{
  const data = collectFormData();
  const resp = demoResponse(data);
  saveToHistory(data,resp);
  results.classList.remove('hidden');
  renderGauge(resp.probability);
  renderModalities(resp.modalities);
  renderImportance(resp.feature_importance);
});

// Init
renderHistory();
