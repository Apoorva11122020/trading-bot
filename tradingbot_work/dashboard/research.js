const researchState={status:null,result:null,split:'TRAIN',selected:null};

function researchRow(r){
  return `<tr data-strategy="${esc(r.strategy_id)}" data-underlying="${esc(r.underlying)}" class="research-row">
    <td>${esc(r.underlying)}</td><td><button class="link-btn strategy-link" type="button">${esc(r.strategy_id)}</button></td>
    <td>${Number(r.profit_factor||0).toFixed(2)}</td><td>${money(r.max_drawdown||0)}</td>
    <td>${(Number(r.win_rate||0)*100).toFixed(1)}%</td><td>${Number(r.average_r||0).toFixed(2)}</td>
    <td class="${Number(r.total_pnl||0)>=0?'positive':'negative'}">${money(r.total_pnl||0)}</td><td>${r.resolved_trades??0}</td></tr>`;
}

async function postResearch(path, body){
  let k=sessionStorage.getItem('dashboardApiKey')||'';
  if(!k){k=prompt('Dashboard API key (required for control actions)')||'';if(!k)return null;sessionStorage.setItem('dashboardApiKey',k)}
  const r=await fetch(API_BASE+path,{method:'POST',headers:{'Content-Type':'application/json','X-API-Key':k},body:JSON.stringify(body)});
  if(!r.ok){alert(`Request failed (${r.status})`);return null}
  return r.json();
}

function renderHistoricalPanel(status){
  const badge=$('#historicalStatusBadge');
  const body=$('#historicalDataBody');
  if(!badge||!body)return;
  const latest=status?.latest_job;
  const token=status?.token_configured?'configured':'missing';
  badge.textContent=status?.download_running?'DOWNLOADING':latest?.status||'READY';
  const summary=status?.summary||{};
  const lines=Object.entries(summary).map(([u,d])=>{
    const opt=d.options||{};
    return `<div><strong>${esc(u)}</strong><span>${opt.rows||0} option rows · ${opt.instruments||0} contracts</span></div>`;
  }).join('');
  body.innerHTML=`<div class="research-kv"><span>Token</span><strong>${esc(token)}</strong></div>
    <div class="research-kv"><span>Source</span><strong>${esc(status?.data_source||'Upstox Historical Data')}</strong></div>
    <div class="research-kv"><span>Window</span><strong>${esc(status?.session?.allow_entries_from||'09:15')}–${esc(status?.session?.stop_new_entries||'14:30')}</strong></div>
    <div class="research-kv"><span>Daily loss cap</span><strong>₹${Number(status?.max_daily_loss_inr||3000).toLocaleString('en-IN')}</strong></div>
    <div class="research-summary">${lines||'<span class="muted-text">No parquet data stored yet.</span>'}</div>`;
}

function renderResearchTable(){
  const tbody=$('#researchTableBody');
  const panel=$('#researchSignalsPanel');
  if(!tbody)return;
  const rows=[];
  const comparisons=researchState.result?.comparisons||{};
  Object.entries(comparisons).forEach(([underlying,comp])=>{
    (comp.rows||[]).filter(r=>r.split===researchState.split).forEach(r=>rows.push({...r,underlying}));
  });
  tbody.innerHTML=rows.length?rows.map(researchRow).join(''):`<tr><td colspan="8" class="empty-row">No backtest results for ${esc(researchState.split)} yet.</td></tr>`;
  if(panel&&!researchState.selected)panel.classList.add('hidden');
}

async function loadResearchStatus(){
  try{researchState.status=await get('/api/historical/options/status')}catch(e){researchState.status=null}
  renderHistoricalPanel(researchState.status);
  const badge=$('#researchRunBadge');
  if(badge)badge.textContent=researchState.result?.status||'IDLE';
}

async function renderResearch(){
  await loadResearchStatus();
  renderResearchTable();
}

document.addEventListener('click',async e=>{
  const tab=e.target.closest('#researchSplitTabs button');
  if(tab){researchState.split=tab.dataset.split;$$('#researchSplitTabs button').forEach(b=>b.classList.toggle('active',b===tab));renderResearchTable();return}
  const link=e.target.closest('.strategy-link');
  if(link){
    const row=link.closest('.research-row');
    const strategy=row?.dataset.strategy;
    const underlying=row?.dataset.underlying;
    const panel=$('#researchSignalsPanel');
    if(!panel||!researchState.result?.run_id)return;
    try{
      const data=await get(`/api/backtest/options/signals/${researchState.result.run_id}`);
      const signals=(data.signals||[]).filter(s=>s.strategy_id===strategy&&s.underlying===underlying).slice(0,20);
      panel.classList.remove('hidden');
      panel.innerHTML=`<div class="panel-head"><div><span class="eyebrow">SIGNALS</span><h3>${esc(strategy)} · ${esc(underlying)}</h3></div></div>
        ${signals.length?signals.map(s=>`<div class="signal-line"><time>${esc(fmtTime(s.timestamp))}</time><strong>${esc(s.direction)}</strong><span>${esc(JSON.stringify(s.conditions||{}))}</span></div>`).join(''):'<p class="muted-text">No stored signals for this strategy.</p>'}`;
    }catch(err){panel.classList.remove('hidden');panel.innerHTML='<p class="muted-text">Signals unavailable.</p>'}
    return;
  }
  if(e.target.id==='downloadHistoricalBtn'){
    const today=new Date();
    const to=today.toISOString().slice(0,10);
    const from=new Date(today.getTime()-7*86400000).toISOString().slice(0,10);
    const job=await postResearch('/api/historical/options/download',{underlyings:['NIFTY'],from_date:from,to_date:to,include_options:true,include_futures:true});
    if(job)alert(`Download started (job ${job.job_id||'queued'})`);
    await renderResearch();
  }
});
