// Live Quality Gate V1 dashboard view.
// Read-only: all values are supplied by the backend endpoint.
const qg={};
const qgMoney=v=>v==null?'—':`${Number(v)<0?'-':''}₹${Math.abs(Number(v)).toLocaleString('en-IN',{maximumFractionDigits:2})}`;
const qgNum=v=>v==null?'—':Number(v).toFixed(2);
const qgTime=t=>{if(!t)return '—';const d=new Date(t);return isNaN(d)?String(t):d.toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',hour12:false})};
const qgCls=s=>({WIN:'st-filled',LOSS:'st-bad',PENDING:'st-open',ACCEPTED:'st-filled',SKIPPED:'st-cancelled',AMBIGUOUS:'st-cancelled'}[String(s||'').toUpperCase()]||'');
function renderQualityGate(s){
  if(!s||!$('#qualityGatePanel'))return;
  const m=s.summary||s;
  $('#qgGenerated').textContent=m.generated??m.total??0;
  $('#qgAccepted').textContent=m.accepted??0;
  $('#qgSkipped').textContent=m.skipped??0;
  $('#qgResolved').textContent=m.resolved??0;
  $('#qgWins').textContent=m.wins??0;
  $('#qgLosses').textContent=m.losses??0;
  $('#qgPending').textContent=m.pending??0;
  $('#qgAccuracy').textContent=m.accuracy==null?'N/A':`${Number(m.accuracy).toFixed(1)}%`;
  $('#qgPf').textContent=m.profit_factor==null?'N/A':Number(m.profit_factor).toFixed(2);
  $('#qgPnl').textContent=qgMoney(m.pnl??m.net_pnl??0);
  $('#qgDd').textContent=qgMoney(m.max_drawdown??0);
  $('#qgAcceptRate').textContent=m.acceptance_rate==null?'0%':`${Number(m.acceptance_rate).toFixed(1)}%`;
  $('#qgTarget').textContent=m.target_status||'INSUFFICIENT SAMPLE';
  $('#qgTargetDetail').textContent=m.target_detail||'Needs ≥30 resolved';
  $('#qgUpdated').textContent=m.updated_at?qgTime(m.updated_at):new Date().toLocaleTimeString('en-IN',{hour12:false});
  const ok=String(m.target_status||'').toUpperCase()==='TARGET ACHIEVED';
  $('#qgTarget').className=ok?'positive':(m.resolved>=30?'warn':'neutral');
  $('#qgBadge').textContent=m.live_orders===0?'PAPER · NO ORDERS':'CHECK SAFETY';
  $('#qgBadge').className=`badge ${m.live_orders===0?'good':'bad'}`;
  const rows=s.signals||[];
  $('#qgSignals').innerHTML=rows.length?rows.map(r=>`<tr>
    <td class="nowrap">${esc(qgTime(r.timestamp||r.time))}</td>
    <td class="${String(r.direction).toUpperCase()==='BUY'?'side-long':'side-short'}">${esc(r.direction||'—')}</td>
    <td>${r.gate_score==null?'—':`${esc(r.gate_score)}/${esc(r.gate_max_score??3)}`}</td>
    <td><span class="status-pill ${qgCls(r.decision)}">${esc(r.decision||'—')}</span></td>
    <td>${esc(r.contract||r.option||'—')}</td>
    <td>${qgNum(r.entry)}</td><td>${qgNum(r.stop)}</td><td>${qgNum(r.target)}</td>
    <td><span class="status-pill ${qgCls(r.status)}">${esc(r.status||'—')}</span></td>
    <td class="${Number(r.pnl||0)>0?'positive':Number(r.pnl||0)<0?'negative':''}">${qgMoney(r.pnl)}</td>
    <td>${qgNum(r.r_multiple)}</td>
  </tr>`).join(''):'<tr><td colspan="11" class="empty-row">No Quality Gate signals recorded today.</td></tr>';
  $('#qgNote').textContent=m.note||'Quality Gate is observation-only. It does not place orders or modify production S2.';
}
async function refreshQualityGate(){
  try{const s=await get('/api/live-quality-gate');renderQualityGate(s)}catch(e){
    if($('#qgBadge')){$('#qgBadge').textContent='API UNAVAILABLE';$('#qgBadge').className='badge warn'}
  }
}
refreshQualityGate();setInterval(refreshQualityGate,2000);
