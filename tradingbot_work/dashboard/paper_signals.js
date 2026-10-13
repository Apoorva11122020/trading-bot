// Live PAPER signal validation view. Display only: every value comes from /api/paper/* (backend trackers);
// nothing here calculates a signal, a level or an outcome.
const pv={snap:null,source:null,day:'',lastTick:null};
const pvNum=v=>v===null||v===undefined?'—':Number(v).toLocaleString('en-IN',{maximumFractionDigits:2});
const pvMoney=v=>v===null||v===undefined?'—':`${v<0?'-':''}\u20B9${Math.abs(Number(v)).toLocaleString('en-IN',{maximumFractionDigits:2})}`;
const pvClock=t=>{if(!t)return '—';const d=new Date(t);return isNaN(d)?String(t):d.toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false})};
const pvStatusCls=s=>({WIN:'st-filled',LOSS:'st-bad',PENDING:'st-open',AMBIGUOUS:'st-cancelled',UNRESOLVED:'st-cancelled'}[s]||'');
const pvSet=(sel,text,cls)=>{const el=$(sel);if(!el)return;el.textContent=text;if(cls!==undefined)el.className=cls};

function pvRenderSummary(s){
  const m=s.summary;
  pvSet('#pvMarket',s.market.state,`kpi-value ${s.market.state==='LIVE'?'positive':'neutral'}`);
  pvSet('#pvMode',s.mode,`kpi-value ${s.mode==='PAPER'?'positive':'negative'}`);
  pvSet('#pvTotal',m.total);pvSet('#pvResolved',m.resolved);pvSet('#pvPending',m.pending);
  pvSet('#pvWins',m.wins);pvSet('#pvLosses',m.losses);
  pvSet('#pvTotalFoot',`${s.date}${pv.day?' · stored session':' · today'}`);
  pvSet('#pvOtherFoot',`Ambiguous ${m.ambiguous} · Unresolved ${m.unresolved}`);
  pvSet('#pvAccuracy',m.accuracy===null?'N/A':`${m.accuracy.toFixed(2)}%`);
  const lbl=m.accuracy_label||m.accuracy_status;
  pvSet('#pvAccuracyStatus',lbl==='N/A'?lbl:`${lbl} · ${m.resolved} resolved`,'badge warn');
  pvSet('#pvPnl',pvMoney(m.simulated_pnl),`kpi-value ${m.simulated_pnl>0?'positive':m.simulated_pnl<0?'negative':''}`);
  pvSet('#pvDd',pvMoney(m.max_drawdown));
  pvSet('#pvPf',m.profit_factor===null?(m.gross_profit>0&&m.pnl_signals?'No losses':'—'):m.profit_factor.toFixed(2));
  const c=s.cumulative;pvSet('#pvCumulative',`ALL SESSIONS · ${c.total} signals · ${c.accuracy===null?'N/A':c.accuracy.toFixed(2)+'%'} ${c.accuracy_label||'HISTORICAL ACCURACY'}`);
}

function pvRenderActive(s){
  const a=s.active,el=$('#pvActive');
  if(!a){el.innerHTML=`<div class="panel-head"><div><span class="eyebrow">ACTIVE SIGNAL</span><h2>NIFTY</h2></div></div><div class="pv-none">NO ACTIVE SIGNAL</div><div class="research-note">${pv.day?'Viewing a stored session; active signals are shown for today only.':'The card fills as soon as a tracker records a new signal.'}</div>`;return}
  const unit=a.unit||'';
  const prem=a.premium_stale?(a.current_display||'STALE'):`${pvNum(a.current)}${a.premium_freshness==='CONNECTED'?' · FRESH':''}`;
  const premCls=a.premium_stale?'negative':'';
  const cells=[['Direction',`<span class="${a.direction==='BUY'?'side-long':'side-short'}">${esc(a.direction)}</span>`],['Option',esc(a.option||'NIFTY spot')],['Entry',a.awaiting_entry?'Awaiting entry candle':pvNum(a.entry)],['SL',pvNum(a.stop_loss)],['Target',pvNum(a.target)],['Current Premium',a.basis==='OPTION_PREMIUM'?`<span class="${premCls}">${esc(String(prem))}</span>`:'—'],['Distance to SL',pvNum(a.distance_to_sl)],['Distance to Target',pvNum(a.distance_to_target)],['Status',`<span class="status-pill st-open">${esc(a.status)}</span>`],['Signal Time',esc(pvClock(a.time))],['Last Update',esc(pvClock(a.current_time))]];
  if(a.basis!=='OPTION_PREMIUM')cells.splice(5,1,['Current NIFTY',pvNum(a.current)]);
  el.innerHTML=`<div class="panel-head"><div><span class="eyebrow">ACTIVE SIGNAL · ${esc(a.strategy||'')}</span><h2>NIFTY ${s.market.nifty.price!=null?pvNum(s.market.nifty.price):''}</h2></div><span class="status-pill st-open">PENDING</span></div><div class="pv-active">${cells.map(([k,v])=>`<div><span>${k}</span><strong>${v}</strong></div>`).join('')}</div><div class="research-note">Levels in ${esc(unit)} · source ${esc((a.sources||[]).join(' + '))}${a.expiry?` · expiry ${esc(a.expiry)}`:''}</div>`;
}

function pvHealthCls(v){return v==='CONNECTED'||v==='RUNNING'?true:v==='STALE'||v==='DEGRADED'?undefined:false}
function pvRenderHealth(s){
  const h=s.health;
  const row=(k,v,good)=>`<div><span>${k}</span><strong class="${good===true?'positive':good===false?'negative':'neutral'}">${v}</strong></div>`;
  const tickAge=h.premium_tick_age_seconds??h.premium_data_age_seconds;
  const tickAgeTxt=tickAge==null?'—':`${Number(tickAge).toFixed(0)} s`;
  const feed=h.option_premium_feed||h.premium_data;
  $('#pvHealth').innerHTML=[row('Market Data',h.market_data,pvHealthCls(h.market_data)),row('WebSocket',h.websocket,pvHealthCls(h.websocket)),row('NIFTY Last Tick',esc(fmtTime(h.last_tick))),row('NIFTY Last Completed Candle',esc(fmtTime(h.last_completed_candle))),`<div><span>NIFTY Data Age</span><strong id="pvAge">—</strong></div>`,row('Signal Engine',h.signal_engine,pvHealthCls(h.signal_engine)),row('Option Premium Tracker',h.premium_tracker,pvHealthCls(h.premium_tracker)),row('Premium Subscription',h.premium_subscription||'—',pvHealthCls(h.premium_subscription)),row('Option Premium Feed',feed,pvHealthCls(feed)),row('Option Tick Age',tickAgeTxt,pvHealthCls(feed)),row('Last Option Tick',esc(fmtTime(h.premium_last_tick))),row('Last Completed Option Candle',esc(fmtTime(h.premium_last_candle))),row('Broker Execution',esc(h.broker_execution)),row('Broker Orders',h.broker_orders,h.broker_orders===0)].join('');
  const overall=h.overall_status||h.system;
  pvSet('#pvStale',overall==='RUNNING'?'HEALTHY':overall==='DEGRADED'?'DEGRADED — PREMIUM STALE':'STOPPED',`badge ${overall==='RUNNING'?'good':overall==='DEGRADED'?'warn':'bad'}`);
  pv.lastTick=h.last_tick;pvAge();
  if(typeof renderPaperSidebar==='function')renderPaperSidebar(h);
}

function pvAge(){
  const el=$('#pvAge');if(!el)return;
  if(!pv.lastTick){el.textContent='—';return}
  const sec=Math.max(0,(Date.now()-new Date(pv.lastTick).getTime())/1000);
  const stale=pv.snap?.health?.stale_after_seconds??15;
  el.textContent=`${sec<120?sec.toFixed(0)+' s':sec<7200?(sec/60).toFixed(0)+' min':(sec/3600).toFixed(1)+' h'}`;
  el.className=sec>stale?'negative':'positive';
}

function pvRenderSignals(s){
  pvSet('#pvTableTitle',`Signals · ${s.date}`);
  const rows=s.signals||[];
  const cur=r=>r.status==='PENDING'?(r.premium_stale?(r.current_display||'STALE'):pvNum(r.current)):(r.exit_price??r.current);
  $('#pvSignals').innerHTML=rows.length?rows.map(r=>`<tr><td class="nowrap">${esc(pvClock(r.time))}</td><td class="${r.direction==='BUY'?'side-long':'side-short'}">${esc(r.direction)}</td><td>${r.option?`${esc(r.option)}${r.option_type?` <span class="type-tag ${esc(r.option_type.toLowerCase())}">${esc(r.option_type)}</span>`:''}`:'<span class="muted-text">NIFTY spot</span>'}</td><td>${pvNum(r.strike)}</td><td>${pvNum(r.entry)}</td><td>${pvNum(r.stop_loss)}</td><td>${pvNum(r.target)}</td><td class="${r.premium_stale?'negative':''}">${esc(String(cur(r)))}</td><td><span class="status-pill ${pvStatusCls(r.status)}" title="${esc(r.reason||'')}">${esc(r.status)}</span></td><td class="${r.pnl_rupees>0?'positive':r.pnl_rupees<0?'negative':''}">${pvMoney(r.pnl_rupees)}</td></tr>`).join(''):`<tr><td colspan="10" class="empty-row">No signals recorded for ${esc(s.date)}</td></tr>`;
}

function pvRenderDaily(s){
  const days=s.daily||[];
  $('#pvDaily').innerHTML=days.length?days.map(d=>`<tr><td>${esc(d.date)}</td><td>${d.total}</td><td class="positive">${d.wins}</td><td class="negative">${d.losses}</td><td>${d.pending}</td><td>${d.accuracy===null?'N/A':d.accuracy.toFixed(2)+'%'} <span class="muted-text">${esc(d.accuracy_label||'HISTORICAL ACCURACY')}</span></td><td class="${d.simulated_pnl>0?'positive':d.simulated_pnl<0?'negative':''}">${pvMoney(d.simulated_pnl)}</td></tr>`).join(''):'<tr><td colspan="7" class="empty-row">No stored paper sessions yet</td></tr>';
  $('#pvNotes').innerHTML=(s.notes||[]).map(n=>`<li>${esc(n)}</li>`).join('');
  const sel=$('#pvDate'),have=new Set([...sel.options].map(o=>o.value));
  days.forEach(d=>{if(!have.has(d.date)){const o=document.createElement('option');o.value=d.date;o.textContent=d.date;sel.appendChild(o)}});
}

function pvChart(canvas,{series,levels=[],markers=[],empty}){
  if(!canvas||canvas.offsetParent===null)return;
  const r=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;canvas.width=r.width*dpr;canvas.height=r.height*dpr;
  const x=canvas.getContext('2d');x.scale(dpr,dpr);const w=r.width,h=r.height,cs=getComputedStyle(document.documentElement);
  const col=n=>cs.getPropertyValue(n).trim()||'#888';x.font='11px sans-serif';
  if(!series.length){x.fillStyle=col('--muted');x.fillText(empty,12,h/2);return}
  const vals=[...series.flatMap(p=>[p.low??p.v,p.high??p.v]),...levels.map(l=>l.v),...markers.map(m=>m.v)].filter(v=>v!=null);
  const lo=Math.min(...vals),hi=Math.max(...vals),pad=(hi-lo||1)*.08,L=lo-pad,H=hi+pad,left=8,right=w-62;
  const t0=series[0].t,t1=series[series.length-1].t,X=t=>left+(t1===t0?0.5:(t-t0)/(t1-t0))*(right-left),Y=v=>h-14-(v-L)/(H-L)*(h-28);
  x.strokeStyle=col('--border');x.lineWidth=1;for(let i=0;i<=4;i++){const yy=14+i*(h-28)/4;x.beginPath();x.moveTo(left,yy);x.lineTo(right,yy);x.stroke();x.fillStyle=col('--muted');x.fillText((H-i*(H-L)/4).toFixed(1),right+6,yy+4)}
  if(series[0].open!==undefined){const bw=Math.max(2,Math.min(9,(right-left)/series.length*.6));series.forEach(c=>{const up=c.close>=c.open;x.strokeStyle=x.fillStyle=col(up?'--pos':'--neg');x.beginPath();x.moveTo(X(c.t),Y(c.high));x.lineTo(X(c.t),Y(c.low));x.stroke();const y1=Y(Math.max(c.open,c.close)),y2=Y(Math.min(c.open,c.close));x.fillRect(X(c.t)-bw/2,y1,bw,Math.max(1,y2-y1))})}
  else{x.strokeStyle=col('--accent');x.lineWidth=1.6;x.beginPath();series.forEach((p,i)=>i?x.lineTo(X(p.t),Y(p.v)):x.moveTo(X(p.t),Y(p.v)));x.stroke()}
  levels.forEach(l=>{x.setLineDash(l.dash||[5,4]);x.strokeStyle=col(l.color);x.lineWidth=1;x.beginPath();x.moveTo(left,Y(l.v));x.lineTo(right,Y(l.v));x.stroke();x.setLineDash([]);x.fillStyle=col(l.color);x.fillText(l.label,left+4,Y(l.v)-3)});
  markers.forEach(m=>{const px=X(Math.min(Math.max(m.t,t0),t1)),py=Y(m.v),up=m.dir==='BUY';x.fillStyle=col(up?'--pos':'--neg');x.beginPath();if(up){x.moveTo(px,py+4);x.lineTo(px-5,py+13);x.lineTo(px+5,py+13)}else{x.moveTo(px,py-4);x.lineTo(px-5,py-13);x.lineTo(px+5,py-13)}x.closePath();x.fill()});
}

function pvRenderChart(){
  const s=pv.snap;if(!s)return;const ch=s.chart,T=t=>new Date(t).getTime();
  const candles=ch.candles.map(c=>({t:T(c.time),...c}));
  const levels=[];
  if(ch.opening_range){levels.push({v:ch.opening_range.high,label:'ORB high',color:'--muted',dash:[2,3]},{v:ch.opening_range.low,label:'ORB low',color:'--muted',dash:[2,3]})}
  if(ch.levels){levels.push({v:ch.levels.entry,label:'Entry',color:'--accent'},{v:ch.levels.stop_loss,label:'SL',color:'--neg'},{v:ch.levels.target,label:'Target',color:'--pos'})}
  pvChart($('#pvChart'),{series:candles,levels:levels.filter(l=>l.v!=null),markers:ch.markers.filter(m=>m.price!=null).map(m=>({t:T(m.time),v:m.price,dir:m.direction})),empty:'No completed NIFTY 5-minute candles recorded by the realtime tracker for this session'});
  pvSet('#pvCandle',s.market.latest_candle?`Last candle ${pvClock(s.market.latest_candle.timestamp)} · C ${pvNum(s.market.latest_candle.close)}`:'Last candle —');
  pvSet('#pvChartNote',[ch.opening_range?ch.opening_range.label:null,ch.vwap_note,'Markers = S2 signal candles from the trackers.'].filter(Boolean).join(' · '));
  const p=ch.premium;$('#pvPremiumWrap').classList.toggle('hidden',!p);
  if(p){pvSet('#pvPremiumTitle',`Option premium · ${p.option||''} (completed 5-minute candles)`);pvChart($('#pvPremiumChart'),{series:(p.history||[]).map(q=>({t:T(q.time),v:q.close})),levels:[{v:p.entry,label:'Entry',color:'--accent'},{v:p.stop_loss,label:'SL',color:'--neg'},{v:p.target,label:'Target',color:'--pos'}].filter(l=>l.v!=null),empty:'Waiting for the first completed premium candle from the tracker'})}
}

function pvRender(s){
  pv.snap=s;pvRenderSummary(s);pvRenderActive(s);pvRenderHealth(s);pvRenderSignals(s);pvRenderDaily(s);pvRenderChart();
  pvSet('#pvNifty',`NIFTY ${pvNum(s.market.nifty.price)} · ${pvClock(s.market.nifty.time)}`);
}

function pvTick(t){
  if(!pv.snap)return;pv.snap.market.nifty=t.nifty;pv.snap.health.last_tick=t.last_tick;pv.lastTick=t.last_tick;
  pvSet('#pvNifty',`NIFTY ${pvNum(t.nifty.price)} · ${pvClock(t.nifty.time)}`);pvAge();
}

function pvConnect(){
  if(pv.source)pv.source.close();
  const q=new URLSearchParams();if(pv.day)q.set('date',pv.day);if(apiKey())q.set('api_key',apiKey());
  const src=new EventSource(`${API_BASE}/api/paper/stream${q.toString()?'?'+q:''}`);pv.source=src;
  pvSet('#pvStream','CONNECTING','badge warn');
  src.addEventListener('open',()=>pvSet('#pvStream','LIVE STREAM','badge good'));
  src.addEventListener('snapshot',e=>{pvSet('#pvStream','LIVE STREAM','badge good');pvRender(JSON.parse(e.data))});
  src.addEventListener('tick',e=>pvTick(JSON.parse(e.data)));
  src.addEventListener('error',()=>pvSet('#pvStream',src.readyState===EventSource.CLOSED?'STREAM CLOSED':'RECONNECTING','badge bad'));
}

function paperViewShown(){if(!pv.source||pv.source.readyState===EventSource.CLOSED)pvConnect();else pvRenderChart()}
$('#pvDate').addEventListener('change',e=>{pv.day=e.target.value;pvConnect()});
window.addEventListener('resize',pvRenderChart);
setInterval(pvAge,1000);
pvConnect();
