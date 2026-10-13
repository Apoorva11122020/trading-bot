// ---------------------------------------------------------------------------
// Option (NSE F&O) console. Uses helpers from app.js ($, esc, money, get, kv,
// orDash, pct, timelineHtml, apiKey, API_BASE, refresh). Every stage shown
// comes from /api/options/demo/status; quotes are never invented.
// ---------------------------------------------------------------------------
const opt={type:'EQUITY',timer:null,refreshedFor:null,chain:null,expiries:[],last:null};
const OPT_CONDITIONS={ema_trend:'EMA Trend',vwap:'VWAP',rsi:'RSI',breakout:'Breakout',volume:'Volume'};
const num=v=>v==null||v===''?'—':Number(v).toLocaleString('en-IN',{maximumFractionDigits:2});

function setTradeType(type){
  opt.type=type;
  $$('.trade-tab').forEach(b=>b.classList.toggle('active',b.dataset.tradeType===type));
  const isOption=type!=='EQUITY';
  $('#optionConsole').classList.toggle('hidden',!isOption);
  $('#demoConsole').classList.toggle('hidden',isOption);
  if(isOption){
    $('#optUnderlyingLabel').textContent=type==='INDEX_OPTION'?'Index':'Stock';
    loadUnderlyings(type);
  }
}

function optAlert(title,detail,cls='bad'){const a=$('#optAlert');a.innerHTML=`<strong>${esc(title)}</strong><div>${esc(detail)}</div>`;a.className=`demo-alert ${cls}`}
function errText(e){if(!e)return'';if(typeof e==='string')return e;if(e.detail)return typeof e.detail==='string'?e.detail:(e.detail.message||JSON.stringify(e.detail));return JSON.stringify(e)}

async function optGet(url){
  const r=await fetch(API_BASE+url,{cache:'no-store',headers:headers()});
  let body={};try{body=await r.json()}catch(e){}
  if(!r.ok)throw new Error(errText(body)||`HTTP ${r.status}`);
  return body;
}

async function loadUnderlyings(type){
  const sel=$('#optUnderlying');
  sel.innerHTML='<option value="">Loading instruments…</option>';
  try{
    const d=await optGet(`/api/options/underlyings?market_type=${type}`);
    const preferred=type==='INDEX_OPTION'?d.configured.index:d.configured.stock;
    const listed=d.underlyings.filter(u=>preferred.includes(u.symbol));
    const others=d.underlyings.filter(u=>!preferred.includes(u.symbol));
    const option=u=>`<option value="${esc(u.symbol)}">${esc(u.symbol)} · lot ${esc(u.lot_size)}</option>`;
    sel.innerHTML=listed.map(option).join('')+(others.length?`<optgroup label="Other NSE F&amp;O">${others.map(option).join('')}</optgroup>`:'');
    if(!d.underlyings.length)sel.innerHTML='<option value="">No listed options</option>';
    await loadExpiries();
  }catch(e){
    sel.innerHTML='<option value="">Instrument data unavailable</option>';
    optAlert('OPTION INSTRUMENTS UNAVAILABLE',e.message);
  }
}

async function loadExpiries(){
  const u=$('#optUnderlying').value;const sel=$('#optExpiry');
  if(!u){sel.innerHTML='';return}
  try{
    const d=await optGet(`/api/options/expiries?underlying=${encodeURIComponent(u)}`);
    opt.expiries=d.expiries;
    const [n,x]=d.expiries;
    sel.innerHTML=`<option value="nearest">Nearest · ${esc(n||'—')}</option>`+(x?`<option value="next">Next · ${esc(x)}</option>`:'')+d.expiries.map(e=>`<option value="date:${esc(e)}">${esc(e)}</option>`).join('');
    await loadChain();
  }catch(e){sel.innerHTML='';optAlert('EXPIRIES UNAVAILABLE',e.message)}
}

function selectedExpiryDate(){
  const v=$('#optExpiry').value||'nearest';
  if(v.startsWith('date:'))return v.slice(5);
  return v==='next'?opt.expiries[1]:opt.expiries[0];
}

function quoteCells(contract,q,side){
  if(!contract)return '<td class="muted">—</td><td class="muted">—</td><td class="muted">not listed</td>';
  const attrs=`class="pick ${side.toLowerCase()}" data-pick="${side}" title="Select ${esc(contract.trading_symbol)} (${esc(contract.instrument_key)})"`;
  return `<td ${attrs}>${num(q?.ltp)}</td><td ${attrs}>${num(q?.volume)}</td><td ${attrs}>${num(q?.oi)}</td>`;
}

async function loadChain(){
  const u=$('#optUnderlying').value;const expiry=selectedExpiryDate();
  if(!u||!expiry)return;
  try{
    const c=await optGet(`/api/options/chain?underlying=${encodeURIComponent(u)}&expiry=${encodeURIComponent(expiry)}`);
    opt.chain=c;
    $('#optChainTitle').textContent=`Option chain · ${c.underlying} · ${c.expiry}`;
    const notice=$('#optChainNotice');
    notice.classList.toggle('hidden',c.market_data_available);
    notice.textContent=c.market_data_message||'Option chain requires market-data credentials';
    $('#optChainBadge').textContent=c.market_data_available?'MARKET DATA':`${c.rows.length} LISTED STRIKES · NO QUOTES`;
    $('#optChainBadge').title=c.spot_source==='REQUEST'||c.spot_source==='MARKET_DATA'?'':'ATM is relative to the demo reference spot (median listed strike), not a market price';
    const rows=c.rows;const atmIndex=Math.max(0,rows.findIndex(r=>r.strike===c.atm_strike));
    const view=rows.slice(Math.max(0,atmIndex-12),atmIndex+13);
    $('#optChainBody').innerHTML=view.map(r=>`<tr class="${r.strike===c.atm_strike?'atm':''}" data-strike="${r.strike}"><td class="strike-col">${num(r.strike)}${r.strike===c.atm_strike?' <span class="atm-tag">ATM</span>':''}</td>${quoteCells(r.ce,r.ce_quote,'CE')}${quoteCells(r.pe,r.pe_quote,'PE')}</tr>`).join('')||'<tr><td colspan="7">No contracts</td></tr>';
    const strikeSel=$('#optStrike');const current=strikeSel.value;
    strikeSel.innerHTML=rows.map(r=>`<option value="${r.strike}">${num(r.strike)}${r.strike===c.atm_strike?' (ATM*)':''}</option>`).join('');
    strikeSel.value=current&&rows.some(r=>String(r.strike)===current)?current:String(c.atm_strike??'');
    await resolveContract();
  }catch(e){
    $('#optChainBody').innerHTML=`<tr><td colspan="7">${esc(e.message)}</td></tr>`;
  }
}

function demoParams(){
  const exp=$('#optExpiry').value||'nearest';
  const p={market_type:opt.type,underlying:$('#optUnderlying').value,expiry_mode:exp.startsWith('date:')?'date':exp,option_type:$('#optType').value,strike_mode:$('#optStrikeMode').value,strike_offset:Number($('#optOffset').value||0),signal_source:$('#optSource').value};
  if(exp.startsWith('date:'))p.expiry_date=exp.slice(5);
  if(p.strike_mode==='EXACT')p.strike=Number($('#optStrike').value);
  return p;
}

async function resolveContract(){
  const p=demoParams();const box=$('#optContract');
  $('#optStrike').disabled=p.strike_mode!=='EXACT';
  $('#optOffset').disabled=p.strike_mode==='ATM'||p.strike_mode==='EXACT';
  if(!p.underlying){box.innerHTML='<span class="demo-empty">Select an underlying.</span>';return}
  const types=p.option_type==='AUTO'?['CE','PE']:[p.option_type];
  const q=t=>{const s=new URLSearchParams({underlying:p.underlying,option_type:t,expiry_mode:p.expiry_mode,strike_mode:p.strike_mode,strike_offset:p.strike_offset});if(p.expiry_date)s.set('expiry_date',p.expiry_date);if(p.strike)s.set('strike',p.strike);return `/api/options/resolve?${s}`};
  try{
    const found=await Promise.all(types.map(t=>optGet(q(t)).then(r=>({t,r})).catch(e=>({t,e}))));
    box.innerHTML=found.map(({t,r,e})=>e?`<div class="opt-contract-item bad"><b>${t}</b><span>${esc(e.message)}</span></div>`:`<div class="opt-contract-item" title="${esc(r.contract.instrument_key)} · freeze ${esc(r.contract.freeze_quantity??'—')} · strike vs ${r.spot_source==='REQUEST'?'requested spot':'demo reference spot'} ${num(r.spot)}"><span class="type-tag ${t.toLowerCase()}">${t}</span><span class="mono">${esc(r.contract.trading_symbol)}</span><small class="ci-meta">${p.option_type==='AUTO'?(t==='CE'?'on BUY signal':'on SELL signal'):'selected'} · Lot ${esc(r.contract.lot_size)}</small></div>`).join('');
  }catch(e){box.innerHTML=`<span class="demo-empty">${esc(e.message)}</span>`}
}

function pickFromChain(ev){
  const cell=ev.target.closest('td.pick');if(!cell)return;
  const strike=cell.closest('tr').dataset.strike;
  $('#optType').value=cell.dataset.pick;$('#optStrikeMode').value='EXACT';$('#optStrike').value=strike;
  $$('#optChainBody td.pick').forEach(td=>td.classList.remove('selected'));
  cell.closest('tr').querySelectorAll(`td[data-pick="${cell.dataset.pick}"]`).forEach(td=>td.classList.add('selected'));
  resolveContract();
}

function optSignalCard(s){
  const g=s.signal;const d=s.decision;
  if(!g&&!d)return `<div class="eyebrow">OPTION SIGNAL</div><div class="demo-empty">${s.running?'Evaluating strategy…':'No signal yet. Click START OPTION DEMO.'}</div>`;
  const conds=Object.keys(OPT_CONDITIONS).map(k=>{const v=(g?.conditions||d?.conditions||{})[k];return `<div class="cond ${v==null?'':v?'ok':'bad'}"><span>${OPT_CONDITIONS[k]}</span><b>${v==null?'—':v?'✓':'✕'}</b></div>`}).join('');
  const action=g?.action||d?.action;
  return `<div class="eyebrow">OPTION SIGNAL · ${esc(d?.source||'UNDERLYING')}</div><div class="demo-kvs">${kv('Action',orDash(action),action==='BUY CALL'?'side-long':'side-short')}${kv('Underlying',orDash(s.underlying))}${kv('Spot (demo ref.)',num(s.spot))}${kv('Contract',orDash(s.trading_symbol))}${kv('Expiry',orDash(s.expiry))}${kv('Strike',num(s.strike))}${kv('Entry Premium',g?money(g.entry_price):'—')}${kv('Stop Loss',g?money(g.stop_loss):'—')}${kv('Target',g?money(g.target):'—')}${kv('Lots × Lot Size',g?`${esc(g.lots)} × ${esc(g.lot_size)} = ${esc(g.quantity)}`:'—')}</div><div class="demo-sub-head">Conditions</div><div class="cond-list">${conds}</div>${kv('Reason',orDash(d?.reason))}<div class="data-tag">Premium &amp; underlying candles: ${esc(s.data_labels?.premium||'DEMO DATA')}</div>`;
}

function optRiskCard(s){
  const r=s.risk;
  if(!r)return `<div class="eyebrow">OPTION RISK</div><div class="demo-empty">${s.running?'Waiting for risk validation…':'Risk is validated after the contract is resolved.'}</div>`;
  const ok=r.status==='ALLOWED';
  return `<div class="eyebrow">OPTION RISK · BUY ONLY</div><div class="demo-kvs">${kv('Demo Equity',money(r.equity))}${kv('Risk / Trade',pct(r.limits?.risk_per_trade_percent))}${kv('Risk / Unit',money(r.risk_per_unit))}${kv('Max Loss',money(r.max_loss))}${kv('Premium Exposure',money(r.premium_exposure))}${kv('Max Premium Exp.',pct(r.limits?.max_premium_exposure_pct,1))}${kv('Daily Loss Limit',money(r.daily_loss_limit))}${kv('Max Option Positions',orDash(r.limits?.max_option_positions))}${kv('Sizing Limit',orDash(r.sizing?.limiting_factor))}${kv('Lot Size',orDash(r.lot_size))}</div>${kv('Validation',esc(r.status),ok?'positive':'negative')}${ok?'':kv('Reason',orDash(r.reason),'negative')}`;
}

function optOrderCard(s){
  const o=s.order;const b=s.broker?.order;const st=String(o?.status||'').toUpperCase();
  const cls=st==='CANCELLED'?'positive':st==='REJECTED'||st==='FAILED'?'negative':'';
  return `<div class="eyebrow">NSE_FO SANDBOX ORDER</div><div class="demo-kvs">${kv('Broker',esc(s.broker?.name||'UPSTOX SANDBOX'))}${kv('Connection',esc(s.broker?.status||'IDLE'))}${kv('Instrument Key',orDash(o?.symbol||s.instrument_key))}${kv('Side',orDash(o?.side),'side-long')}${kv('Quantity',orDash(o?.requested_quantity))}${kv('Order Type',orDash(o?.order_type))}${kv('Status',o?esc(st):'—',cls)}${kv('Broker Verified',orDash(b?.status))}${kv('Filled',o?esc(o.filled_quantity??0):'—')}</div><div class="demo-sub-head">Sandbox Order ID</div><div class="mono-id order-id" id="optOrderId">${o?esc(o.order_id):'—'}</div>`;
}

function renderOption(s){
  if(!s)return;opt.last=s;
  const stage=s.stage||'IDLE';const running=!!s.running;const done=stage==='COMPLETED';const failed=stage==='FAILED';
  $('#optionConsole').classList.toggle('is-idle',stage==='IDLE'&&!running);
  const pre=s.preconditions||{};const unsafe=pre.live_trading_enabled||(pre.trading_mode&&pre.trading_mode!=='PAPER')||pre.options_enabled===false;
  const chip=$('#optState');chip.textContent=running?'OPTION DEMO RUNNING':(s.headline||'SYSTEM READY');chip.className=`demo-state ${running?'run':done?'ok':failed?'bad':''}`;
  $('#optDemoBadge').classList.toggle('hidden',!running);
  const start=$('#optStartBtn');start.disabled=running||unsafe;start.textContent=running?'● OPTION DEMO RUNNING':'▶ START OPTION DEMO';
  $('#optResetBtn').disabled=running||stage==='IDLE';
  $('#optProgressBar').style.width=`${Number(s.progress||0)}%`;$('#optProgressBar').className=failed?'bad':done?'ok':'';
  $('#optMessage').textContent=`${stage}${s.demo_run_id?` · ${s.demo_run_id}`:''} — ${s.message||''}`;
  const alert=$('#optAlert');
  if(failed&&s.error){alert.innerHTML=`<strong>${esc(s.error.headline)}</strong><div>Blocked stage: ${esc(s.error.stage)} · Code: ${esc(s.error.code)}</div><div>Reason: ${esc(s.error.message)}</div><div>Option resolution: ${esc(s.option_resolution?.status)} · Sandbox execution: ${esc(s.sandbox_execution?.status)}</div>${s.error.cancel_required?'<div><b>Manual check required: the Sandbox order may still be open.</b></div>':''}`;alert.className='demo-alert bad'}
  else if(done){alert.innerHTML=`<strong>OPTION DEMO COMPLETED</strong><div>${esc(s.trading_symbol)} · order ${esc(s.order?.order_id)} submitted, verified and ${esc(String(s.order?.status||'').toLowerCase())} in the Upstox Sandbox.</div>`;alert.className='demo-alert ok'}
  else if(unsafe){alert.innerHTML=`<strong>OPTION DEMO UNAVAILABLE</strong><div>${pre.options_enabled===false?'options.enabled is false':'Live trading must be disabled and TRADING_MODE must be PAPER'}</div>`;alert.className='demo-alert bad'}
  else if(!running){alert.className='demo-alert hidden'}
  $('#optTimeline').innerHTML=timelineHtml(s.timeline);
  $('#optSignalCard').innerHTML=optSignalCard(s);$('#optRiskCard').innerHTML=optRiskCard(s);$('#optOrderCard').innerHTML=optOrderCard(s);
  $('#optResolution').textContent=s.option_resolution?.status||'IDLE';
  $('#optExecution').textContent=s.sandbox_execution?.status||'IDLE';
  $('#optData').textContent=s.data_labels?.underlying||'DEMO DATA';
  $('#optLive').textContent=s.live_trading_enabled?'ENABLED':'DISABLED';
  if(s.notice)$('#optNotice').textContent=s.notice;
}

async function optFinished(s){
  if(!s.demo_run_id||opt.refreshedFor===s.demo_run_id)return;
  opt.refreshedFor=s.demo_run_id;await refresh();renderOption(s);
}

function pollOption(){
  clearTimeout(opt.timer);
  const tick=async()=>{try{const s=await get('/api/options/demo/status');renderOption(s);if(s.running){opt.timer=setTimeout(tick,1000)}else{await optFinished(s)}}catch(e){opt.timer=setTimeout(tick,2500)}};
  tick();
}

async function optControl(path,body,promptText){
  let k=apiKey();
  if(!k){k=prompt(promptText)||'';if(!k)return null;sessionStorage.setItem('dashboardApiKey',k)}
  const r=await fetch(API_BASE+path,{method:'POST',headers:{'Content-Type':'application/json','X-API-Key':k},body:body?JSON.stringify(body):undefined});
  let data={};try{data=await r.json()}catch(e){}
  if(r.status===401)sessionStorage.removeItem('dashboardApiKey');
  return {ok:r.ok,status:r.status,body:data};
}

async function startOptionDemo(){
  const btn=$('#optStartBtn');if(btn.disabled)return;
  const p=demoParams();
  if(!p.underlying){optAlert('OPTION DEMO NOT STARTED','Select an underlying first');return}
  btn.disabled=true;
  try{
    const r=await optControl('/api/options/demo/start',p,'Dashboard API key (required to start the option Sandbox demo)');
    if(!r){btn.disabled=false;return}
    if(!r.ok){optAlert(r.status===409?'A DEMO IS ALREADY RUNNING':'OPTION DEMO NOT STARTED',errText(r.body)||`HTTP ${r.status}`);btn.disabled=false;if(r.status===409)pollOption();return}
    opt.refreshedFor=null;renderOption(r.body.status);pollOption();
  }catch(e){optAlert('OPTION DEMO NOT STARTED','API unavailable');btn.disabled=false}
}

async function resetOptionDemo(){
  const r=await optControl('/api/options/demo/reset',null,'Dashboard API key (required to reset the option console)');
  if(!r)return;
  if(!r.ok){optAlert('RESET NOT ALLOWED',errText(r.body)||`HTTP ${r.status}`);return}
  opt.refreshedFor=null;
  const s=await get('/api/options/demo/status').catch(()=>r.body);renderOption(s);
}

(function initOptions(){
  $$('.trade-tab').forEach(b=>b.addEventListener('click',()=>setTradeType(b.dataset.tradeType)));
  $('#optUnderlying').addEventListener('change',loadExpiries);
  $('#optExpiry').addEventListener('change',loadChain);
  ['#optType','#optStrikeMode','#optOffset','#optStrike'].forEach(id=>$(id).addEventListener('change',resolveContract));
  $('#optChainBody').addEventListener('click',pickFromChain);
  $('#optStartBtn').addEventListener('click',startOptionDemo);
  $('#optResetBtn').addEventListener('click',resetOptionDemo);
  const fromHash={'#index-options':'INDEX_OPTION','#stock-options':'STOCK_OPTION'}[location.hash];
  if(fromHash)setTradeType(fromHash);
  get('/api/options/demo/status').then(s=>{opt.refreshedFor=s.running?null:s.demo_run_id;renderOption(s);if(s.running){setTradeType(s.market_type||s.request?.market_type||'INDEX_OPTION');pollOption()}}).catch(()=>renderOption({stage:'IDLE',timeline:[],message:'Option demo API unavailable'}));
})();
