// ---------------------------------------------------------------------------
// Presentation of persisted demo runs (Overview card, Signals, Positions
// context, Analytics). Every value comes from /api/demo/overview, which is
// built from stored demo results and database records only.
// ---------------------------------------------------------------------------
const MARKET_LABEL={EQUITY:'EQUITY',INDEX_OPTION:'INDEX OPTION',STOCK_OPTION:'STOCK OPTION'};
const marketLabel=m=>MARKET_LABEL[m]||String(m||'—').replace(/_/g,' ');
const marketPill=m=>`<span class="market-pill ${m&&m!=='EQUITY'?'opt':''}">${esc(marketLabel(m))}</span>`;
const pill=(text,cls='')=>`<span class="status-pill ${cls}">${esc(text)}</span>`;
const priceOrDash=v=>v==null||v===''?'—':money(v);
const condLabel=k=>(typeof CONDITIONS!=='undefined'&&CONDITIONS[k])||String(k).replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());

function runQuantity(r){if(r.quantity==null)return '—';const q=Number(r.quantity).toLocaleString('en-IN');return r.lots&&r.lot_size?`${q} <small class="muted-text">(${esc(r.lots)} lot${r.lots>1?'s':''} × ${esc(r.lot_size)})</small>`:esc(q)}
function signalClass(s){const t=String(s||'').toUpperCase();return t.includes('PUT')||t==='SELL'?'side-short':t?'side-long':''}
function runResult(r){if(r.execution_result==='SUCCESS')return pill('SUCCESS','st-filled');if(r.execution_result)return pill(r.execution_result,'st-bad');return pill(r.stage||'—')}
function runStatus(r,running){if(running)return {text:'RUNNING',cls:'run'};const s=String(r.stage||'').toUpperCase();return {text:s||'—',cls:s==='COMPLETED'?'ok':s==='FAILED'?'bad':''}}

function flowHtml(steps){const icon={done:'✓',failed:'✕',pending:'',active:''};return `<div class="flow">${(steps||[]).map(s=>`<div class="flow-step ${esc(s.state)} ${s.key==='FINAL'?'final':''}" title="${esc(s.detail||'')}"><span class="flow-icon">${icon[s.state]??''}</span><div class="flow-text"><strong>${esc(s.key==='FINAL'&&s.detail?s.detail:s.label)}</strong><span>${s.key==='FINAL'?esc(s.label):s.detail?esc(s.detail):s.state==='pending'?'Not reached':''}</span></div></div>`).join('<span class="flow-link"></span>')}</div>`}


// ---------------- Overview: Latest Demo Run ----------------
function latestDemoCard(ov){const card=$('#latestDemoCard');const r=ov?.latest;
if(!r){card.innerHTML=`<div class="ld-empty"><div><span class="eyebrow">LATEST DEMO RUN</span><h2>No demo run yet</h2><p>Run an Equity, Index Option or Stock Option demo; the result will appear here.</p></div><button class="btn demo-start" data-view="demo">▶ Open Demo</button></div>`;return}
const running=ov.running&&(ov.running.equity||ov.running.option);const st=runStatus(r,running);
const steps=(r.timeline||[]).filter(s=>s.key!=='FINAL');const done=steps.filter(s=>s.state==='done').length;const failedStep=steps.find(s=>s.state==='failed');
const order=r.order_id?`${esc(r.side||'')} ${esc(r.quantity??'')} · ${esc(r.order_type||'')}`:'—';
const paper=ov.safety?.trading_mode==='PAPER'&&!ov.safety?.live_trading_enabled;
card.innerHTML=`<div class="ld-head"><div><span class="eyebrow">LATEST DEMO RUN</span><h2>${esc(r.contract||r.underlying||'—')} ${marketPill(r.market_type)}</h2><span class="ld-sub">${esc(fmtTime(r.finished_at||r.started_at))}${r.instrument_key?` · <span class="mono">${esc(r.instrument_key)}</span>`:''}${running?' · a new demo is running…':''}</span></div><div class="ld-badges"><span class="demo-state ${st.cls}">${esc(st.text)}</span><button class="btn ghost" data-view="demo">Open Demo →</button></div></div>
<div class="ld-grid">
${ldItem('Signal',esc(r.signal||'—'),signalClass(r.signal))}
${ldItem('Order',order+(r.lots&&r.lot_size?`<small>${esc(r.lots)} lots × ${esc(r.lot_size)}</small>`:''))}
${ldItem('Order ID',`<span class="mono">${orDash(r.order_id)}</span>`)}
${ldItem('Broker',esc(r.broker||'UPSTOX SANDBOX'))}
${ldItem('Final Status',r.final_status?pill(r.final_status,statusCls(r.final_status)):'—')}
${ldItem('Execution',r.execution_result?esc(r.execution_result):'—',r.execution_result==='SUCCESS'?'positive':'negative')}
</div>
<div class="ld-foot"><div class="mini-steps">${steps.map(s=>`<span class="${esc(s.state)}" title="${esc(s.label)}${s.detail?' · '+esc(s.detail):''}"></span>`).join('')}</div><span class="ld-steps-text">${failedStep?`Stopped at ${esc(failedStep.label)}`:`${done} / ${steps.length} steps completed`}</span><button class="link-btn" data-view="signals">View timeline →</button><span class="ld-safety ${paper?'positive':'negative'}">${paper?'PAPER TRADING · LIVE EXECUTION LOCKED':'CHECK SAFETY SETTINGS'}</span></div>
${r.error?`<div class="demo-alert bad"><strong>${esc(r.error.headline||'DEMO FAILED')}</strong><div>${esc(r.error.message||'')}</div></div>`:''}`}
function ldItem(label,value,cls=''){return `<div class="ld-item"><span>${esc(label)}</span><strong class="${cls}">${value}</strong></div>`}

// ---------------- Signals ----------------
function signalsPage(ov){const r=ov?.latest;$('#signalsEmpty').classList.toggle('hidden',!!r);const box=$('#signalsContent');if(!r){box.innerHTML='';return}
const opt=r.demo_type==='OPTION';
const rows=[kv('Market',esc(marketLabel(r.market_type))),kv('Underlying',orDash(r.underlying)),kv('Instrument',`<span class="mono">${esc(opt?r.contract||'—':r.instrument_key||'—')}</span>`),kv('Signal',esc(r.signal||'—'),signalClass(r.signal)),kv(opt?'Entry (premium)':'Entry',priceOrDash(r.entry)),kv('Stop Loss',priceOrDash(r.stop_loss),'negative'),kv('Target',priceOrDash(r.target),'positive'),kv('Quantity',runQuantity(r))].join('');
const optRows=opt?[kv('Expiry',orDash(r.expiry)),kv('Strike',r.strike==null?'—':esc(Number(r.strike).toLocaleString('en-IN'))),kv('CE / PE',r.option_type?`<span class="type-tag ${esc(String(r.option_type).toLowerCase())}">${esc(r.option_type)}</span>`:'—'),kv('Lot Size',orDash(r.lot_size)),kv('Instrument Key',`<span class="mono">${orDash(r.instrument_key)}</span>`)].join(''):'';
const conds=r.conditions&&Object.keys(r.conditions).length?`<div class="cond-list">${Object.entries(r.conditions).map(([k,v])=>`<div class="cond ${v===true?'ok':v===false?'bad':''}"><span>${esc(condLabel(k))}</span><b>${v===true?'✓':v===false?'✕':'—'}</b></div>`).join('')}</div>`:`<div class="demo-empty">Strategy conditions were not stored for this run. Runs started after this update record them.</div>`;
const prev=(ov.runs||[]).slice(0,6);
box.innerHTML=`<div class="page-card sig-card"><div class="page-card-head"><div><span class="eyebrow">STRATEGY OUTPUT</span><h2>Latest Demo Signal</h2></div><div class="ld-badges">${marketPill(r.market_type)}<span class="badge demo-badge">DEMO / PAPER</span></div></div>
<div class="sig-grid"><div class="sig-col"><div class="demo-sub-head">Signal</div><div class="demo-kvs">${rows}</div>${opt?`<div class="demo-sub-head">Option contract</div><div class="demo-kvs">${optRows}</div>`:''}</div>
<div class="sig-col"><div class="demo-sub-head">Strategy conditions used</div>${conds}<div class="demo-kvs">${kv('Risk Status',esc(r.risk_status||'—'),r.risk_status==='ALLOWED'?'positive':'negative')}${kv('Reason',orDash(r.signal_reason||r.risk_reason))}${kv('Signal ID',`<span class="mono">${esc(String(r.signal_id||'—').slice(0,18))}</span>`)}${kv('Generated',esc(fmtTime(r.started_at)))}</div><div class="data-note">Prices come from ${esc(r.data_label||'DEMO DATA')} candles${opt&&r.spot!=null?` (reference spot ${esc(r.spot)})`:''}; the Upstox Sandbox has no live market data.</div></div></div></div>
<div class="page-card"><div class="page-card-head"><div><span class="eyebrow">EXECUTION</span><h2>Demo Execution Timeline</h2></div>${r.execution_result==='SUCCESS'?'<span class="badge good">DEMO COMPLETED</span>':`<span class="badge bad">${esc(r.stage||'')}</span>`}</div>${flowHtml(r.timeline)}</div>
${prev.length>1?`<div class="page-card"><div class="page-card-head"><div><span class="eyebrow">HISTORY</span><h2>Recent Demo Signals</h2></div></div><div class="table-scroll"><table><thead><tr><th>Time</th><th>Market</th><th>Contract</th><th>Signal</th><th>Entry</th><th>Qty</th><th>Order ID</th><th>Result</th></tr></thead><tbody>${prev.map(runRow).join('')}</tbody></table></div></div>`:''}`}
function runRow(r){return `<tr><td class="nowrap">${esc(fmtTime(r.finished_at||r.started_at))}</td><td>${marketPill(r.market_type)}</td><td class="mono">${esc(r.contract||r.underlying||'—')}</td><td class="${signalClass(r.signal)}">${esc(r.signal||'—')}</td><td>${priceOrDash(r.entry)}</td><td>${r.quantity??'—'}</td><td class="mono">${orDash(r.order_id)}</td><td>${r.final_status?pill(r.final_status,statusCls(r.final_status)):''} ${runResult(r)}</td></tr>`}

// ---------------- Positions context ----------------
function positionsContext(ov){const box=$('#positionsContext');const r=ov?.latest;if(state.data.positions.length||!r){box.innerHTML='';return}
const ok=r.final_status==='CANCELLED';
box.innerHTML=`<div class="context-card"><div class="cc-icon">◈</div><div class="cc-body"><strong>NO OPEN POSITIONS</strong><p>${ok?'Demo orders completed successfully and were cancelled in Sandbox, so an empty position book is expected.':`The latest demo did not open a position (${esc(r.error?.headline||r.stage||'not completed')}).`}</p>
<div class="cc-last"><span class="eyebrow">LAST DEMO</span><div class="cc-row"><span class="mono">${esc(r.contract||r.underlying||'—')}</span><span class="${signalClass(r.signal)}">${esc(r.signal||'—')}</span><span>Qty ${esc(r.quantity??'—')}</span><span class="mono">${orDash(r.order_id)}</span>${r.final_status?pill(r.final_status,statusCls(r.final_status)):''}</div></div></div></div>`}

// ---------------- Analytics ----------------
function analyticsPage(ov){const s=ov?.stats;const box=$('#analyticsContent');const has=s&&(s.demo_runs||s.total_trades||s.total_orders);$('#analyticsEmpty').classList.toggle('hidden',!!has);if(!has){box.innerHTML='';return}
const latest=ov.latest;const tile=(label,value,foot)=>`<article class="kpi"><div class="kpi-head"><span>${esc(label)}</span></div><div class="kpi-value">${value}</div><div class="kpi-foot"><span class="neutral">${foot}</span></div></article>`;
const byMarket={};(ov.runs||[]).forEach(r=>{const k=r.market_type||'—';byMarket[k]=byMarket[k]||{runs:0,ok:0};byMarket[k].runs++;if(r.execution_result==='SUCCESS')byMarket[k].ok++});
const maxRuns=Math.max(1,...Object.values(byMarket).map(v=>v.runs));
const pnl=s.realized_pnl==null?`<div class="pnl-na"><strong>P&amp;L: N/A</strong><span>${esc(String(s.pnl_note||'').replace(/^N\/A\s*—\s*/,''))}</span></div>`:`<div class="pnl-value ${s.realized_pnl>=0?'positive':'negative'}">${s.realized_pnl>=0?'+':''}${money(s.realized_pnl)}</div><span class="muted-text">Realized from ${esc(s.closed_trades)} closed trade${s.closed_trades===1?'':'s'}</span>`;
box.innerHTML=`<div class="analytics-head"><div><span class="eyebrow">DEMO / SANDBOX</span><h2>Demo Analytics</h2><p class="muted-text">Counts come from persisted demo results and database records. No profitability is assumed.</p></div><span class="badge demo-badge">DEMO / SANDBOX</span></div>
<div class="kpi-grid analytics-kpis">
${tile('Demo Runs',esc(s.demo_runs),`${esc(s.failed_demo_runs)} failed`)}
${tile('Completed Demo Orders',esc(s.completed_demo_runs),'submitted · verified · cancelled')}
${tile('Cancelled Orders',esc(s.cancelled_orders),`of ${esc(s.total_orders)} recorded orders`)}
${tile('Open Positions',esc(s.open_positions),s.open_positions?'paper positions':'none, as expected')}
${tile('Total Demo Trades',esc(s.total_trades),s.total_trades?`${esc(s.closed_trades)} closed`:'no fills, so no trades')}
${tile('Latest Signal',`<span class="${signalClass(latest?.signal)}">${esc(latest?.signal||'—')}</span>`,esc(latest?.contract||'—'))}
</div>
<div class="lower-grid analytics-grid"><article class="panel"><div class="panel-head"><div><span class="eyebrow">PERFORMANCE</span><h2>Profit &amp; Loss</h2></div></div>${pnl}<p class="perf-note">${esc(s.performance_note||'')}</p></article>
<article class="panel"><div class="panel-head"><div><span class="eyebrow">COVERAGE</span><h2>Demo Runs by Market</h2></div></div><div class="bars">${Object.entries(byMarket).map(([k,v])=>`<div class="bar-row"><span>${esc(marketLabel(k))}</span><div class="bar"><span style="width:${(v.runs/maxRuns*100).toFixed(1)}%"></span></div><strong>${v.ok}/${v.runs}</strong></div>`).join('')||'<div class="demo-empty">No demo runs yet.</div>'}</div><p class="perf-note">Successful / total runs per market.</p></article></div>
${(ov.runs||[]).length?`<div class="page-card"><div class="page-card-head"><div><span class="eyebrow">HISTORY</span><h2>Demo Run History</h2></div><span class="badge">${esc((ov.runs||[]).length)} most recent</span></div><div class="table-scroll"><table><thead><tr><th>Time</th><th>Market</th><th>Contract</th><th>Signal</th><th>Entry</th><th>Qty</th><th>Order ID</th><th>Result</th></tr></thead><tbody>${ov.runs.map(runRow).join('')}</tbody></table></div></div>`:''}`}

function renderPresentation(ov,status){latestDemoCard(ov);signalsPage(ov);positionsContext(ov);analyticsPage(ov);const s=ov?.stats;if(s){$('#pnlCaption').textContent=s.closed_trades?`${s.closed_trades} closed trades`:'No filled trades yet';$('#ordersBadge').textContent=s.demo_orders?`${s.demo_orders} DEMO ORDERS · DATABASE`:'DATABASE RECORDS'}}
