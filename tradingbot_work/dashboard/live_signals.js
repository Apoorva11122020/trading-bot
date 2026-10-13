async function refreshLiveSignals(){
  const [stats,list]=await Promise.all([get('/api/signals/live/stats').catch(()=>null),get('/api/signals/live').catch(()=>null)]);
  if(!stats||!$('#liveSignalPanel'))return;
  const num=v=>v===null||v===undefined?'—':Number(v).toLocaleString('en-IN',{maximumFractionDigits:2});
  $('#lsTotal').textContent=stats.total;
  $('#lsResolved').textContent=stats.resolved;
  $('#lsPending').textContent=stats.pending;
  $('#lsWins').textContent=stats.wins;
  $('#lsLosses').textContent=stats.losses;
  $('#lsAccuracy').textContent=stats.resolved?`${Number(stats.accuracy).toFixed(2)}%`:'N/A';
  $('#lsPf').textContent=stats.profit_factor===null?(stats.resolved&&stats.gross_profit>0?'No losses':'N/A'):Number(stats.profit_factor).toFixed(2);
  $('#lsNet').textContent=money(stats.net_pnl);
  $('#lsDd').textContent=money(stats.max_drawdown);
  $('#liveSignalBadge').textContent=stats.live_trading_enabled?'WARNING: LIVE FLAG ON · TRACKER PLACES NO ORDERS':'NO ORDERS · MEASURED ONLY';
  $('#liveSignalBadge').className=`badge ${stats.live_trading_enabled?'bad':'good'}`;
  const rows=(list&&list.signals)||[];
  const cls=s=>s==='WIN'?'positive':s==='LOSS'?'negative':'neutral';
  $('#liveSignalBody').innerHTML=rows.length?rows.map(r=>`<tr><td>${esc(fmtTime(r.time))}</td><td>${esc(r.direction)}</td><td>${esc(r.contract||'—')}</td><td>${num(r.entry_price)}</td><td>${num(r.stop_loss)}</td><td>${num(r.target)}</td><td class="${cls(r.status)}" title="${esc(r.reason||'')}">${esc(r.status)}</td><td class="${r.pnl>0?'positive':r.pnl<0?'negative':''}">${r.pnl===null||r.pnl===undefined?'—':money(r.pnl)}</td><td>${num(r.r_multiple)}</td></tr>`).join(''):'<tr><td colspan="9" class="empty-row">No signals recorded today</td></tr>';
  $('#liveSignalNote').textContent=`${stats.date} · Accuracy = wins / (wins + losses). Pending ${stats.pending}, ambiguous ${stats.ambiguous}, invalid ${stats.invalid} excluded. P&L is gross premium move × recorded quantity (costs excluded).`;
}
refreshLiveSignals();
setInterval(refreshLiveSignals,5000);
