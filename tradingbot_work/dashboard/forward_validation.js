async function refreshForwardValidation(){
  const s=await get('/api/forward-validation/stats').catch(()=>null);
  if(!s||!$('#forwardValidationPanel'))return;
  $('#fvToday').textContent=s.today?s.today.signals:0;
  $('#fvResolved').textContent=s.resolved;
  $('#fvWins').textContent=s.wins;
  $('#fvLosses').textContent=s.losses;
  $('#fvAccuracy').textContent=s.resolved?`${Number(s.accuracy).toFixed(2)}%${s.accuracy_is_strategy_accuracy?'':' (observed)'}`:'N/A';
  $('#fvPf').textContent=s.profit_factor===null?(s.resolved&&s.gross_profit>0?'No losses':'N/A'):Number(s.profit_factor).toFixed(2);
  $('#fvNet').textContent=money(s.net_pnl);
  $('#fvDd').textContent=money(s.max_drawdown);
  $('#fvSample').textContent=s.sample_status;
  $('#fvDays').textContent=s.days_tracked;
  $('#fvSampleBadge').textContent=s.live_trading_enabled?'WARNING: LIVE FLAG ON · NO ORDERS PLACED':s.sample_status;
  $('#fvSampleBadge').className=`badge ${s.live_trading_enabled?'bad':s.resolved>=200?'good':'warn'}`;
  $('#fvHistorical').textContent=s.historical_comparable_1dte;
  $('#fvNote').textContent=`${s.accuracy_label}. Pending ${s.pending}, ambiguous ${s.ambiguous}, invalid ${s.invalid}, data unavailable ${s.data_unavailable} excluded. P&L is gross premium move × 1 lot (costs excluded). Sessions ${s.first_session||'—'} to ${s.last_session||'—'}.`;
}
refreshForwardValidation();
setInterval(refreshForwardValidation,5000);
