// Gedeelde statusbalk bovenaan /desk en /swing: Swarm en Swing naast elkaar.
// Per desk: portefeuillewaarde (en winst sinds start), open trades, W/V van de lopende handelsdag
// (22.00 tot 22.00) en hoe lang de laatste analyse geleden is. Eén bestand, zodat beide bladen hetzelfde tonen.
(() => {
  if (document.getElementById('sbar')) return;
  const DESKS = [
    { key: 'swarm', name: 'Swarm', href: '/desk', url: '/api/desk/summary', what: 'coach' },
    { key: 'swing', name: 'Swing', href: '/swing', url: '/api/swing/summary', what: 'analyse' },
    { key: 'trading', name: 'Trading Desk', href: '/trading', url: '/api/trading/summary', what: 'review' },
  ];
  const here = location.pathname.replace(/\/+$/, '') || '/';
  const css = `
  #sbar { position: sticky; top: 0; z-index: 1000; display: flex; gap: 1px; background: #1b2140;
    font: 12px/1.3 ui-monospace, "SF Mono", "JetBrains Mono", Consolas, monospace; color: #e6ebf8; }
  #sbar a.d { flex: 1; display: flex; align-items: center; gap: 3px 12px; flex-wrap: wrap; padding: 5px 12px; background: #0a0d19;
    color: inherit; text-decoration: none; min-width: 0; white-space: nowrap; }
  #sbar a.d:hover { background: #111528; }
  #sbar a.d.here { box-shadow: inset 0 -2px 0 #5b8cff; flex: 1.6; }
  #sbar .nm { font: 700 11px system-ui, sans-serif; letter-spacing: .08em; text-transform: uppercase; }
  #sbar .nm i { display: inline-block; vertical-align: middle; line-height: 1.25; font-style: normal; font-weight: 600; font-size: 9px; color: #7581a3; margin-left: 5px; }
  #sbar .k { display: flex; flex-direction: column; }
  #sbar .k span { font: 600 8px system-ui, sans-serif; letter-spacing: .1em; text-transform: uppercase; color: #7581a3; }
  #sbar .k b { font-weight: 600; font-variant-numeric: tabular-nums; }
  #sbar .up { color: #16b364; } #sbar .dn { color: #ef3a5d; } #sbar .mu { color: #7581a3; } #sbar .wa { color: #f08c00; }
  @media (max-width: 900px) { #sbar { flex-direction: column; } #sbar a.d { gap: 12px; } }`;
  const st = document.createElement('style');
  st.textContent = css;
  document.head.appendChild(st);
  const bar = document.createElement('div');
  bar.id = 'sbar';
  bar.innerHTML = DESKS.map(d => `<a class="d${here === d.href ? ' here' : ''}" href="${d.href}" id="sbar-${d.key}">
      <span class="nm">${d.name}<i></i></span><span class="mu">laden…</span></a>`).join('');
  document.body.prepend(bar);

  const usd = v => (v < 0 ? '−$' : '$') + Math.abs(v).toLocaleString('nl-BE', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const sgn = v => (v > 0 ? '+' : v < 0 ? '−' : '') + '$' + Math.abs(v).toLocaleString('nl-BE', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const cls = v => (v > 0 ? 'up' : v < 0 ? 'dn' : 'mu');
  const ago = ms => {
    if (!ms) return 'nog nooit';
    const m = Math.max(0, Math.round((Date.now() - ms) / 60000));
    if (m < 1) return 'net nu';
    if (m < 60) return m + ' min';
    const h = Math.floor(m / 60);
    if (h < 24) return h + 'u' + String(m % 60).padStart(2, '0');
    return Math.floor(h / 24) + 'd ' + (h % 24) + 'u';
  };
  const clock = ms => ms ? new Date(ms).toLocaleString('nl-BE', { weekday: 'short', hour: '2-digit', minute: '2-digit' }) : '';
  const k = (label, val, title = '') => `<span class="k"${title ? ` title="${title}"` : ''}><span>${label}</span><b>${val}</b></span>`;

  // Winrate alleen zegt niets: 40% winst met winnaars van 3× de verliezers is goed, 70% met winnaars van ⅓ is verlies.
  // Daarom naast elkaar: winrate, gem. winnaar, gem. verliezer, en de winrate die nodig is om quitte te spelen.
  const pc = v => Math.round(v * 100) + '%';
  function edgeK(a) {
    if (!a || !a.n) return k('winrate · gem. W / V', '<span class="mu">nog geen trades</span>');
    const ok = a.be_winrate == null ? 'mu' : a.winrate > a.be_winrate ? 'up' : 'dn';
    return k('winrate', `<span class="${ok}">${pc(a.winrate)}</span> <span class="mu">nodig ${a.be_winrate == null ? '–' : pc(a.be_winrate)}</span>`,
        `${a.wins} winnaars, ${a.losses} verliezers sinds de start (${a.n} trades).\nNodig = gem. verlies ÷ (gem. winst + gem. verlies): daarboven verdien je geld.`)
      + k('gem. winnaar / verliezer', `<span class="up">${sgn(a.avg_win)}</span> / <span class="dn">${sgn(-a.avg_loss)}</span>`,
          'gemiddelde winst per winnende trade / gemiddeld verlies per verliezende trade, na kosten')
      + k('per trade', `<span class="${cls(a.expectancy)}">${sgn(a.expectancy)}</span>`,
          `verwachting per trade = winrate × gem. winnaar − (1 − winrate) × gem. verliezer\nverhouding winnaar/verliezer ${a.avg_loss ? (a.avg_win / a.avg_loss).toFixed(2).replace('.', ',') : '–'}`);
  }

  function edgeShort(a) {  // op de andere bladen: alleen winrate tegen nodig, zodat drie desks naast elkaar passen
    if (!a || !a.n) return '';
    const ok = a.be_winrate == null ? 'mu' : a.winrate > a.be_winrate ? 'up' : 'dn';
    return k('winrate', `<span class="${ok}">${pc(a.winrate)}</span><span class="mu">/${a.be_winrate == null ? '–' : pc(a.be_winrate)}</span>`,
      `winrate / nodige winrate · gem. winnaar ${sgn(a.avg_win)}, gem. verliezer ${sgn(-a.avg_loss)}, per trade ${sgn(a.expectancy)}`);
  }

  function render(d, s) {
    const el = document.getElementById('sbar-' + d.key);
    if (!s) {
      el.innerHTML = `<span class="nm">${d.name}</span><span class="mu">niet bereikbaar</span>`;
      return;
    }
    const t = s.today || { wins: 0, losses: 0, pnl: 0 };
    const pct = s.start ? (s.pnl / s.start) * 100 : 0;
    const days = (s.days || []).slice(-7).map(r => `${r.day.slice(5)}: ${r.wins}W/${r.losses}V ${sgn(r.pnl)}`).join('\n');
    const open = s.open + (s.pending ? ` <span class="mu">+${s.pending} wacht</span>` : '');
    el.innerHTML = `<span class="nm">${d.name}<i>${s.mode === 'live' ? 'LIVE' : 'papier'}${s.on === false ? ' · uit' : ''}<br><span title="${'laatste ' + d.what + ': ' + clock(s.analysis_at)}">${d.what} ${s.analysis_running ? '<span class="wa">bezig…</span>' : ago(s.analysis_at)}</span></i></span>`
      + k('waarde', `${usd(s.equity)} <span class="${cls(s.pnl)}">${pct >= 0 ? '+' : '−'}${Math.abs(pct).toFixed(1).replace('.', ',')}%</span>`,
          `start ${usd(s.start)}, winst sinds start ${sgn(s.pnl)}, waarvan open ${sgn(s.upnl || 0)}`)
      + k('open', open, `niet-gerealiseerd ${sgn(s.upnl || 0)}`)
      + (here !== d.href && here !== '/' ? k('W/V vandaag', `${t.wins}/${t.losses} <span class="${cls(t.pnl)}">${sgn(t.pnl)}</span>`, 'handelsdag 22.00 tot 22.00') : k('W/V vandaag', `${t.wins}/${t.losses}${t.wins + t.losses ? ` · ${Math.round(t.wins / (t.wins + t.losses) * 100)}%` : ''} `
          + `<span class="${cls(t.pnl)}">${sgn(t.pnl)}${s.start ? ` · ${t.pnl >= 0 ? '+' : '−'}${Math.abs(t.pnl / s.start * 100).toFixed(1).replace('.', ',')}%` : ''}</span>`,
          'handelsdag 22.00 tot 22.00\n' + (days || 'nog geen afgesloten trades')))
      + (here === d.href || here === '/' ? edgeK(s.all) : edgeShort(s.all))
  }

  async function tick() {
    await Promise.all(DESKS.map(async d => {
      try {
        const r = await fetch(d.url, { cache: 'no-store' });
        render(d, r.ok ? await r.json() : null);
      } catch { render(d, null); }
    }));
  }
  tick();
  setInterval(tick, 15000);
})();
