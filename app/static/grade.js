// Ticker-grade (A+ tot F) overal op de backtest server, en op elke andere pagina die dit script laadt
// (bv. de portfolio server: <script src="http://192.168.1.202:9000/static/grade.js"></script>).
//
// Gebruik in een pagina: zet <span data-gsym="AAPL"></span> waar de grade moet komen (of Grade.badge('AAPL')).
// Het script haalt alle grades één keer op (/api/grades), vult elke [data-gsym] in, ook als de pagina later
// opnieuw tekent, en toont bij een klik de details: welke strategieën werken, welke nu open staan, hun cijfers,
// de formule en hoe goed de grade historisch voorspelde (walk-forward check).
(() => {
  if (window.Grade) return;
  const me = document.currentScript;
  const BASE = (me && me.dataset.base) || (me && me.src ? new URL(me.src).origin : location.origin);
  const COLORS = { 'A+': '#0f9d58', A: '#22c55e', B: '#84cc16', C: '#eab308', D: '#f97316', F: '#ef4444' };
  const css = `
  .grd { display: inline-flex; align-items: center; justify-content: center; min-width: 24px; height: 18px; padding: 0 5px;
    border-radius: 4px; font: 700 11px/1 system-ui, -apple-system, "Segoe UI", sans-serif; color: #fff; cursor: pointer;
    vertical-align: middle; margin-right: 6px; letter-spacing: .02em; user-select: none; }
  .grd.none { background: transparent; color: #8b93a1; border: 1px dashed #8b93a1; font-weight: 500; }
  .grd:hover { filter: brightness(1.15); }
  .grd.gbig { font-size: 15px; height: 26px; min-width: 34px; margin: 0 8px; }
  #grd-pop { position: fixed; inset: 0; z-index: 3000; background: rgba(0,0,0,.55); display: flex; align-items: flex-start;
    justify-content: center; padding: 40px 12px; overflow: auto; }
  #grd-pop .box { background: #151820; color: #e6e8eb; border: 1px solid #262b36; border-radius: 10px; max-width: 980px;
    width: 100%; padding: 18px 20px; font: 13px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  #grd-pop h2 { margin: 0; font-size: 18px; display: flex; align-items: center; gap: 10px; }
  #grd-pop h3 { margin: 18px 0 6px; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: #8b93a1; }
  #grd-pop .big { font-size: 22px; height: 34px; min-width: 46px; border-radius: 6px; }
  #grd-pop .x { margin-left: auto; cursor: pointer; color: #8b93a1; font-size: 20px; background: none; border: 0; }
  #grd-pop table { border-collapse: collapse; width: 100%; font-size: 12px; }
  #grd-pop th, #grd-pop td { padding: 4px 6px; border-bottom: 1px solid #262b36; text-align: right; white-space: nowrap;
    background: transparent; color: inherit; position: static; }
  #grd-pop th { color: #8b93a1; font-weight: 600; font-size: 11px; text-transform: none; letter-spacing: 0; }
  #grd-pop th:first-child, #grd-pop td:first-child { text-align: left; }
  #grd-pop .up { color: #22c55e; } #grd-pop .dn { color: #ef4444; } #grd-pop .mu { color: #8b93a1; }
  #grd-pop .open { background: #3b82f6; color: #fff; border-radius: 3px; padding: 0 4px; font-size: 10px; font-weight: 700; }
  #grd-pop .bar { display: inline-block; height: 6px; background: #3b82f6; border-radius: 3px; vertical-align: middle; }
  #grd-pop .kpi { display: flex; gap: 22px; flex-wrap: wrap; margin-top: 10px; }
  #grd-pop .kpi > div > span { display: block; font-size: 10px; color: #8b93a1; text-transform: uppercase; letter-spacing: .08em; }
  #grd-pop .kpi div b { font-size: 15px; }
  #grd-pop ol { margin: 4px 0; padding-left: 20px; color: #c9ced6; }
  #grd-pop .verdict { padding: 8px 10px; border-radius: 6px; background: #1b1f29; border-left: 3px solid #f59e0b; margin: 6px 0; }
  #grd-pop button.run { padding: 5px 10px; border-radius: 5px; border: 1px solid #262b36; background: #1b1f29; color: #e6e8eb; cursor: pointer; }
  #grd-pop details summary { cursor: pointer; color: #8b93a1; margin-top: 12px; }`;
  const st = document.createElement('style');
  st.textContent = css;
  document.head.appendChild(st);

  let grades = null, aliases = {}, check = null, formula = [];
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (v, d = 1) => v == null ? '–' : (v > 0 ? '+' : '') + (v * 100).toFixed(d).replace('.', ',') + '%';
  const cl = v => v == null ? 'mu' : v > 0 ? 'up' : v < 0 ? 'dn' : '';
  const key = s => {
    if (!grades || !s) return null;
    if (grades[s]) return s;
    const u = String(s).toUpperCase();
    return aliases[u] || (u.includes(':') ? aliases[u.split(':')[1]] : null) || null;
  };
  const get = s => { const k = key(s); return k ? grades[k] : null; };

  function fill(el) {
    const s = el.dataset.gsym, g = get(s);
    const txt = g ? g.grade : '–';
    if (el.dataset.gdone === txt) return;
    el.dataset.gdone = txt;
    el.className = 'grd' + (g ? '' : ' none') + (el.dataset.gbig ? ' gbig' : '');
    el.style.background = g ? COLORS[g.grade] : '';
    el.textContent = txt;
    el.title = g ? `Grade ${g.grade} · score ${String(g.score).replace('.', ',')}/100 · ${g.reason} Klik voor details.`
      : (grades ? 'Nog geen grade: ticker niet volledig gescand' : 'Grade laden…');
  }
  // Automatisch (bv. op de portfolio server, script met data-auto): een grade vóór elke klikbare ticker
  // (elementen met onclick="tkNaar(...)" of class ticker-link). Tickers zonder grade krijgen niets, zodat het rustig blijft.
  const AUTO = !!(me && me.dataset.auto !== undefined);
  const AUTO_SEL = me && me.dataset.auto ? me.dataset.auto : '[onclick*="tkNaar"], .ticker-link';
  function autoTag() {
    if (!AUTO || !grades) return;
    document.querySelectorAll(AUTO_SEL).forEach(el => {
      if (el.dataset.gauto || el.closest('#grd-pop') || el.querySelector('[data-gsym]')) return;
      const m = /tkNaar\(\s*['"]([^'"]+)['"]/.exec(el.getAttribute('onclick') || '');
      const t = (el.dataset.t || (m && m[1]) || el.textContent || '').trim().toUpperCase();
      el.dataset.gauto = '1';
      if (!t || t.length > 15 || !get(t)) return;
      const b = document.createElement('span');
      b.dataset.gsym = t;
      const host = el.querySelector('.wl-sym') || el;
      if (host.tagName === 'A' || !host.children.length && host.textContent.trim() === t) host.before(b);
      else host.prepend(b);
    });
  }
  function decorate(root = document) { autoTag(); root.querySelectorAll('[data-gsym]').forEach(fill); }
  let timer = null;
  new MutationObserver(() => { clearTimeout(timer); timer = setTimeout(decorate, 50); })
    .observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-gsym'] });

  async function load() {
    try {
      const d = await (await fetch(BASE + '/api/grades')).json();
      grades = d.grades || {}; aliases = d.aliases || {}; check = d.check; formula = d.formula || [];
      document.querySelectorAll('[data-gsym]').forEach(el => { delete el.dataset.gdone; });
      document.querySelectorAll('[data-gauto]').forEach(el => { delete el.dataset.gauto; });
      decorate();
      document.dispatchEvent(new CustomEvent('grades-loaded'));
    } catch (e) { console.warn('grades', e); }
  }

  function checkHtml(c) {
    if (!c || !c.by_grade) return `<p class="mu">Nog niet gemeten. Start de walk-forward check: die rekent de grade terug voor elke maand
      in het verleden (alleen met de koersen tot die dag) en kijkt wat het aandeel de maand erna deed.</p>`;
    const rows = c.by_grade.map(b => `<tr><td><span class="grd" style="background:${COLORS[b.grade]}">${b.grade}</span></td>
      <td>${b.n}</td><td class="${cl(b.avg)}">${pct(b.avg, 2)}</td><td class="${cl(b.median)}">${pct(b.median, 2)}</td><td class="${cl(b.rel)}">${pct(b.rel, 2)}</td>
      <td>${b.hit == null ? '–' : Math.round(b.hit * 100) + '%'}</td><td class="${cl(b.fwd3)}">${pct(b.fwd3, 1)}</td></tr>`).join('');
    return `<div class="verdict">${esc(c.verdict)}</div>
      <table><tr><th>Grade</th><th>Meetpunten</th><th title="Uitschieters afgetopt op −50% en +100%">Maand erna gem.</th><th>Mediaan</th><th>t.o.v. eigen gem. ticker</th><th>% positief</th><th>3 maanden erna</th></tr>
      ${rows}<tr><td class="mu">Alle</td><td>${c.all?.n ?? '–'}</td><td class="${cl(c.all?.avg)}">${pct(c.all?.avg, 2)}</td><td class="${cl(c.all?.median)}">${pct(c.all?.median, 2)}</td><td>–</td>
      <td>${c.all?.hit == null ? '–' : Math.round(c.all.hit * 100) + '%'}</td><td></td></tr></table>
      <p class="mu">${c.tickers} tickers, ${esc(c.from)} t/m ${esc(c.to)}, een meetpunt op elk maandeinde, rendement over de ${c.horizon_days} handelsdagen erna met instap op de
      opening van de dag erna. Rangcorrelatie score ↔ maand erna (IC): ${c.ic == null ? '–' : c.ic.toFixed(3).replace('.', ',')}.
      Gemeten ${esc(c.finished || '')}.</p>`;
  }

  function stratRows(list, withShare) {
    return list.map(x => `<tr><td>${x.open ? '<span class="open">OPEN</span> ' : ''}${esc(x.strategy)}<br><span class="mu">${esc(x.category_label)}</span></td>
      <td>${x.open ? esc(x.since || '–') + (x.bars != null ? ` <span class="mu">(${x.bars} d)</span>` : '') : '<span class="mu">cash</span>'}</td>
      <td>${x.entry != null ? x.entry.toLocaleString('nl-BE', { maximumFractionDigits: 2 }) : '–'}</td>
      <td class="${cl(x.trade_return)}">${pct(x.trade_return)}</td>
      <td class="${cl(x.train_return)}">${pct(x.train_return)}</td><td class="${cl(x.test_return)}">${pct(x.test_return)}</td>
      <td class="${cl(x.return_2j)}">${pct(x.return_2j)}</td>
      <td>${x.win_rate != null ? Math.round(x.win_rate * 100) + '%' + (x.trades != null ? ` <span class="mu">/${x.trades}</span>` : '') : '–'}</td>
      ${withShare ? `<td><span class="bar" style="width:${Math.max(2, Math.round(x.share * 80))}px"></span> ${Math.round(x.share * 100)}%</td>` : ''}</tr>`).join('');
  }
  const head = share => `<tr><th>Strategie</th><th>Open sinds</th><th>Instap</th><th>Trade nu</th><th>Leerperiode</th><th>Testjaar</th>
    <th>Laatste 2 jaar</th><th>Winst-trades</th>${share ? '<th>Gewicht</th>' : ''}</tr>`;

  async function show(sym) {
    let pop = document.getElementById('grd-pop');
    if (!pop) {
      pop = document.createElement('div');
      pop.id = 'grd-pop';
      pop.addEventListener('click', e => { if (e.target === pop || e.target.classList.contains('x')) pop.remove(); });
      document.body.appendChild(pop);
    }
    pop.innerHTML = `<div class="box"><h2>${esc(sym)} <button class="x">×</button></h2><p class="mu">Laden…</p></div>`;
    let g;
    try {
      const r = await fetch(BASE + '/api/grade/' + encodeURIComponent(key(sym) || sym));
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      g = await r.json();
    } catch (e) {
      pop.querySelector('.box').innerHTML = `<h2>${esc(sym)} <button class="x">×</button></h2><p>${esc(e.message)}</p>`;
      return;
    }
    const c = g.check;
    const myRow = c?.by_grade?.find(b => b.grade === g.grade);
    pop.querySelector('.box').innerHTML = `
      <h2><span class="grd big" style="background:${COLORS[g.grade]}">${g.grade}</span> ${esc(g.symbol)}
        <span class="mu" style="font-size:13px;font-weight:400">score ${String(g.score).replace('.', ',')}/100</span><button class="x">×</button></h2>
      <p>${esc(g.reason)}</p>
      <div class="kpi">
        <div><span>Stemming (gewogen open)</span><b>${Math.round(g.vote * 100)}%</b></div>
        <div><span>Vertrouwen</span><b>${Math.round(g.confidence * 100)}%</b></div>
        <div><span>Werkende strategieën</span><b>${g.open_working} open / ${g.working}</b></div>
        <div><span>Alle open buys</span><b>${g.open_total} van ${g.strategies}</b></div>
        <div><span>B&amp;H leerperiode · testjaar</span><b><span class="${cl(g.bh_train)}">${pct(g.bh_train)}</span> · <span class="${cl(g.bh_test)}">${pct(g.bh_test)}</span></b></div>
        ${myRow && myRow.n ? `<div><span>Historisch na grade ${g.grade}</span><b class="${cl(myRow.avg)}">${pct(myRow.avg, 2)}</b> <span class="mu">maand erna, ${Math.round(myRow.hit * 100)}% positief</span></div>` : ''}
      </div>
      <h3>Strategieën die voor deze ticker werken (${g.working})</h3>
      ${g.working_list.length ? `<table>${head(true)}${stratRows(g.working_list, true)}</table>` : '<p class="mu">Geen enkele.</p>'}
      ${g.open_other.length ? `<details><summary>Ook open, maar niet bewezen voor deze ticker (${g.open_other.length}, telt niet mee)</summary>
        <table>${head(false)}${stratRows(g.open_other, false)}</table></details>` : ''}
      <h3>Hoe de grade berekend wordt</h3><ol>${(g.formula || formula).map(f => `<li>${esc(f)}</li>`).join('')}</ol>
      <p class="mu">Leerperiode ${esc(g.train?.start || '')} t/m ${esc(g.train?.end || '')}, testjaar ${esc(g.test?.start || '')} t/m ${esc(g.test?.end || '')}.
        Gescand ${esc(g.scanned_at || '')}. Geen beleggingsadvies.</p>
      <h3>Voorspelde de grade iets? (walk-forward)</h3>${checkHtml(c)}
      <p><button class="run">Check opnieuw draaien</button> <span class="mu cks"></span></p>`;
    pop.querySelector('.run').onclick = async () => {
      const out = pop.querySelector('.cks');
      const r = await fetch(BASE + '/api/grade/check', { method: 'POST' });
      const d = await r.json().catch(() => ({}));
      out.textContent = r.ok ? `Gestart op ${d.tickers} tickers; duurt enkele minuten. Open dit venster daarna opnieuw.` : (d.detail || 'Mislukt');
    };
  }

  document.addEventListener('click', e => {
    const el = e.target.closest && e.target.closest('[data-gsym]');
    if (!el) return;
    e.preventDefault(); e.stopPropagation();
    show(el.dataset.gsym);
  }, true);

  window.Grade = {
    badge: s => s ? `<span data-gsym="${esc(s)}"></span>` : '',
    get, show, reload: load, colors: COLORS,
    score: s => get(s)?.score ?? -1,
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => { decorate(); load(); });
  else { decorate(); load(); }
  setInterval(load, 5 * 60 * 1000);
})();
