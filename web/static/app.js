const COLORS = {C1:[239,68,68], C2:[34,197,94], C3:[59,130,246], C4:[234,179,8]};
const $ = s => document.querySelector(s);
const rgb = a => `rgb(${a[0]},${a[1]},${a[2]})`;
let SAMPLES = [], EXP = null, curUpload = null;

// 백엔드(/api) 우선, 없으면 정적 파일로 폴백 → 정적 호스팅에서도 동작
async function getJSON(api, staticPath) {
  try { const r = await fetch(api); if (r.ok) return await r.json(); } catch {}
  try { return await (await fetch(staticPath)).json(); } catch { return {}; }
}

// ── 탭 ──
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); $('#' + b.dataset.t).classList.add('active');
});

async function boot() {
  // health
  try {
    const h = await (await fetch('/api/health')).json();
    const el = $('#status');
    if (h.models_loaded) { el.className = 'badge on'; el.textContent = `● 모델 로드됨 (${h.device})`; }
    else { el.className = 'badge off'; el.textContent = '● 정적 폴백(모델 미로드)'; }
  } catch { $('#status').className = 'badge off'; $('#status').textContent = '● 오프라인'; }

  const sj = await getJSON('/api/samples', 'samples/samples.json'); SAMPLES = sj.samples || [];
  EXP = await getJSON('/api/experiments', 'experiments.json');
  renderConsole(); renderThumbs(); renderLegend(); renderGate(); renderCharts(); renderVs();
  if (SAMPLES.length) selectSample(0);
}

// ── 통합 콘솔 ──
function renderConsole() {
  const n = SAMPLES.length || 1;
  const defects = SAMPLES.filter(s => s.per_class.some(p => p.area > 0));
  const meanD = (SAMPLES.reduce((a, s) => a + (s.mean_dice ?? 0), 0) / n);
  const kpis = [
    ['검사 수', n, '데모 샘플', 'var(--acc)'],
    ['결함율', (100 * defects.length / n).toFixed(0) + '%', `${defects.length}/${n}`, 'var(--warn)'],
    ['격리 대기', defects.length, '결함 검출분', 'var(--bad)'],
    ['양품률', (100 * (n - defects.length) / n).toFixed(0) + '%', '정상 판정', 'var(--ok)'],
    ['추정 비용절감', '₩' + (defects.length * 18).toLocaleString() + '만', '결함 1건=180만 가정', 'var(--ok)'],
  ];
  $('#kpis').innerHTML = kpis.map(k =>
    `<div class="card kpi"><div class="v" style="color:${k[3]}">${k[1]}</div><div class="l">${k[0]}</div><div class="d muted">${k[2]}</div></div>`).join('');
  $('#queue').querySelector('tbody').innerHTML = defects.map(s => {
    const cls = s.per_class.filter(p => p.area > 0).map(p => `<span class="tag" style="color:${rgb(COLORS[p.cls])}">${p.cls}</span>`).join(' ');
    return `<tr><td>${s.id}</td><td>${cls}</td><td>${s.mean_dice ?? '–'}</td><td><span class="tag warn">격리</span></td></tr>`;
  }).join('') || '<tr><td colspan=4 class="muted">결함 없음</td></tr>';
  $('#bestdice').textContent = EXP.milestones.dice.at(-1).toFixed(4);
}

// ── Segmentation (LIVE 레이어 합성: 클래스 on/off·투명도·GT비교) ──
let SEG = null;            // {base, layers[4], gt[4]|null, per_class, dice}
let curId = null;          // 현재 샘플 id (null=업로드)
const vis = [true, true, true, true];

function renderThumbs() {
  $('#thumbs').innerHTML = SAMPLES.map((s, i) =>
    `<img src="samples/${s.base}" data-i="${i}" title="${s.id}">`).join('');
  document.querySelectorAll('#thumbs img').forEach(im => im.onclick = () => selectSample(+im.dataset.i));
}
function renderLegend() {
  const names = ['C1', 'C2', 'C3', 'C4'];
  $('#legend').innerHTML = names.map(c => `<span class="flex" style="font-size:12px;gap:4px"><span class="sw" style="background:${rgb(COLORS[c])}"></span>${c}</span>`).join('');
  $('#clstoggles').innerHTML = names.map((c, i) => `<label class="flex muted" style="font-size:12px;gap:3px"><input type="checkbox" data-c="${i}" checked>${c}</label>`).join('');
  document.querySelectorAll('#clstoggles input').forEach(cb => cb.onchange = () => { vis[+cb.dataset.c] = cb.checked; redraw(); });
}
function params() { return {min_prob: $('#min_prob').value, max_prob: $('#max_prob').value, min_area: $('#min_area').value, gate: $('#gate').checked}; }
function viewMode() { return (document.querySelector('input[name=view]:checked') || {}).value || 'pred'; }

function redraw() {
  if (!SEG) return;
  const cv = $('#cv'), ctx = cv.getContext('2d'), op = +$('#opacity').value;
  const layers = (viewMode() === 'gt' && SEG.gt) ? SEG.gt : SEG.layers;
  const b = new Image();
  b.onload = () => {
    ctx.clearRect(0, 0, cv.width, cv.height); ctx.drawImage(b, 0, 0, cv.width, cv.height);
    (layers || []).forEach((src, i) => {
      if (!vis[i] || !src) return;
      const o = new Image();
      o.onload = () => { ctx.globalAlpha = op; ctx.drawImage(o, 0, 0, cv.width, cv.height); ctx.globalAlpha = 1; };
      o.src = src;
    });
  };
  b.src = SEG.base;
}
function setSeg(r) {
  SEG = {base: r.base_png, layers: r.class_overlays, gt: r.gt_overlays, per_class: r.per_class, dice: r.mean_dice};
  renderBars(r.per_class, r.mean_dice); redraw();
}
function renderBars(per, dice) {
  $('#segdice').textContent = dice ?? '–';
  $('#pcbars').innerHTML = per.map(p => {
    const prob = (p.present_prob * 100).toFixed(0);
    const dtxt = p.dice != null ? ` · Dice ${p.dice}` : '';
    const g = p.gated_off ? '<span class="tag off">게이트 OFF</span>' : (p.area > 0 ? '<span class="tag on">검출</span>' : '<span class="tag">없음</span>');
    return `<div class="pcbar"><span class="sw" style="background:${rgb(p.color || COLORS[p.cls])}"></span>
      <span class="nm">${p.cls}</span><div class="bar"><span style="width:${prob}%;background:${rgb(p.color || COLORS[p.cls])}"></span></div>
      <span style="width:150px;text-align:right" class="muted">${prob}%${dtxt} · ${p.area}px</span>${g}</div>`;
  }).join('');
}
async function selectSample(i) {
  curUpload = null; const s = SAMPLES[i]; curId = s.id;
  document.querySelectorAll('#thumbs img').forEach((im, j) => im.classList.toggle('sel', j === i));
  $('#segid').textContent = '· ' + s.id; $('#livenote').textContent = 'LIVE 추론중…';
  try {
    const q = new URLSearchParams({id: s.id, ...params()}).toString();
    const r = await (await fetch('/api/infer_sample?' + q)).json();
    if (!r.available) throw 0;
    setSeg(r); $('#livenote').textContent = 'LIVE · GT 비교 가능 · 슬라이더 재추론';
  } catch {
    SEG = null;                               // 정적 폴백
    const cv = $('#cv'), ctx = cv.getContext('2d');
    const b = new Image(); b.onload = () => { ctx.clearRect(0, 0, cv.width, cv.height); ctx.drawImage(b, 0, 0, cv.width, cv.height); const o = new Image(); o.onload = () => ctx.drawImage(o, 0, 0, cv.width, cv.height); o.src = 'samples/' + s.overlay; }; b.src = 'samples/' + s.base;
    renderBars(s.per_class, s.mean_dice); $('#livenote').textContent = '정적 폴백(모델 미로드)';
  }
}
$('#file').onchange = e => { curUpload = e.target.files[0]; curId = null; if (curUpload) runUpload(); };
async function runUpload() {
  if (!curUpload) return;
  $('#segid').textContent = '· 업로드'; $('#livenote').textContent = 'LIVE 추론중…';
  const p = params(), fd = new FormData(); fd.append('file', curUpload);
  fd.append('min_prob', p.min_prob); fd.append('max_prob', p.max_prob);
  fd.append('min_area', p.min_area); fd.append('gate', p.gate);
  try {
    const r = await (await fetch('/api/infer', {method: 'POST', body: fd})).json();
    if (!r.available) { $('#livenote').textContent = '모델 미로드 — 업로드 추론 불가'; return; }
    setSeg(r); $('#livenote').textContent = 'LIVE · 슬라이더 재추론 (업로드는 GT 없음)';
  } catch { $('#livenote').textContent = '추론 실패'; }
}
['min_prob', 'max_prob', 'min_area'].forEach(id => $('#' + id).oninput = () => {
  $('#v_min').textContent = (+$('#min_prob').value).toFixed(2);
  $('#v_max').textContent = (+$('#max_prob').value).toFixed(2);
  $('#v_area').textContent = $('#min_area').value; debounce();
});
$('#gate').onchange = refetch;
$('#opacity').oninput = redraw;
document.querySelectorAll('input[name=view]').forEach(r => r.onchange = redraw);
let t; function debounce() { clearTimeout(t); t = setTimeout(refetch, 350); }
function refetch() { if (curUpload) runUpload(); else if (curId != null) { const i = SAMPLES.findIndex(s => s.id === curId); if (i >= 0) selectSample(i); } }
$('#export').onclick = () => {
  if (!SEG) return;
  const rows = [['class', 'present_prob', 'dice', 'area', 'gated_off']].concat(
    SEG.per_class.map(p => [p.cls, p.present_prob, p.dice ?? '', p.area, p.gated_off]));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([rows.map(r => r.join(',')).join('\n')], {type: 'text/csv'}));
  a.download = (curId || 'upload') + '_result.csv'; a.click();
};

// ── Gate 탭 ──
function renderGate() {
  $('#gatesamples').innerHTML = SAMPLES.map(s => {
    const bars = s.per_class.map((p, c) => {
      const prob = (s.clf_probs[c] * 100).toFixed(0);
      const off = p.gated_off ? '<span class="tag off">차단</span>' : '<span class="tag on">통과</span>';
      return `<div class="pcbar"><span class="sw" style="background:${rgb(COLORS[p.cls])}"></span><span class="nm">${p.cls}</span>
        <div class="bar"><span style="width:${prob}%;background:${rgb(COLORS[p.cls])}"></span></div>
        <span style="width:60px;text-align:right" class="muted">${prob}%</span>${off}</div>`;
    }).join('');
    return `<div class="card" style="margin:10px 0;background:var(--panel2)"><b>${s.id}</b>${bars}</div>`;
  }).join('');
}

// ── Charts ──
function bar(id, labels, datasets, opts = {}) {
  new Chart($('#' + id), {type: opts.type || 'bar',
    data: {labels, datasets}, options: {responsive:true, maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#cdd9ec'}}}, scales:{
        x:{ticks:{color:'#8aa0bf'},grid:{color:'#26314a'}},
        y:{ticks:{color:'#8aa0bf'},grid:{color:'#26314a'},...(opts.y||{})}}}});
}
function renderCharts() {
  const A = '#3b82f6', G = '#22c55e', Y = '#f59e0b', R = '#ef4444';
  bar('c_ms', EXP.milestones.labels, [{label:'mean Dice', data:EXP.milestones.dice, backgroundColor:A}], {y:{min:0.92,max:0.96}});
  bar('c_gain', EXP.ensemble_gain.labels, [{label:'Dice', data:EXP.ensemble_gain.dice, backgroundColor:G}], {y:{min:0.93,max:0.96}});
  bar('c_pc', EXP.per_class_dice.labels, [
    {label:'ungated', data:EXP.per_class_dice.ungated, backgroundColor:'#64748b'},
    {label:'gated', data:EXP.per_class_dice.gated, backgroundColor:G}], {y:{min:0.8,max:1}});
  bar('c_pp', EXP.preproc_cross.labels, [{label:'Dice', data:EXP.preproc_cross.dice,
    backgroundColor:EXP.preproc_cross.dice.map((d,i)=>i===0?G:Y)}], {y:{min:0.94,max:0.953}});
  bar('c_leak', EXP.leak.labels, [
    {label:'AUC', data:EXP.leak.auc, backgroundColor:A},
    {label:'acc', data:EXP.leak.acc, backgroundColor:Y},
    {label:'F1', data:EXP.leak.f1, backgroundColor:G}], {y:{min:0.8,max:1}});
  bar('c_5f', EXP.fivefold.labels, [{label:'Dice', data:EXP.fivefold.dice, backgroundColor:A}], {y:{min:0.948,max:0.955}});
  bar('c_auroc', EXP.gate.labels, [{label:'AUROC', data:EXP.gate.auroc, backgroundColor:G}], {y:{min:0.98,max:1}});
  bar('c_block', EXP.gate.labels, [{label:'empty 차단율', data:EXP.gate.empty_block_at_recall95, backgroundColor:A}], {y:{min:0.9,max:1}});
}
function renderVs() {
  $('#vs').querySelector('tbody').innerHTML =
    '<tr><th></th><th>Vision-Q TEAM 1</th><th>본 프로젝트</th></tr>' +
    EXP.vs_other_team.map(r => `<tr><td class="muted">${r[0]}</td><td>${r[1]}</td><td style="color:var(--ok)">${r[2]}</td></tr>`).join('');
}
boot();
