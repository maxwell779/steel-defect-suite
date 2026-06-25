import React, { useEffect, useRef, useState } from 'react';
import { api, sampleUrl } from '../api/client.js';

const CLS = ['C1', 'C2', 'C3', 'C4'];
const COLOR = { C1: '#ef4444', C2: '#22c55e', C3: '#3b82f6', C4: '#f59e0b' };

export default function SegViewer({ samples }) {
  const [idx, setIdx] = useState(0);
  const [res, setRes] = useState(null);        // {base_png, class_overlays, gt_overlays, per_class, mean_dice}
  const [view, setView] = useState('pred');    // pred | gt
  const [vis, setVis] = useState([true, true, true, true]);
  const [op, setOp] = useState(0.85);
  const [pp, setPp] = useState({ min_prob: 0.6, max_prob: 0.7, min_area: 600, gate: true });
  const [note, setNote] = useState('');
  const [upload, setUpload] = useState(null);
  const cv = useRef(null);

  // 추론(샘플 or 업로드)
  useEffect(() => {
    let on = true;
    (async () => {
      setNote('추론 중…');
      let r;
      if (upload) r = await api.inferUpload(upload, pp);
      else if (samples[idx]) r = await api.inferSample(samples[idx].id, pp);
      if (!on) return;
      if (r && r.available) { setRes(r); setNote(upload ? 'LIVE · 업로드(GT 없음)' : 'LIVE · GT 비교 가능'); }
      else if (samples[idx]) {  // 정적 폴백
        setRes({ base: sampleUrl(samples[idx].base), overlayStatic: sampleUrl(samples[idx].overlay),
          per_class: samples[idx].per_class, mean_dice: samples[idx].mean_dice });
        setNote('정적 폴백');
      }
    })();
    return () => { on = false; };
  }, [idx, upload, pp, samples]);

  // 캔버스 그리기
  useEffect(() => {
    if (!res) return;
    const c = cv.current, ctx = c.getContext('2d');
    const base = new Image();
    base.onload = () => {
      ctx.clearRect(0, 0, c.width, c.height); ctx.drawImage(base, 0, 0, c.width, c.height);
      if (res.overlayStatic) { const o = new Image(); o.onload = () => ctx.drawImage(o, 0, 0, c.width, c.height); o.src = res.overlayStatic; return; }
      const layers = view === 'gt' && res.gt_overlays ? res.gt_overlays : res.class_overlays;
      (layers || []).forEach((src, i) => {
        if (!vis[i] || !src) return;
        const o = new Image(); o.onload = () => { ctx.globalAlpha = op; ctx.drawImage(o, 0, 0, c.width, c.height); ctx.globalAlpha = 1; }; o.src = src;
      });
    };
    base.src = res.base || res.base_png;
  }, [res, view, vis, op]);

  const per = res?.per_class || [];
  return (
    <div className="card viewer-card" style={{ height: 'auto' }}>
      <div className="card-head sm">
        <div><div className="card-title">Segmentation — 4색 마스크 오버레이</div>
          <div className="card-sub">{upload ? '업로드' : samples[idx]?.id} · {note}</div></div>
        <div className="viewer-toggles">
          {['pred', 'gt'].map((v) => <button key={v} className={`toggle ${view === v ? 'on' : ''}`} onClick={() => setView(v)}>{v === 'pred' ? '예측' : '정답'}</button>)}
        </div>
      </div>

      <div className="seg-nav">
        <button className="navbtn" disabled={!upload && idx <= 0}
          onClick={() => { setUpload(null); setIdx((i) => Math.max(0, i - 1)); }}>◀ 이전</button>
        <span className="seg-nav-id mono">{upload ? '업로드 이미지' : `${samples[idx]?.id || ''}  ·  ${idx + 1}/${samples.length}`}</span>
        <button className="navbtn" disabled={!upload && idx >= samples.length - 1}
          onClick={() => { setUpload(null); setIdx((i) => Math.min(samples.length - 1, i + 1)); }}>다음 ▶</button>
        <label className="navbtn" style={{ cursor: 'pointer', marginLeft: 'auto' }}>＋ 업로드
          <input type="file" accept="image/*" hidden onChange={(e) => e.target.files[0] && setUpload(e.target.files[0])} /></label>
      </div>

      <canvas ref={cv} width="1600" height="256" className="seg-canvas" />

      <div className="viewer-foot">
        <div className="cls-toggles">
          {CLS.map((c, i) => (
            <label key={c} className={`toggle ${vis[i] ? 'on' : ''}`}>
              <input type="checkbox" checked={vis[i]} onChange={() => setVis((v) => v.map((x, j) => j === i ? !x : x))} hidden />
              <span className="toggle-dot" style={{ background: COLOR[c] }} />{c}
            </label>
          ))}
        </div>
        <label className="dim sm">투명도 <input type="range" min="0.2" max="1" step="0.1" value={op} onChange={(e) => setOp(+e.target.value)} /></label>
      </div>

      <div className="seg-grid">
        <div className="pc-bars">
          {per.map((p) => {
            const prob = Math.round((p.present_prob ?? 0) * 100);
            return (
              <div key={p.cls} className="pcbar">
                <span className="legend-dot" style={{ background: COLOR[p.cls] }} /><span className="pc-nm">{p.cls}</span>
                <div className="pc-track"><span style={{ width: `${prob}%`, background: COLOR[p.cls] }} /></div>
                <span className="pc-meta mono">{prob}%{p.dice != null ? ` · D ${p.dice}` : ''} · {p.area}px</span>
                <span className={`sev ${p.gated_off ? 'sev-critical' : p.area > 0 ? 'on' : ''}`}>{p.gated_off ? '게이트OFF' : p.area > 0 ? '검출' : '없음'}</span>
              </div>
            );
          })}
          {res?.mean_dice != null && <div className="seg-dice">mean Dice <b>{res.mean_dice}</b></div>}
        </div>
        <div className="pp-ctrl">
          <div className="param-group-title">마스크 정제 · 게이트 (LIVE)</div>
          {[['min_prob', 0.1, 0.9, 0.05], ['max_prob', 0.3, 0.95, 0.05], ['min_area', 0, 3000, 100]].map(([k, mn, mx, st]) => (
            <label key={k} className="slider-head">{k} <b className="mono">{pp[k]}</b>
              <input type="range" min={mn} max={mx} step={st} value={pp[k]} style={{ width: '100%' }}
                onChange={(e) => setPp((s) => ({ ...s, [k]: k === 'min_area' ? +e.target.value : +e.target.value }))} /></label>
          ))}
          <label className="select-row"><span className="select-label">분류 게이트</span>
            <input type="checkbox" checked={pp.gate} onChange={(e) => setPp((s) => ({ ...s, gate: e.target.checked }))} /></label>
        </div>
      </div>
    </div>
  );
}
