// 차트 모음 — App 방식(외부 라이브러리 없이 SVG/CSS 직접). 모두 반응형.
import React from 'react';

const fmtN = (v) => (v >= 1000 ? v.toLocaleString() : v);

// ── 도넛 ──
export function Donut({ data, unit = '' }) {
  const total = data.reduce((a, d) => a + d.count, 0) || 1;
  const R = 70, CX = 80, CY = 80;
  let acc = 0;
  const seg = (val) => {
    const start = (acc / total) * 360, end = ((acc + val) / total) * 360; acc += val;
    if (Math.abs(end - start) >= 359.99)
      return `M ${CX} ${CY - R} A ${R} ${R} 0 1 1 ${CX - 0.01} ${CY - R} Z`;
    const s = ((start - 90) * Math.PI) / 180, e = ((end - 90) * Math.PI) / 180;
    const x1 = CX + R * Math.cos(s), y1 = CY + R * Math.sin(s);
    const x2 = CX + R * Math.cos(e), y2 = CY + R * Math.sin(e);
    return `M ${CX} ${CY} L ${x1} ${y1} A ${R} ${R} 0 ${end - start > 180 ? 1 : 0} 1 ${x2} ${y2} Z`;
  };
  return (
    <div className="pie-wrap">
      <svg className="pie" viewBox="0 0 160 160">
        {data.map((d, i) => <path key={i} d={seg(d.count)} fill={d.color} stroke="#fff" strokeWidth="1.5" />)}
        <circle cx={CX} cy={CY} r="40" fill="#fff" />
        <text x={CX} y={CY - 4} textAnchor="middle" fontSize="22" fontWeight="800" fill="#0f172a">{fmtN(total)}</text>
        <text x={CX} y={CY + 14} textAnchor="middle" fontSize="10" fill="#64748b">{unit || 'total'}</text>
      </svg>
      <div className="pie-legend">
        {data.map((d, i) => (
          <div key={i} className="legend-row">
            <span className="legend-dot" style={{ background: d.color }} />
            <span>{d.name}</span>
            <span className="mono">{fmtN(d.count)}</span>
            <span className="mono dim">{((d.count / total) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── 단일/그룹 막대 ──
export function Bars({ labels, series, yMin = 0, yMax = 1, fmt = (v) => v.toFixed(3), height = 200 }) {
  const groups = labels.length, n = series.length;
  const span = (yMax - yMin) || 1;
  return (
    <div className="chart-bars" style={{ height }}>
      {labels.map((lb, gi) => (
        <div key={gi} className="bar-group">
          <div className="bar-stack-h">
            {series.map((s, si) => {
              const v = s.data[gi];
              const h = Math.max(2, ((v - yMin) / span) * (height - 34));
              return (
                <div key={si} className="bar-col" title={`${s.name}: ${fmt(v)}`}
                  style={{ height: h, background: s.color }}>
                  <span className="bar-val">{fmt(v)}</span>
                </div>
              );
            })}
          </div>
          <div className="bar-label">{lb}</div>
        </div>
      ))}
      {n > 1 && (
        <div className="chart-legend">
          {series.map((s, i) => <span key={i}><span className="legend-dot" style={{ background: s.color }} />{s.name}</span>)}
        </div>
      )}
    </div>
  );
}

// ── 수평 진행바 리스트 ──
export function ProgressList({ items }) {
  return (
    <div className="prog-list">
      {items.map((it, i) => (
        <div key={i} className="prog-row">
          <span className="prog-k">{it.label}</span>
          <div className="prog-bar"><span style={{ width: `${it.pct}%`, background: it.color || '#3b82f6' }} /></div>
          <span className="prog-v mono">{it.text}</span>
        </div>
      ))}
    </div>
  );
}

// ── 라인(관리도 스타일: 값 + 기준선) ──
export function LineChart({ points, ucl, lcl, cl, color = '#3b82f6', fmt = (v) => v.toFixed(4), yPad = 0.0008 }) {
  const W = 620, H = 180, P = 34;
  const ys = points.map((p) => p.y);
  const lo = Math.min(...ys, lcl ?? Infinity) - yPad, hi = Math.max(...ys, ucl ?? -Infinity) + yPad;
  const span = (hi - lo) || 1;
  const xAt = (i) => P + ((W - 2 * P) * i) / Math.max(1, points.length - 1);
  const yAt = (v) => H - P - ((H - 2 * P) * (v - lo)) / span;
  const path = points.map((p, i) => `${i ? 'L' : 'M'} ${xAt(i)} ${yAt(p.y)}`).join(' ');
  const line = (v, c, dash) => v != null && (
    <line x1={P} x2={W - P} y1={yAt(v)} y2={yAt(v)} stroke={c} strokeWidth="1" strokeDasharray={dash} />
  );
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      {line(ucl, '#ef4444', '4 3')}{line(cl, '#22c55e', '2 2')}{line(lcl, '#ef4444', '4 3')}
      <path d={path} fill="none" stroke={color} strokeWidth="2" />
      {points.map((p, i) => <circle key={i} cx={xAt(i)} cy={yAt(p.y)} r="3.5" fill={color} />)}
      {points.map((p, i) => <text key={i} x={xAt(i)} y={H - P + 16} textAnchor="middle" fontSize="10" fill="#64748b">{p.x}</text>)}
      {points.map((p, i) => <text key={i} x={xAt(i)} y={yAt(p.y) - 8} textAnchor="middle" fontSize="9" fill="#334155" className="mono">{fmt(p.y)}</text>)}
    </svg>
  );
}

// ── 스파크라인 ──
export function Spark({ data, color = '#3b82f6', h = 36 }) {
  const max = Math.max(...data, 1), min = Math.min(...data, 0), w = 100;
  const pts = data.map((v, i) => `${(i / Math.max(1, data.length - 1)) * w},${h - ((v - min) / Math.max(1e-6, max - min)) * h}`).join(' ');
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <polyline points={`0,${h} ${pts} ${w},${h}`} fill={color} opacity="0.15" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2.2" strokeLinejoin="round" />
    </svg>
  );
}
