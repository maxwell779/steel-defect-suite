import React from 'react';

export default function TopBar({ mode, setMode, health }) {
  const ok = health?.models_loaded;
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark"><span style={{ color: '#fff', fontWeight: 800, fontSize: 18 }}>S</span></div>
        <div>
          <div><span className="brand-name">Steel Inspection</span><span className="brand-sub">CONSOLE</span></div>
          <div className="brand-line">Severstal · 픽셀 세그멘테이션 · 분류 게이트</div>
        </div>
      </div>
      <div className="mode-switch">
        <button className={`mode-btn ${mode === 'ops' ? 'active' : ''}`} onClick={() => setMode('ops')}>
          <span className="dot" style={{ background: '#3b82f6' }} />운영
        </button>
        <button className={`mode-btn ${mode === 'analysis' ? 'active' : ''}`} onClick={() => setMode('analysis')}>
          <span className="dot" style={{ background: '#a855f7' }} />분석
        </button>
      </div>
      <div className="top-right">
        <span className={`top-chip ${ok ? 'ok' : 'warn'}`}>
          <span className="pulse" style={{ background: ok ? '#22c55e' : '#f59e0b' }} />
          {ok ? `모델 로드됨 · ${health.device}` : '정적 데모'}
        </span>
        <span className="top-chip">fold0 0.9573 · OOF 0.9532</span>
      </div>
    </header>
  );
}
