import React from 'react';
import { Spark } from '../charts.jsx';

function KPI({ label, value, sub, tone, spark, gauge, gaugeColor }) {
  return (
    <div className={`kpi tone-${tone}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-sub">{sub}</div>
      {spark && <Spark data={spark} color={{ ok: '#22c55e', ng: '#ef4444', rate: '#f59e0b', util: '#06b6d4', neutral: '#3b82f6' }[tone]} />}
      {gauge != null && <div className="kpi-gauge"><div className="kpi-gauge-fill" style={{ width: `${Math.min(100, gauge * 100)}%`, background: gaugeColor || '#3b82f6' }} /></div>}
    </div>
  );
}

export default function KPICards({ k }) {
  if (!k) return null;
  return (
    <div className="kpis">
      <KPI label="검증 검사 수" value={k.total.toLocaleString()} sub={`fold0 held-out · train ${k.train_images.toLocaleString()}`} tone="neutral" spark={[40, 52, 48, 60, 58, 66, 62, 70]} />
      <KPI label="결함 검출" value={k.defect.toLocaleString()} sub={`결함율 ${k.defect_rate}%`} tone="ng" gauge={k.defect_rate / 100} gaugeColor="#ef4444" />
      <KPI label="정상" value={k.normal.toLocaleString()} sub={`양품률 ${k.normal_rate}%`} tone="ok" gauge={k.normal_rate / 100} gaugeColor="#22c55e" />
      <KPI label="최종 mean Dice" value={k.mean_dice_fold0.toFixed(4)} sub="5모델+TTA+게이트(fold0)" tone="util" gauge={k.mean_dice_fold0} gaugeColor="#06b6d4" />
      <KPI label="무누수 OOF Dice" value={k.oof_dice.toFixed(4)} sub="전체 train 일반화" tone="rate" gauge={k.oof_dice} gaugeColor="#f59e0b" />
    </div>
  );
}
