import React, { useEffect, useState } from 'react';
import TopBar from './components/TopBar.jsx';
import KPICards from './components/KPICards.jsx';
import SegViewer from './components/SegViewer.jsx';
import { Donut, Bars, ProgressList, LineChart } from './charts.jsx';
import { api } from './api/client.js';

const Card = ({ title, sub, children, span }) => (
  <div className="card" style={span ? { gridColumn: `span ${span}` } : undefined}>
    <div className="card-head sm"><div><div className="card-title">{title}</div>{sub && <div className="card-sub">{sub}</div>}</div></div>
    {children}
  </div>
);

export default function App() {
  const [mode, setMode] = useState('ops');
  const [health, setHealth] = useState(null);
  const [dash, setDash] = useState(null);
  const [exp, setExp] = useState(null);
  const [samples, setSamples] = useState([]);

  useEffect(() => {
    api.health().then(setHealth);
    api.dashboard().then(setDash);
    api.experiments().then(setExp);
    api.samples().then((d) => setSamples(d?.samples || []));
  }, []);

  return (
    <div className="app">
      <TopBar mode={mode} setMode={setMode} health={health} />
      <main className="main">
        {!dash || !exp ? <div className="dim" style={{ padding: 40 }}>로딩 중…</div> : mode === 'ops'
          ? <Ops dash={dash} exp={exp} samples={samples} />
          : <Analysis exp={exp} dash={dash} />}
      </main>
      <footer className="footer dim sm">Steel Inspection Console · Severstal · DeepLabV3+/EffNet-B3 + 분류 게이트 · 무누수 OOF 0.9532</footer>
    </div>
  );
}

function Ops({ dash, exp, samples }) {
  const det = dash.detection;
  return (
    <>
      <KPICards k={dash.kpis} />
      <div className="grid-main">
        <div className="left-col">
          <SegViewer samples={samples} />
          <Card title="클래스 균형 재학습 — 희귀 클래스 검출률 회복" sub="FocalTversky+오버샘플+pos_weight (실측)">
            <Bars labels={det.labels}
              series={[{ name: '기존', data: det.baseline, color: '#cbd5e1' }, { name: '균형학습', data: det.balanced, color: '#22c55e' }]}
              yMin={0} yMax={100} fmt={(v) => `${v}%`} height={210} />
          </Card>
        </div>
        <div className="right-col">
          <Card title="결함 클래스 분포" sub="train 12,568장 · 클래스별 보유 이미지 수">
            <Donut data={dash.class_dist.map((c) => ({ name: c.name, count: c.count, color: c.color }))} unit="images" />
          </Card>
          <Card title="검사 판정 비율" sub="fold0 검증셋 2,514장">
            <Donut data={[{ name: '정상', count: dash.kpis.normal, color: '#22c55e' }, { name: '결함', count: dash.kpis.defect, color: '#ef4444' }]} unit="caps" />
          </Card>
          <Card title="분류 게이트 — empty 차단율" sub="결함 recall 95% 유지 시 정상 차단(%)">
            <ProgressList items={exp.gate.labels.map((l, i) => ({ label: l, pct: exp.gate.empty_block_at_recall95[i] * 100, text: `${(exp.gate.empty_block_at_recall95[i] * 100).toFixed(1)}%`, color: '#3b82f6' }))} />
          </Card>
        </div>
      </div>
    </>
  );
}

function Analysis({ exp }) {
  const ff = exp.fivefold;
  return (
    <div className="chart-grid">
      <Card title="마일스톤별 mean Dice" sub="단계별 누적 성능">
        <Bars labels={exp.milestones.labels} series={[{ name: 'Dice', data: exp.milestones.dice, color: '#3b82f6' }]} yMin={0.93} yMax={0.96} height={210} />
      </Card>
      <Card title="레버별 누적 이득" sub="단일→앙상블→정제→게이트">
        <Bars labels={exp.ensemble_gain.labels} series={[{ name: 'Dice', data: exp.ensemble_gain.dice, color: '#22c55e' }]} yMin={0.93} yMax={0.96} height={210} />
      </Card>
      <Card title="per-class Dice — 게이트 전/후" sub="C3(대형/스크래치)에서 개선">
        <Bars labels={exp.per_class_dice.labels}
          series={[{ name: 'ungated', data: exp.per_class_dice.ungated, color: '#94a3b8' }, { name: '+gate', data: exp.per_class_dice.gated, color: '#22c55e' }]}
          yMin={0.8} yMax={1} height={210} />
      </Card>
      <Card title="전처리 교차검증" sub="지도 seg에선 무전처리(control)가 최고">
        <Bars labels={exp.preproc_cross.labels}
          series={[{ name: 'Dice', data: exp.preproc_cross.dice, color: '#f59e0b' }]} yMin={0.94} yMax={0.953} height={210} />
      </Card>
      <Card title="누수 폭로 — patch-split vs image-split" sub="패치 분할이 점수를 부풀림(+1.25%p)">
        <Bars labels={exp.leak.labels}
          series={[{ name: 'AUC', data: exp.leak.auc, color: '#3b82f6' }, { name: 'acc', data: exp.leak.acc, color: '#f59e0b' }, { name: 'F1', data: exp.leak.f1, color: '#22c55e' }]}
          yMin={0.8} yMax={1} height={210} />
      </Card>
      <Card title="분류 게이트 per-class AUROC" sub="멀티라벨 EfficientNet-B3">
        <Bars labels={exp.gate.labels} series={[{ name: 'AUROC', data: exp.gate.auroc, color: '#a855f7' }]} yMin={0.98} yMax={1} height={210} fmt={(v) => v.toFixed(4)} />
      </Card>
      <Card title="5-fold 견고성 (관리도)" sub={`평균 ${ff.mean} · 편차 ±${ff.std}`}>
        <LineChart points={ff.labels.map((l, i) => ({ x: l, y: ff.dice[i] }))} cl={ff.mean} ucl={ff.mean + 0.0015} lcl={ff.mean - 0.0015} color="#3b82f6" />
      </Card>
      <Card title="다른 팀(Vision-Q) 대비" sub="패치 분류 우회 → 본 프로젝트는 세그멘테이션">
        <table className="cmp-table"><tbody>
          <tr><th></th><th>Vision-Q</th><th>본 프로젝트</th></tr>
          {exp.vs_other_team.map((r, i) => <tr key={i}><td className="dim">{r[0]}</td><td>{r[1]}</td><td style={{ color: '#15803d', fontWeight: 600 }}>{r[2]}</td></tr>)}
        </tbody></table>
      </Card>
    </div>
  );
}
