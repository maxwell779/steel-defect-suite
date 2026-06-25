import React, { useEffect, useState } from 'react';
import TopBar from './components/TopBar.jsx';
import KPICards from './components/KPICards.jsx';
import SegViewer from './components/SegViewer.jsx';
import { Donut, Bars, ProgressList, LineChart } from './charts.jsx';
import { api } from './api/client.js';

const Card = ({ title, sub, desc, children, span }) => (
  <div className="card" style={span ? { gridColumn: `span ${span}` } : undefined}>
    <div className="card-head sm"><div><div className="card-title">{title}</div>{sub && <div className="card-sub">{sub}</div>}</div></div>
    {children}
    {desc && <p className="card-desc">{desc}</p>}
  </div>
);

const CLASS_GUIDE = [
  { c: 'C1', color: '#ef4444', pct: '13%', note: '희소 결함. 초기 학습에서 검출 0%로 붕괴 → 균형 재학습으로 83% 회복.' },
  { c: 'C2', color: '#22c55e', pct: '3%', note: '가장 희소(약 247장). 0%→94% 회복. 게이트로 빈 마스크 FP 억제.' },
  { c: 'C3', color: '#3b82f6', pct: '73%', note: '최다 결함(대형·스크래치성). 면적이 커 Dice 천장이 낮음(0.87대).' },
  { c: 'C4', color: '#f59e0b', pct: '11%', note: '비교적 안정적으로 검출(0.98대).' },
];

function ClassGuide() {
  return (
    <Card title="결함 클래스 안내 (C1~C4)" sub="Severstal 결함 4종 — 공식 명칭은 비공개(익명), 분포·특성만 공개"
      desc="평가지표는 (이미지×클래스) mean Dice이며 85.9%가 빈 마스크다. 따라서 희소 클래스(C1·C2) 검출과 정상 영역의 빈 마스크 유지(FP 억제)가 점수를 좌우한다." span={2}>
      <div className="guide-grid">
        {CLASS_GUIDE.map((g) => (
          <div key={g.c} className="guide-cell">
            <div className="gc-h"><span className="gc-sw" style={{ background: g.color }} />{g.c}</div>
            <div className="gc-pct">보유 이미지 비중 {g.pct}</div>
            <div className="gc-note">{g.note}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

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
      <ClassGuide />
      <Card title="마일스톤별 mean Dice" sub="단계별 누적 성능"
        desc="M1 베이스라인(UNet)에서 M5 게이트까지 단계별 검증 Dice. 클래스 균형 재학습(M2)으로 희소 클래스 붕괴를 고친 뒤, 모델·전처리 탐색(M3)·앙상블/정제(M4)·게이트(M5)가 차례로 점수를 올렸다.">
        <Bars labels={exp.milestones.labels} series={[{ name: 'Dice', data: exp.milestones.dice, color: '#3b82f6' }]} yMin={0.93} yMax={0.96} height={210} />
      </Card>
      <Card title="레버별 누적 이득" sub="단일→앙상블→정제→게이트"
        desc="단일 모델(0.9514)에 5모델 앙상블·TTA, 3-임계 마스크 정제, 분류 게이트를 차례로 더해 0.9573에 도달. 각 레버가 독립적으로 누적 기여한다.">
        <Bars labels={exp.ensemble_gain.labels} series={[{ name: 'Dice', data: exp.ensemble_gain.dice, color: '#22c55e' }]} yMin={0.93} yMax={0.96} height={210} />
      </Card>
      <Card title="per-class Dice — 게이트 전/후" sub="분류 게이트의 클래스별 효과"
        desc="분류기가 '해당 클래스 없음'으로 판단하면 세그 마스크를 꺼서 빈 마스크 FP를 줄인다. 면적이 큰 C3에서 가장 크게(+0.006) 오르고, 이미 높은 C2·C4는 거의 유지된다.">
        <Bars labels={exp.per_class_dice.labels}
          series={[{ name: 'ungated', data: exp.per_class_dice.ungated, color: '#94a3b8' }, { name: '+gate', data: exp.per_class_dice.gated, color: '#22c55e' }]}
          yMin={0.8} yMax={1} height={210} />
      </Card>
      <Card title="전처리 교차검증" sub="지도 seg에선 무전처리(control)가 최고"
        desc="상위 전처리 5종(gamma·highpass·clahe·tophat·bilateral_dog)을 승자 모델에 적용했으나 전부 무전처리(0.9514)에 못 미쳤다. 큰 백본이 전처리 이득을 흡수하기 때문 — 비지도(busbar)에선 반대였다. 결론: 원본 입력 사용.">
        <Bars labels={exp.preproc_cross.labels}
          series={[{ name: 'Dice', data: exp.preproc_cross.dice, color: '#f59e0b' }]} yMin={0.94} yMax={0.953} height={210} />
      </Card>
      <Card title="누수 폭로 — patch-split vs image-split" sub="다른 팀 우회의 위험 정량화"
        desc="다른 팀처럼 50% 겹침 패치를 패치 단위로 나누면 인접 패치가 train/test에 동시에 들어가 점수가 +1.25%p 부풀려진다. 본 프로젝트는 이미지 단위 fold로 이 누수를 차단했다.">
        <Bars labels={exp.leak.labels}
          series={[{ name: 'AUC', data: exp.leak.auc, color: '#3b82f6' }, { name: 'acc', data: exp.leak.acc, color: '#f59e0b' }, { name: 'F1', data: exp.leak.f1, color: '#22c55e' }]}
          yMin={0.8} yMax={1} height={210} />
      </Card>
      <Card title="분류 게이트 per-class AUROC" sub="멀티라벨 EfficientNet-B3"
        desc="이미지에 각 클래스 결함이 있는지 판단하는 분류기의 클래스별 판별력. 모두 0.99+로, 결함 recall 95%를 유지하면서 정상 이미지의 빈 마스크를 약 98% 차단할 만큼 강하다.">
        <Bars labels={exp.gate.labels} series={[{ name: 'AUROC', data: exp.gate.auroc, color: '#a855f7' }]} yMin={0.98} yMax={1} height={210} fmt={(v) => v.toFixed(4)} />
      </Card>
      <Card title="5-fold 견고성 (관리도)" sub={`평균 ${ff.mean} · 편차 ±${ff.std}`}
        desc="승자 모델을 5개 fold로 각각 학습해 자기 fold만 평가(OOF)한 결과. 점선은 평균±관리한계. fold 편차 ±0.001로, 단일 fold 점수가 우연이 아니라 일반화된 성능임을 보여준다.">
        <LineChart points={ff.labels.map((l, i) => ({ x: l, y: ff.dice[i] }))} cl={ff.mean} ucl={ff.mean + 0.0015} lcl={ff.mean - 0.0015} color="#3b82f6" />
      </Card>
      <Card title="다른 팀(Vision-Q) 대비" sub="패치 분류 우회 → 본 프로젝트는 세그멘테이션"
        desc="다른 팀은 대회 과제(픽셀 세그멘테이션)를 패치 분류로 우회했다. 본 프로젝트는 원 과제를 풀고, 이미지단위 누수 통제·empty FP 억제·per-class 평가까지 갖췄다.">
        <table className="cmp-table"><tbody>
          <tr><th></th><th>Vision-Q</th><th>본 프로젝트</th></tr>
          {exp.vs_other_team.map((r, i) => <tr key={i}><td className="dim">{r[0]}</td><td>{r[1]}</td><td style={{ color: '#15803d', fontWeight: 600 }}>{r[2]}</td></tr>)}
        </tbody></table>
      </Card>
    </div>
  );
}
