// 백엔드(/api) 우선, 실패 시 정적(BASE/static) 폴백 → GitHub Pages 등 정적 호스팅에서도 동작
const BASE = import.meta.env.BASE_URL || '/';   // 예: '/' (서버) 또는 '/steel-defect-suite/' (Pages)
const S = (p) => `${BASE}static/${p}`;

async function j(url, fallback) {
  try { const r = await fetch(url); if (r.ok) return await r.json(); } catch {}
  if (fallback) { try { return await (await fetch(fallback)).json(); } catch {} }
  return null;
}
export const api = {
  health: () => j('/api/health'),
  samples: () => j('/api/samples', S('samples/samples.json')),
  experiments: () => j('/api/experiments', S('experiments.json')),
  dashboard: () => j('/api/dashboard', S('dashboard.json')),
  inferSample: (id, p) => {
    const q = new URLSearchParams({ id, ...p }).toString();
    return j(`/api/infer_sample?${q}`);
  },
  inferUpload: async (file, p) => {
    const fd = new FormData(); fd.append('file', file);
    Object.entries(p).forEach(([k, v]) => fd.append(k, v));
    try { const r = await fetch('/api/infer', { method: 'POST', body: fd }); return await r.json(); }
    catch { return { available: false }; }
  },
};
export const sampleUrl = (f) => S(`samples/${f}`);
