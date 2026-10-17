'use strict';
const $ = id => document.getElementById(id);
let polling, lastResult;
const report = message => { $('status').textContent = message; };
async function api(path, options = {}) {
  const headers = { ...options.headers };
  const key = $('api-key').value.trim();
  if (key) headers.Authorization = `Bearer ${key}`;
  const response = await fetch(path, { ...options, headers });
  let body;
  try { body = await response.json(); } catch { throw new Error('服务暂不可用，请稍后重试。'); }
  if (response.status === 401) throw new Error('访问密钥无效。请填写 Render 服务 Environment 中的 API_KEY，不是账号 API Keys 中的 rnd_ 密钥。');
  if (!response.ok) throw new Error(body.msg || body.message || `请求失败 (${response.status})`);
  return body;
}
function busy(value) { $('submit').disabled = value; $('convert-json').disabled = value; }
function stop() { clearTimeout(polling); busy(false); }
async function query(taskId) {
  clearTimeout(polling);
  try {
    const body = await api(`/tasks/${encodeURIComponent(taskId)}`);
    if (body.status === 'running') {
      busy(true);
      report(`预测计算中，因素分析可能需要较长时间。任务：${taskId}`);
      polling = setTimeout(() => query(taskId), 5000);
    } else {
      stop();
      if (body.status === 'error') throw new Error(body.result.error || '预测失败');
      display(body.result);
      report(`预测完成，共 ${body.result.data.length} 天。任务：${taskId}`);
    }
  } catch (error) { stop(); report(error.message); }
}
function saveJson(data, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const link = document.createElement('a'); link.href=url; link.download=filename; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function readForecastFile() {
  const file = $('data-file').files[0];
  if (!file) throw new Error('请先上传 Excel 或 JSON 历史数据文件。');
  if (file.size > 8 * 1024 * 1024) throw new Error('文件不能超过 8 MB。');
  const horizon = Number($('horizon').value);
  if (/\.xlsx?$/i.test(file.name)) {
    report('正在读取 Excel、检查数据并转换为 JSON…');
    const form = new FormData(); form.append('file',file); form.append('id',String(horizon));
    const converted = await api('/convert',{method:'POST',body:form});
    const summary = converted.summary;
    $('file-info').textContent = `已读取 ${summary.total_days} 天，使用最近 ${summary.used_days} 天：${summary.start_date} 至 ${summary.end_date}`;
    return converted.request;
  }
  if (!/\.json$/i.test(file.name)) throw new Error('请选择 .xlsx、.xls 或 .json 文件。');
  let data;
  try { data = JSON.parse(await file.text()); } catch { throw new Error('JSON 文件格式无效。'); }
  if (!data || data.id !== horizon) throw new Error('文件中的 id 与所选预测周期不一致。');
  return data;
}
$('forecast-form').addEventListener('submit', async event => {
  event.preventDefault(); busy(true);
  try {
    const data = await readForecastFile();
    report('数据已读取，正在提交预测任务…');
    const task = await api('/tasks', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data) });
    $('task-id').value = task.task_id;
    query(task.task_id);
  } catch (error) { stop(); report(error.message); }
});
$('convert-json').addEventListener('click', async () => {
  busy(true);
  try {
    const data = await readForecastFile();
    saveJson(data,`coal-input-${data.id}.json`);
    report(`JSON 已转换并下载，包含 ${data.price_data.length} 天历史数据。尚未提交预测。`);
  } catch (error) { report(error.message); }
  finally { busy(false); }
});
$('lookup').addEventListener('click', () => {
  const id = $('task-id').value.trim();
  if (id) query(id); else report('请输入任务编号。');
});
$('data-file').addEventListener('change', () => {
  const file = $('data-file').files[0];
  $('file-info').textContent = file ? `${file.name} · ${(file.size / 1024).toFixed(0)} KB` : '尚未选择文件';
});
function display(result) {
  if (!Array.isArray(result.data) || !result.data.length || result.data.some(row => !Number.isFinite(row.predict_price))) throw new Error('返回的预测数据不完整。');
  lastResult = result;
  const rows = result.data;
  const prices = rows.map(row => row.predict_price);
  $('first').textContent = prices[0].toFixed(2);
  $('last').textContent = prices.at(-1).toFixed(2);
  $('average').textContent = (prices.reduce((a, b) => a + b, 0) / prices.length).toFixed(2);
  $('rows').replaceChildren(...rows.map(row => {
    const tr = document.createElement('tr');
    for (const value of [row.date, row.predict_price.toFixed(2)]) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
    return tr;
  }));
  $('factors').replaceChildren(...Object.entries(rows[0].factor_analysis || {}).sort((a,b) => b[1] - a[1]).slice(0,8).map(([name, value]) => {
    const li = document.createElement('li'); li.textContent = `${name} · ${Number(value).toFixed(4)}`; return li;
  }));
  draw(rows);
  $('download').disabled = false;
}
function draw(rows) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 700 260');
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', '每日煤价预测曲线，完整数值见下方表格');
  const prices = rows.map(row => row.predict_price);
  const bottom = Math.min(...prices), top = Math.max(...prices), span = top - bottom || 1;
  const low = bottom - span * .12, high = top + span * .12;
  const x = i => 60 + i * 620 / Math.max(rows.length - 1, 1);
  const y = value => 220 - (value - low) * 200 / (high - low);
  function element(tag, attrs, text) { const el = document.createElementNS(ns,tag); for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v); if(text !== undefined) el.textContent=text; svg.append(el); return el; }
  for(let i=0;i<=4;i++) { const value=low+(high-low)*i/4, pos=y(value); element('line',{x1:60,x2:680,y1:pos,y2:pos,stroke:'#e8eee2'}); element('text',{x:50,y:pos+4,'text-anchor':'end',fill:'#8a987e','font-size':10},value.toFixed(0)); }
  const points=prices.map((p,i)=>`${x(i)},${y(p)}`).join(' ');
  element('polygon',{points:`60,220 ${points} ${x(prices.length-1)},220`,fill:'#eaf1e3'});
  element('polyline',{points,fill:'none',stroke:'#517a4f','stroke-width':2.5});
  for(const i of new Set([0,Math.floor((rows.length-1)/2),rows.length-1])) element('text',{x:x(i),y:246,'text-anchor':i===0?'start':i===rows.length-1?'end':'middle',fill:'#8a987e','font-size':10},rows[i].date);
  $('chart').replaceChildren(svg);
}
$('download').addEventListener('click', () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult,null,2)],{type:'application/json'}));
  const link = document.createElement('a'); link.href=url; link.download=`coal-forecast-${lastResult.id}.json`; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
fetch('/health').then(response => { if(!response.ok) throw new Error(); $('health').textContent='● 服务就绪'; }).catch(()=> { $('health').textContent='○ 服务暂不可用'; });
