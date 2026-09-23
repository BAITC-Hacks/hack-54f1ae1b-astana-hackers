import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts'
import './style.css'

const api = async (path, options) => {
  const response = await fetch(path, options)
  if (!response.ok) throw new Error(await response.text())
  return response.json()
}

function App() {
  const [mode, setMode] = useState('validation')
  const [site, setSite] = useState('turbine_1')
  const [summary, setSummary] = useState({})
  const [rows, setRows] = useState([])
  const [date, setDate] = useState('2026-01-20')
  const [logs, setLogs] = useState([])
  const [question, setQuestion] = useState('Почему модель ошиблась 2026-01-20?')
  const [chat, setChat] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => { api('/api/summary').then(setSummary).catch(e => setError(e.message)) }, [])
  useEffect(() => {
    api(`/api/forecast?mode=${mode}&site=${site}`).then(setRows).catch(e => setError(e.message))
    setDate(mode === 'validation' ? '2026-01-20' : '2026-02-10')
  }, [mode, site])
  useEffect(() => {
    api(`/api/logs?mode=${mode}&site=${site}&date=${date}`).then(setLogs).catch(e => setError(e.message))
  }, [mode, site, date])

  const daily = useMemo(() => {
    const groups = {}
    rows.forEach(row => {
      const day = row.valid_time.slice(0, 10)
      if (!groups[day]) groups[day] = { day, prediction: 0, operational: 0, baseline: 0, actual: 0, n: 0, actualN: 0 }
      groups[day].prediction += Number(row.prediction)
      groups[day].operational += Number(row.operational_prediction ?? row.prediction)
      groups[day].baseline += Number(row.baseline)
      groups[day].n += 1
      if (row.power !== null && row.power !== undefined) {
        groups[day].actual += Number(row.power)
        groups[day].actualN += 1
      }
    })
    return Object.values(groups).sort((a, b) => a.day.localeCompare(b.day)).map(d => ({
      day: d.day, prediction: +(d.prediction / d.n).toFixed(3), operational: +(d.operational / d.n).toFixed(3),
      baseline: +(d.baseline / d.n).toFixed(3),
      actual: d.actualN ? +(d.actual / d.actualN).toFixed(3) : null
    }))
  }, [rows])
  const metrics = summary.validation?.metrics?.[site]
  const send = async event => {
    event.preventDefault()
    setChat(null)
    try { setChat(await api('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, mode }) })) }
    catch (e) { setError(e.message) }
  }

  return <main>
    <header><div><span className="eyebrow">WIND OPERATIONS / AGENT TRACE</span><h1>Прогноз выработки ВЭС</h1><p>Почасовой прогноз, проверка на факте и решения агента</p></div><div className="badge">● Single Runs API</div></header>
    {error && <aside className="error">{error}</aside>}
    <section className="controls"><label>Режим<select value={mode} onChange={e => setMode(e.target.value)}><option value="validation">Январь · self-test</option><option value="production">Февраль · прогноз</option></select></label><label>Турбина<select value={site} onChange={e => setSite(e.target.value)}><option value="turbine_1">Турбина 1</option><option value="turbine_2">Турбина 2</option></select></label><label>Дата лога<input type="date" value={date} onChange={e => setDate(e.target.value)}/></label></section>
    <section className="cards"><article><small>ML MAE · январь</small><strong>{metrics?.ml_mae?.toFixed(4) ?? '—'}</strong><span>нормализованная мощность</span></article><article><small>Power curve MAE</small><strong>{metrics?.baseline_mae?.toFixed(4) ?? '—'}</strong><span>ветер → мощность</span></article><article><small>Рабочая модель</small><strong>{summary.model_selection?.[site] === 'baseline' ? 'Кривая' : 'ML'}</strong><span>выбрана по январю</span></article><article><small>Февральский факт</small><strong>Нет</strong><span>метрика не заявляется</span></article></section>
    <section className="panel"><div className="panelhead"><h2>Прогноз и факт по дням</h2><p>Средняя нормализованная мощность; факт доступен только в январе</p></div><div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={daily}><CartesianGrid stroke="#dfe5e0" strokeDasharray="3 3"/><XAxis dataKey="day" tick={{fontSize:11}}/><YAxis domain={[0,1]} tick={{fontSize:11}}/><Tooltip/><Legend/><Line type="monotone" dataKey="operational" name="Рабочий прогноз" stroke="#146b54" strokeWidth={3} dot={false}/><Line type="monotone" dataKey="prediction" name="LightGBM" stroke="#82ae90" strokeWidth={1.5} dot={false}/><Line type="monotone" dataKey="baseline" name="Power curve" stroke="#d5a94f" strokeWidth={2} dot={false}/><Line type="monotone" dataKey="actual" name="Факт" stroke="#223047" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer></div></section>
    <div className="columns"><section className="panel"><div className="panelhead"><h2>Шаги агента · {date}</h2><p>Вход, решение и результат каждого вызова</p></div><div className="loglist">{logs.length ? logs.map((row, i) => <article key={i}><div><b>{row.tool}</b><time>{row.as_of}</time></div><p>{row.decision}</p><details><summary>Детали</summary><pre>{JSON.stringify({input:row.input, output:row.output},null,2)}</pre></details></article>) : <p>Для этой даты записей нет.</p>}</div></section><section className="panel"><div className="panelhead"><h2>Спросить агента</h2><p>Ответ опирается на инструменты и текущие файлы</p></div><form onSubmit={send}><textarea value={question} onChange={e => setQuestion(e.target.value)}/><button>Спросить</button></form>{chat && <div className="answer"><p>{chat.answer}</p><small>Вызваны: {chat.tool_calls?.join(', ') || 'нет'}</small></div>}</section></div>
    <footer>Момент погодного запуска и lead_hours сохранены для каждого почасового прогноза в данных проекта.</footer>
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
