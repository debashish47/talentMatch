import { useState } from 'react';
import { api } from './api';

export default function Chat() {
  const [messages, setMessages] = useState([{ role: 'assistant', text: 'Hi! Ask me about open roles, required skills, locations, or which job best fits you.' }]);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const ask = async (event) => {
    event.preventDefault();
    const question = message.trim();
    if (!question || loading) return;
    setMessages(items => [...items, { role: 'user', text: question }]);
    setMessage(''); setLoading(true);
    try {
      const result = await api('/chat', { method: 'POST', body: JSON.stringify({ message: question }) });
      setMessages(items => [...items, { role: 'assistant', text: result.answer, sources: result.sources, cache: result.cache_hit }]);
    } catch (error) { setMessages(items => [...items, { role: 'assistant', text: error.message, error: true }]); }
    finally { setLoading(false); }
  };
  return <section className="page narrow chat-page"><p className="eyebrow">RAG JOB ASSISTANT</p><h1>Ask about your next role.</h1><p className="lead">Answers are grounded in currently open job listings and personalized with your saved profile.</p><div className="chat-window">{messages.map((item, index) => <div className={`chat-message ${item.role}${item.error ? ' error' : ''}`} key={index}><p>{item.text}</p>{item.cache && <small>⚡ Answered from semantic cache</small>}{item.sources?.length > 0 && <div className="chat-sources">Sources: {item.sources.map(source => <span key={source.job_id}>{source.title} · {source.location}</span>)}</div>}</div>)}{loading && <div className="chat-message assistant"><p>Searching open jobs and preparing an answer…</p></div>}</div><form className="chat-input" onSubmit={ask}><textarea value={message} onChange={event => setMessage(event.target.value)} placeholder="Example: Which open role is best for Python and FastAPI?"/><button disabled={loading}>{loading ? 'Thinking…' : 'Ask assistant'}</button></form></section>;
}
