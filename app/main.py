# app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
import os, re, json
from datetime import datetime, timezone

# Optional: load .env (OPENAI_API_KEY, OPENAI_MODEL)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded OPENAI key prefix:", os.getenv("OPENAI_API_KEY")[:7])
except Exception:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Optional OpenAI client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None  # graceful fallback

app = FastAPI(title="Inbox→Answer Agent (Chat + LLM polish + Feedback)")

# ---------- Config ----------
DATA_PATH = "data/faqs.csv"
FEEDBACK_PATH = "data/feedback.jsonl"
MIN_CONFIDENCE = 0.16  # "don't guess" threshold

# ---------- Globals ----------
faq_df: Optional[pd.DataFrame] = None
word_vectorizer: Optional[TfidfVectorizer] = None
char_vectorizer: Optional[TfidfVectorizer] = None
word_tfidf = None
char_tfidf = None

# ---------- Utilities ----------
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}")

def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.lower()
    t = PUNCT_RE.sub(" ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t

def redact_pii(text: str) -> str:
    if not text:
        return text
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text

# tiny synonym expander
SYNONYMS = {
    "open": "opening hours store hours hours",
    "opening": "opening hours store hours hours",
    "hours": "opening hours store hours hours",
    "time": "opening hours store hours hours",
    "return": "returns refund exchange",
    "returns": "return refund exchange",
    "warranty": "guarantee warranty period",
    "guarantee": "warranty guarantee period",
    "assembly": "assembly assemble installation install setup",
    "assemble": "assembly assemble installation install setup",
    "installation": "assembly assemble installation install setup",
    "install": "assembly assemble installation install setup",
    "setup": "assembly assemble installation install setup",
    "delivery": "delivery shipping schedule reschedule change date",
    "shipping": "delivery shipping schedule reschedule",
    "reschedule": "delivery reschedule change date",
}

def expand_query(q: str) -> str:
    qn = normalize_text(q)
    tokens = qn.split()
    extra = []
    for tok in tokens:
        if tok in SYNONYMS:
            extra.append(SYNONYMS[tok])
    if extra:
        qn = qn + " " + " ".join(extra)
    return qn

def build_query_with_context(question: str, history: Optional[List[str]]) -> str:
    """
    If user sends a short follow-up like 'what about weekends?',
    prepend up to the last 2 user turns so retrieval has context.
    """
    history = (history or [])[-2:]
    if not history:
        return question
    short = len(question) < 40 or any(w in question.lower() for w in ["it", "that", "they", "weekend", "price", "there"])
    return (" ".join(history) + " " + question) if short else question

# ---------- Data / Index ----------
def load_faqs(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"question", "answer", "source"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must have columns: question, answer, source")
    for col in ["question", "answer", "source"]:
        df[col] = df[col].fillna("").astype(str)
    df["question_norm"] = df["question"].apply(normalize_text)
    return df

def build_index(df: pd.DataFrame) -> None:
    """Build both word-level and char-level TF-IDF indices over normalized questions."""
    global word_vectorizer, char_vectorizer, word_tfidf, char_tfidf
    word_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    word_tfidf = word_vectorizer.fit_transform(df["question_norm"])
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    char_tfidf = char_vectorizer.fit_transform(df["question_norm"])

def reload_faqs(path: str = DATA_PATH) -> None:
    global faq_df
    faq_df = load_faqs(path)
    build_index(faq_df)

# ---------- Models ----------
class AnswerItem(BaseModel):
    answer: str
    source: str
    confidence: float

class AnswerResponse(BaseModel):
    query: str
    top_answer: AnswerItem
    alternatives: List[AnswerItem] = []

class Query(BaseModel):
    question: str
    top_k: int = 3
    polish: bool = False
    history: Optional[List[str]] = None  # last few user messages for context

class Feedback(BaseModel):
    question: str
    answer: str
    source: str
    confidence: float
    helpful: bool

# ---------- Optional LLM polish ----------
def rewrite_with_llm(question: str, answer: str, source: str) -> str:
    if not openai_client or not OPENAI_API_KEY:
        return answer
    try:
        prompt = (
            "You are a careful customer support assistant. "
            "Rewrite the given ANSWER in clear, friendly English for the given QUESTION. "
            "Do not add any information not already present in the ANSWER. "
            "If the QUESTION asks for something beyond the ANSWER's content, say you don't know. "
            "Keep it short. Preserve any specific actions. Avoid hallucinations."
        )
        user = f"QUESTION: {question}\n\nANSWER:\n{answer}\n\nSOURCE: {source}"
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=220,
            timeout=15,
        )
        text = resp.choices[0].message.content.strip()
        return text or answer
    except Exception:
        return answer

# ---------- Lifecycle ----------
@app.on_event("startup")
def on_startup():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(
                "question,answer,source\n"
                "\"What are your store hours?\",\"We’re open Mon–Sat 10:00–18:00.\",internal_faq_v1\n"
                "\"How do I start a return?\",\"Email support with your order number; we’ll send you a return label.\",return_policy.md\n"
                "\"Do you offer assembly service?\",\"Yes, within Vienna; pricing depends on item size.\",services_page.html\n"
                "\"How long is the warranty?\",\"Standard warranty is 24 months unless specified otherwise.\",warranty_policy.pdf\n"
                "\"Can I change my delivery date?\",\"Yes—contact support at least 48 hours before your delivery.\",delivery_tos.md\n"
            )
    reload_faqs(DATA_PATH)

# ---------- Chat UI ----------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Inbox→Answer Agent (Chat)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="icon" type="image/svg+xml" href="/favicon.ico">
  <style>
    :root{--bg:#f9fafb;--card:#ffffff;--muted:#6b7280;--border:#e5e7eb;--me:#111827;--bot:#2563eb}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);
         max-width:960px;margin:24px auto;padding:0 16px}
    header{display:flex;align-items:center;gap:12px;margin:16px 0}
    h1{font-size:1.25rem;margin:0}
    .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.8rem;border:1px solid var(--border)}
    .ok{background:#ecfdf5;border-color:#10b981;color:#065f46}
    .no{background:#fef2f2;border-color:#ef4444;color:#7f1d1d}
    .wrap{display:grid;grid-template-columns:1fr 300px;gap:16px}
    @media (max-width:900px){.wrap{grid-template-columns:1fr}}
    .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px}
    .chat{height:60vh;overflow:auto;display:flex;flex-direction:column;gap:10px;padding:8px}
    .msg{max-width:80%;padding:10px 12px;border-radius:14px;border:1px solid var(--border)}
    .me{align-self:flex-end;background:#eef2ff;border-color:#c7d2fe}
    .bot{align-self:flex-start;background:#f8fafc;border-color:#e5e7eb}
    .muted{color:var(--muted);font-size:.9rem}
    .answer{white-space:pre-wrap}
    .inputbar{display:flex;gap:8px;align-items:flex-end;margin-top:8px}
    textarea{flex:1;min-height:44px;max-height:160px;padding:10px;border:1px solid var(--border);border-radius:10px;resize:vertical}
    button{height:36px;padding:0 10px;border:0;border-radius:8px;background:var(--me);color:white;cursor:pointer}
    .row{display:flex;gap:8px;align-items:center;margin:8px 0}
    .tinybtn{height:26px;padding:0 8px;border-radius:6px;background:#374151;color:#fff;border:0;cursor:pointer}
  </style>
</head>
<body>
  <header>
    <h1>📬 Inbox→Answer Agent</h1>
    <span id="llm-badge" class="badge no">LLM: checking…</span>
    <span id="health" class="muted"></span>
  </header>

  <div class="wrap">
    <section class="card">
      <div id="chat" class="chat"></div>
      <div class="row">
        <label style="display:inline-flex;align-items:center;gap:6px">
          <input type="checkbox" id="polish" />
          Use LLM polish
        </label>
        <button id="newchat" class="tinybtn">New chat</button>
        <a href="/review" class="tinybtn" style="text-decoration:none;display:inline-flex;align-items:center">Review feedback</a>
      </div>
      <div class="inputbar">
        <textarea id="q" placeholder="Ask something… (Shift+Enter for newline)"></textarea>
        <button id="send">Send</button>
      </div>
      <div class="muted" style="margin-top:6px">Tip: ask a follow-up like “what about weekends?” — the bot remembers context in this tab.</div>
    </section>

    <aside class="card">
      <h3>Knowledge base</h3>
      <form id="upload-form">
        <input type="file" id="csvfile" accept=".csv" />
        <div style="margin-top:8px">
          <button type="submit">Upload & Reload</button>
          <span id="upload-status" class="muted"></span>
        </div>
      </form>
      <p class="muted">CSV must have columns: <code>question, answer, source</code></p>
    </aside>
  </div>

  <script>
    const chatEl = document.getElementById('chat');
    const qEl = document.getElementById('q');
    const sendBtn = document.getElementById('send');
    const newBtn = document.getElementById('newchat');
    const polishEl = document.getElementById('polish');
    const healthEl = document.getElementById('health');
    const llmBadge = document.getElementById('llm-badge');
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');

    let messages = []; // {role:'user'|'assistant', text:string}
    let msgSeq = 0;
    const answerPayloads = {}; // id -> {question, answer, source, confidence}

    function scrollBottom(){ chatEl.scrollTop = chatEl.scrollHeight; }

    function esc(s){
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    function bubble(role, text, trusted=false){
      const div = document.createElement('div');
      div.className = 'msg ' + (role === 'user' ? 'me' : 'bot');
      const inner = document.createElement('div');
      inner.className = 'answer';
      if (trusted) {
        inner.innerHTML = text; // trusted small HTML from backend/our code
      } else {
        inner.textContent = text; // escape untrusted strings
      }
      div.appendChild(inner);
      chatEl.appendChild(div);
      scrollBottom();
      return inner; // return the node for optional post-processing
    }

    function lastUserTexts(n=4){
      return messages.filter(m=>m.role==='user').slice(-n).map(m=>m.text);
    }

    async function fetchHealth(){
      try{
        const r=await fetch('/health'); if(!r.ok) return;
        const j=await r.json(); healthEl.textContent = '• FAQs loaded: '+j.faqs_loaded;
      }catch(e){ healthEl.textContent = '• API not reachable'; }
    }
    async function fetchLLM(){
      try{
        const r=await fetch('/llm'); if(!r.ok) return;
        const j=await r.json();
        if(j.available){ llmBadge.textContent='LLM: connected'; llmBadge.className='badge ok'; polishEl.checked=true; }
        else{ llmBadge.textContent='LLM: not set'; llmBadge.className='badge no'; }
      }catch(e){ llmBadge.textContent='LLM: error'; llmBadge.className='badge no'; }
    }

    async function sendMessage(){
      const text = qEl.value.trim();
      if(!text) return;
      qEl.value = '';
      bubble('user', text);
      messages.push({role:'user', text});

      sendBtn.disabled = true; sendBtn.textContent = '…';
      try{
        const payload = { question: text, top_k: 3, polish: !!polishEl.checked, history: lastUserTexts(4) };
        const r = await fetch('/answer', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
        const raw = await r.text();
        let j=null; try{ j=JSON.parse(raw);}catch(err){ bubble('assistant','Unexpected server response.'); return; }
        if(!r.ok){ bubble('assistant', j.detail || 'API error.'); return; }
        const top = j.top_answer;
        const unsure = (top.source === 'unknown');

        if (unsure) {
          bubble('assistant', 'Not confident enough. ' + top.answer, false);
        } else {
          const id = String(++msgSeq);
          // store payload for feedback
          answerPayloads[id] = { question: text, answer: top.answer, source: top.source, confidence: top.confidence };

          const safeAnswer = esc(top.answer).replace(/\\n/g, '<br>');
          const meta = '<span class="muted">Source: ' + esc(top.source) + ' • Confidence: ' + Number(top.confidence).toFixed(3) + '</span>';
          const controls = `
            <span style="margin-left:10px">
              <button class="fb" data-id="${id}" data-helpful="true" style="height:24px;padding:0 8px;border-radius:6px;background:#10b981;border:0;color:white;cursor:pointer">👍</button>
              <button class="fb" data-id="${id}" data-helpful="false" style="height:24px;padding:0 8px;border-radius:6px;background:#ef4444;border:0;color:white;cursor:pointer">👎</button>
            </span>`;
          const replyHTML = safeAnswer + '<br><br>' + meta + controls;

          bubble('assistant', replyHTML, true);
          messages.push({role:'assistant', text: top.answer});
        }
      }catch(e){
        bubble('assistant', 'Could not reach /answer.');
      }finally{
        sendBtn.disabled = false; sendBtn.textContent = 'Send';
        qEl.focus();
      }
    }

    // Delegate feedback click events
    chatEl.addEventListener('click', async (ev)=>{
      const btn = ev.target.closest('button.fb');
      if(!btn) return;
      const id = btn.dataset.id;
      const helpful = btn.dataset.helpful === 'true';
      const p = answerPayloads[id];
      if(!p) return;
      btn.disabled = true;
      try{
        await fetch('/feedback', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({
            question: p.question,
            answer: p.answer,
            source: p.source,
            confidence: p.confidence,
            helpful: helpful
          })
        });
        // Replace both buttons in this controls span with a thank-you
        const parentSpan = btn.parentElement;
        if (parentSpan) { parentSpan.innerHTML = '<span class="muted">Thanks for the feedback!</span>'; }
      }catch(e){
        btn.disabled = false;
        alert('Could not record feedback.');
      }
    });

    // events
    sendBtn.addEventListener('click', sendMessage);
    qEl.addEventListener('keydown', (ev)=>{ if(ev.key==='Enter' && !ev.shiftKey){ ev.preventDefault(); sendMessage(); } });
    newBtn.addEventListener('click', ()=>{
      messages = [];
      chatEl.innerHTML='';
      bubble('assistant', 'New chat started. Ask me anything from your FAQs!');
      qEl.focus();
    });

    uploadForm.addEventListener('submit', async (e)=>{
      e.preventDefault();
      const f = document.getElementById('csvfile').files[0];
      uploadStatus.textContent = '';
      if(!f){ uploadStatus.textContent='Choose a CSV first.'; return; }
      const form = new FormData(); form.append('file', f, 'faqs.csv');
      try{
        const r = await fetch('/upload', { method:'POST', body: form });
        const j = await r.json();
        if(r.ok){ uploadStatus.textContent = 'Uploaded • FAQs: '+j.faqs_loaded; }
        else{ uploadStatus.textContent = 'Error: '+(j.detail || 'upload failed'); }
      }catch(e){ uploadStatus.textContent = 'Network error'; }
      fetchHealth();
    });

    // init
    bubble('assistant', 'Hi! I can answer based on your FAQs. Ask away.');
    fetchHealth();
    (async()=>{ try{ const r=await fetch('/llm'); const j=await r.json(); if(j.available){ llmBadge.textContent='LLM: connected'; llmBadge.className='badge ok'; polishEl.checked=true; } else { llmBadge.textContent='LLM: not set'; llmBadge.className='badge no'; } }catch(e){ llmBadge.textContent='LLM: error'; llmBadge.className='badge no'; }})();
  </script>
</body>
</html>
    """

# ---------- API ----------
@app.get("/health")
def health():
    return {"status": "ok", "faqs_loaded": len(faq_df) if faq_df is not None else 0}

@app.get("/llm")
def llm_available():
    key_present = bool(OPENAI_API_KEY)
    client_present = openai_client is not None
    return {
        "available": key_present and client_present,
        "key_present": key_present,
        "client_present": client_present,
    }


@app.post("/answer", response_model=AnswerResponse)
def answer(q: Query):
    if any(x is None for x in [faq_df, word_vectorizer, char_vectorizer, word_tfidf, char_tfidf]):
        raise HTTPException(status_code=500, detail="Index not ready")

    safe_question = redact_pii(q.question)
    contextual = build_query_with_context(safe_question, q.history)
    query_for_match = expand_query(contextual)

    w_vec = word_vectorizer.transform([query_for_match])
    c_vec = char_vectorizer.transform([query_for_match])

    w_sims = cosine_similarity(w_vec, word_tfidf).ravel()
    c_sims = cosine_similarity(c_vec, char_tfidf).ravel()
    sims = (w_sims + c_sims) / 2.0

    if sims.size == 0:
        raise HTTPException(status_code=404, detail="No FAQs loaded")

    top_idx = sims.argsort()[::-1][: max(1, q.top_k)]
    items: List[AnswerItem] = []
    for i in top_idx:
        items.append(AnswerItem(
            answer=faq_df.iloc[i]["answer"],
            source=faq_df.iloc[i]["source"],
            confidence=float(round(sims[i], 4))
        ))
    top = items[0]

    if top.confidence < MIN_CONFIDENCE:
        top = AnswerItem(
            answer=("I’m not fully confident I have the right answer yet. "
                    "Please add a keyword like 'warranty', 'delivery', 'return', 'assembly'."),
            source="unknown",
            confidence=float(top.confidence)
        )
        return AnswerResponse(query=safe_question, top_answer=top, alternatives=[])

    if q.polish:
        polished = rewrite_with_llm(safe_question, top.answer, top.source)
        top = AnswerItem(answer=polished, source=top.source, confidence=top.confidence)

    return AnswerResponse(query=safe_question, top_answer=top, alternatives=items[1:])

@app.post("/reload")
def reload_endpoint():
    try:
        reload_faqs(DATA_PATH)
        return {"status": "reloaded", "faqs_loaded": len(faq_df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)
        contents = await file.read()
        with open(DATA_PATH, "wb") as f:
            f.write(contents)
        reload_faqs(DATA_PATH)
        return JSONResponse({"status": "uploaded", "faqs_loaded": len(faq_df)})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- Feedback endpoints ----------
@app.post("/feedback")
def feedback_endpoint(fb: Feedback):
    try:
        os.makedirs("data", exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": fb.question,
            "answer": fb.answer,
            "source": fb.source,
            "confidence": float(fb.confidence),
            "helpful": bool(fb.helpful),
        }
        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/review", response_class=HTMLResponse)
def review_feedback():
    rows = []
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
    rows = rows[-200:]
    # Simple table
    html = ["<!doctype html><meta charset='utf-8'><title>Feedback review</title>"]
    html.append("<style>body{font-family:system-ui;max-width:1000px;margin:24px auto;padding:0 16px} table{width:100%;border-collapse:collapse} th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:left} .muted{color:#6b7280}</style>")
    html.append("<h2>Feedback (latest 200)</h2>")
    html.append("<table><tr><th>ts</th><th>helpful</th><th>confidence</th><th>source</th><th>question</th><th>answer</th></tr>")
    for r in rows:
        html.append(f"<tr><td class='muted'>{r.get('ts','')}</td><td>{'👍' if r.get('helpful') else '👎'}</td><td>{r.get('confidence','')}</td><td>{r.get('source','')}</td><td>{r.get('question','')}</td><td>{r.get('answer','')}</td></tr>")
    html.append("</table><p class='muted'>File: data/feedback.jsonl</p>")
    return "\n".join(html)

# Debug endpoint to inspect top matches
@app.post("/debug")
def debug(q: Query):
    if any(x is None for x in [faq_df, word_vectorizer, char_vectorizer, word_tfidf, char_tfidf]):
        raise HTTPException(status_code=500, detail="Index not ready")
    safe_question = redact_pii(q.question)
    contextual = build_query_with_context(safe_question, q.history)
    query_for_match = expand_query(contextual)
    w_vec = word_vectorizer.transform([query_for_match])
    c_vec = char_vectorizer.transform([query_for_match])
    w_sims = cosine_similarity(w_vec, word_tfidf).ravel()
    c_sims = cosine_similarity(c_vec, char_tfidf).ravel()
    sims = (w_sims + c_sims) / 2.0
    top_idx = sims.argsort()[::-1][:5]
    rows = []
    for i in top_idx:
        rows.append({
            "i": int(i),
            "score": float(round(sims[i], 4)),
            "question": faq_df.iloc[i]["question"],
            "source": faq_df.iloc[i]["source"]
        })
    return {"query_used": query_for_match, "top5": rows}

# Favicon (📬 SVG) to avoid 404s
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg_icon = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
      <rect width="64" height="64" rx="12" ry="12" fill="#111827"/>
      <text x="50%" y="54%" font-size="40" text-anchor="middle" dominant-baseline="middle">📬</text>
    </svg>
    """
    return Response(content=svg_icon.strip(), media_type="image/svg+xml")
