# 🤖 Inbox Agent + LLM

An AI-powered FAQ answering agent built with **FastAPI**, **scikit-learn**, and **OpenAI GPT**.  
It lets you upload your company FAQs in CSV format, and instantly chat with an intelligent assistant that finds and rephrases the best answers.

---

## 🚀 Features

✅ Dual-layer **TF-IDF retrieval** (word + char n-grams)  
✅ **Confidence-based fallback** (“not confident enough” replies)  
✅ Optional **LLM polish** with GPT-4o-mini for friendlier answers  
✅ Interactive **chat UI** (browser-based, responsive design)  
✅ **CSV upload** & reload for new FAQs  
✅ Context-aware follow-up questions (“What about weekends?”)  
✅ Lightweight, no external database required

---

## 🧠 Architecture

```text
[ User ] → [ FastAPI Backend ]
             ↳ FAQ Retrieval (TF-IDF)
             ↳ LLM Rewrite (optional)
             ↳ Context Memory
             ↳ Confidence Filtering
           → [ Web Chat UI (HTML+JS) ]

🧩 Project Structure
inbox-agent-llm/
│
├── app/
│   └── main.py           # FastAPI app & UI
├── data/
│   └── faqs.csv          # Example knowledge base
├── tests/
│   └── test_api.py       # Simple health endpoint test
│
├── .env                  # API keys (not committed)
├── .gitignore
├── requirements.txt
├── README.md
└── .venv/ (ignored)

⚙️ Setup & Run
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/inbox-agent-llm.git
cd inbox-agent-llm

2️⃣ Create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your OpenAI credentials

Create a file named .env in the project root:

OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini


(Never commit this file to GitHub — it’s listed in .gitignore.)

5️⃣ Run the app
uvicorn app.main:app --reload --port 8001


Then open your browser at
👉 http://127.0.0.1:8001

You’ll see:

LLM: connected ✅ (top badge)

Chat box to type questions

CSV uploader to add or update FAQs

📁 CSV Format

Your knowledge base must be a CSV file with three columns:

question	answer	source
What are your store hours?	We’re open Mon–Sat 10:00–18:00.	internal_faq_v1
How do I return an item?	Email support with your order number to receive a label.	returns.md
🔌 API Routes
Endpoint	Method	Description
/	GET	Web chat interface
/answer	POST	Retrieve best answer (question, top_k, history, polish)
/upload	POST	Upload new FAQ CSV
/reload	POST	Reload the default CSV
/health	GET	Check FAQs loaded
/llm	GET	Check LLM connection
/llm/reload	POST	Reload LLM settings from .env
/debug	POST	Inspect top 5 matches for a query
🧪 Example Request
curl -X POST "http://127.0.0.1:8001/answer" ^
     -H "Content-Type: application/json" ^
     -d "{\"question\": \"What are your store hours?\", \"polish\": true}"


Response:

{
  "query": "what are your store hours",
  "top_answer": {
    "answer": "We’re open Monday to Saturday, from 10:00 to 18:00.",
    "source": "internal_faq_v1",
    "confidence": 0.923
  }
}

🧾 Example Screenshot

(add a screenshot of your chat UI here)


🧪 Testing
pytest -q


This runs simple API tests (health check, etc.).

🧩 Dependencies
Package	Purpose
FastAPI	Backend framework
Uvicorn	ASGI server
scikit-learn	TF-IDF & cosine similarity
pandas	CSV loading
python-dotenv	Load .env file
openai	GPT integration
python-multipart	File uploads
pytest	Testing framework
🛡️ Security Notes

Do not commit your .env or real FAQs with private data.

Use .gitignore (included) to keep credentials safe.

API keys are read once at startup; restart the server after updating.

🧭 Roadmap

 Add support for multiple LLM providers (Anthropic, Ollama)

 Integrate vector embeddings for semantic search

 Add user feedback (👍 / 👎) collection

 Dockerfile for container deployment

 Streamlit dashboard for monitoring

🧑‍💻 Author

Maliheh Tehrani
AI/Data Consultant & Developer — Vienna, Austria
🌐 LinkedIn
 • GitHub

📄 License

MIT License

⭐ If you find this project useful, please give it a star on GitHub!