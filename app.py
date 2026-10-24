import html
import json
import random
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "intents.json"

st.set_page_config(
    page_title="Keystone AI | College Enquiry",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_data
def load_data():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def tokenize(text):
    """Simple word tokenization: lower-case and split text into word tokens."""
    return re.findall(r"[a-zA-Z]+(?:['’-][a-zA-Z]+)?|[0-9]+", text.lower())

@st.cache_resource
def train_model():
    data = load_data()
    training_texts, labels = [], []
    for intent in data["intents"]:
        if intent["tag"] == "fallback":
            continue
        for pattern in intent.get("patterns", []):
            training_texts.append(pattern)
            labels.append(intent["tag"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.25))
    ])
    model.fit(training_texts, labels)
    return model

def get_intent_response(user_text):
    data = load_data()
    model = train_model()
    probabilities = model.predict_proba([user_text])[0]
    best_index = probabilities.argmax()
    confidence = float(probabilities[best_index])
    predicted_tag = model.classes_[best_index]

    # Prefer a matching intent from the project data when the classifier is unsure.
    if confidence < 0.20:
        normalized_text = f" {' '.join(tokenize(user_text))} "
        keyword_intents = [
            ("hostel", ("hostel",)),
            ("library", ("library",)),
            ("placement", ("placement", "placements", "internship", "internships", "career", "careers")),
            ("location", ("where", "address", "location", "reach", "find us")),
            ("fees", ("fee", "fees", "cost", "tuition", "payable")),
            ("entrance_exam", ("entrance exam", "mht cet", "mht-cet", "jee main")),
            ("mba_mca", ("mba", "mca", "postgraduate", "post graduate")),
            ("admission", ("admission", "apply", "application", "eligibility", "cap")),
            ("courses", ("course", "courses", "branch", "branches", "degree", "engineering")),
            ("departments", ("department", "departments")),
            ("facilities", ("facility", "facilities", "campus", "laboratory", "laboratories",
                            "lab", "labs", "canteen", "wifi", "wi-fi", "sports", "transport")),
            ("contact", ("contact", "phone", "email", "website")),
            ("timings", ("timing", "timings", "schedule", "opening time")),
        ]
        for tag, keywords in keyword_intents:
            if any(f" {' '.join(tokenize(keyword))} " in normalized_text for keyword in keywords):
                predicted_tag = tag
                break
        else:
            predicted_tag = "fallback"

    intent = next((item for item in data["intents"] if item["tag"] == predicted_tag), None)
    if not intent:
        intent = next(item for item in data["intents"] if item["tag"] == "fallback")
    responses = [response.strip() for response in intent.get("responses", []) if response.strip()]
    if not responses and predicted_tag != "fallback":
        intent = next(item for item in data["intents"] if item["tag"] == "fallback")
        predicted_tag = "fallback"
        responses = [response.strip() for response in intent.get("responses", []) if response.strip()]
    if not responses:
        raise ValueError("The fallback intent in intents.json does not contain a response.")
    response = random.choice(responses)
    return response, predicted_tag, confidence

def queue_question(question):
    st.session_state.pending_question = question

def icon_for(tag):
    icons = {
        "courses":"🎓", "departments":"🧪", "admission":"📝", "entrance_exam":"📚",
        "fees":"💳", "facilities":"✨", "library":"📖", "hostel":"🏠",
        "placement":"🚀", "timings":"🕒", "contact":"☎️", "location":"📍",
        "college_info":"🏛️", "greeting":"👋", "goodbye":"💜", "mba_mca":"🎯"
    }
    return icons.get(tag, "✦")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 7% 5%, rgba(124,58,237,.20), transparent 32%),
    radial-gradient(ellipse at 92% 16%, rgba(6,182,212,.13), transparent 28%),
    radial-gradient(ellipse at 75% 95%, rgba(236,72,153,.10), transparent 30%),
    #080914;
  color: #f5f5ff;
}
[data-testid="stHeader"] { background: rgba(8,9,20,.7); }
.block-container { max-width: 1220px; padding-top: 1.25rem; padding-bottom: 3rem; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
a { color: #67e8f9 !important; }
div[data-testid="stMarkdownContainer"] p { color: #c8c9dc; }
.topbar {
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  border:1px solid rgba(255,255,255,.11); border-radius:20px;
  background:rgba(16,17,35,.72); backdrop-filter:blur(18px);
  padding:14px 18px; margin-bottom:26px;
  box-shadow:0 12px 40px rgba(0,0,0,.18);
}
.brand { display:flex; align-items:center; gap:12px; color:#fff; font-weight:700; font-size:1.02rem; }
.brand-mark {
 width:42px;height:42px;display:flex;align-items:center;justify-content:center;border-radius:14px;
 background:linear-gradient(135deg,#9333ea,#2563eb 58%,#06b6d4);
 box-shadow:0 0 28px rgba(124,58,237,.35);font-size:1.2rem;
}
.nav-pill { border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.04); padding:8px 13px;border-radius:999px;color:#d9d9ef;font-size:.82rem; }
.eyebrow { display:inline-block; border:1px solid rgba(103,232,249,.25); background:rgba(34,211,238,.07); color:#67e8f9;
  padding:7px 11px; border-radius:999px; font-size:.76rem; letter-spacing:.09em; text-transform:uppercase; font-weight:700; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.45rem,5.8vw,4.75rem); line-height:1.02; letter-spacing:-.065em; font-weight:700; margin:.8rem 0 1rem; color:#fbfaff; }
.gradient-text { background:linear-gradient(90deg,#c084fc,#818cf8 42%,#67e8f9 78%,#f0abfc);-webkit-background-clip:text;background-clip:text;color:transparent; }
.hero-copy { max-width:650px; font-size:1.05rem; line-height:1.75; color:#b7b8d0; }
.hero-panel {
  position:relative; overflow:hidden; border:1px solid rgba(255,255,255,.13); border-radius:26px;
  background:linear-gradient(145deg,rgba(124,58,237,.17),rgba(17,24,48,.82) 55%,rgba(6,182,212,.10));
  padding:26px; box-shadow:0 20px 70px rgba(0,0,0,.25);
}
.orb { width:150px;height:150px;border-radius:50%;margin:4px auto 20px;
  background:radial-gradient(circle at 35% 28%,#e9d5ff 0%,#a855f7 18%,#4f46e5 45%,#0891b2 70%,rgba(8,9,20,.1) 72%);
  box-shadow:0 0 40px rgba(168,85,247,.35),0 0 90px rgba(34,211,238,.14); }
.mini-line { display:flex; align-items:center; gap:10px; margin:10px 0; color:#d9dbf3; font-size:.88rem; }
.mini-dot { width:8px;height:8px;border-radius:50%;background:#67e8f9;box-shadow:0 0 12px #67e8f9; }
.section-title { font-family:'Space Grotesk',sans-serif;font-size:1.55rem;font-weight:700;color:#f5f3ff;margin:1.2rem 0 .3rem; }
.section-copy { color:#a5a7c2;font-size:.93rem;margin-bottom:1rem; }
.metric-card {
  min-height:130px; border:1px solid rgba(255,255,255,.10); border-radius:20px;
  background:linear-gradient(145deg,rgba(255,255,255,.065),rgba(255,255,255,.018));
  padding:19px; box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
}
.metric-icon { font-size:1.35rem; margin-bottom:12px; }
.metric-title { font-weight:700;color:#f4f3ff;font-size:1rem;margin-bottom:5px; }
.metric-desc { color:#a9abc4;font-size:.83rem;line-height:1.5; }
.chat-shell {
 border:1px solid rgba(255,255,255,.12); border-radius:24px;
 background:linear-gradient(160deg,rgba(19,20,42,.92),rgba(12,14,29,.92));
 box-shadow:0 22px 70px rgba(0,0,0,.25); padding:20px;
}
.chat-head { display:flex;align-items:center;gap:12px;padding-bottom:15px;border-bottom:1px solid rgba(255,255,255,.09);margin-bottom:15px; }
.bot-avatar { width:44px;height:44px;border-radius:15px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#9333ea,#2563eb,#06b6d4);font-size:1.25rem; }
.online-dot { display:inline-block;width:7px;height:7px;background:#4ade80;border-radius:50%;margin-right:5px;box-shadow:0 0 10px #4ade80; }
.chat-bubble { border-radius:17px;padding:13px 15px;margin:9px 0;line-height:1.65;font-size:.92rem;white-space:pre-wrap; }
.bot-bubble { background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.07);color:#e8e8f7;border-top-left-radius:5px; }
.user-bubble { background:linear-gradient(120deg,rgba(124,58,237,.34),rgba(37,99,235,.28));border:1px solid rgba(167,139,250,.25);color:#fff;margin-left:10%;border-top-right-radius:5px; }
.small-muted { color:#8f92af;font-size:.78rem; }
div.stButton > button {
 border:1px solid rgba(167,139,250,.23); border-radius:13px; background:rgba(124,58,237,.10);
 color:#e9e7ff; font-weight:600; transition:all .18s ease; min-height:42px;
}
div.stButton > button:hover { border-color:rgba(103,232,249,.65); background:rgba(34,211,238,.11); color:#fff; transform:translateY(-1px); }
div[data-testid="stForm"] { border:1px solid rgba(255,255,255,.12);border-radius:16px;background:rgba(255,255,255,.025);padding:8px; }
.stTextInput input { background:rgba(255,255,255,.045) !important;border:1px solid rgba(255,255,255,.12) !important;color:#fff !important;border-radius:12px !important; }
.stTextInput input:focus { border-color:#a78bfa !important; box-shadow:0 0 0 1px #a78bfa !important; }
div[data-testid="stFormSubmitButton"] button { background:linear-gradient(100deg,#9333ea,#4f46e5 55%,#0891b2) !important;border:0 !important;color:white !important;border-radius:12px !important;font-weight:700 !important; }
div[data-testid="stChatInput"] > div {
 background:linear-gradient(120deg,rgba(19,20,42,.98),rgba(12,14,29,.98)) !important;
 border:1px solid rgba(167,139,250,.35) !important;border-radius:16px !important;
 box-shadow:0 8px 30px rgba(0,0,0,.25);
}
div[data-testid="stChatInput"] > div:focus-within {
 border-color:rgba(103,232,249,.7) !important;box-shadow:0 0 0 1px rgba(103,232,249,.2),0 8px 30px rgba(0,0,0,.25);
}
div[data-testid="stChatInput"] textarea {
 background:transparent !important;color:#f5f5ff !important;caret-color:#67e8f9 !important;
}
div[data-testid="stChatInput"] textarea::placeholder { color:#9698b6 !important;opacity:1; }
div[data-testid="stChatInput"] button {
 background:linear-gradient(100deg,#9333ea,#4f46e5 55%,#0891b2) !important;
 border:0 !important;border-radius:10px !important;color:#fff !important;
}
.footer { border-top:1px solid rgba(255,255,255,.09);padding-top:18px;margin-top:38px;color:#898ba8;font-size:.8rem; }
div[data-testid="stAlert"] { background:rgba(124,58,237,.1); }
@media (max-width: 700px) {
 .block-container { padding-left:1rem;padding-right:1rem; }
 .topbar { padding:11px 12px; }
 .nav-pill { display:none; }
 .hero-panel { padding:20px; margin-top:10px; }
 .hero-title { font-size:2.65rem; }
 .chat-shell { padding:13px; }
}
</style>
""", unsafe_allow_html=True)

data = load_data()
college = data["college"]

st.markdown(f"""
<div class="topbar">
  <div class="brand"><div class="brand-mark">✦</div><div>KEYSTONE <span style="color:#a78bfa">AI</span><div style="font-size:.68rem;color:#9698b6;font-weight:500;letter-spacing:.04em">COLLEGE ENQUIRY ASSISTANT</div></div></div>
  <div style="display:flex;gap:8px;align-items:center"><span class="nav-pill">✧ Student Support</span><span class="nav-pill">● NLP Powered</span></div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown('<span class="eyebrow">✦ YOUR CAMPUS, REIMAGINED</span>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Your campus.<br><span class="gradient-text">Your questions.</span><br>One AI.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Meet your always-ready college guide. Explore branches, understand the admission journey, check fee information and discover campus facilities — all in one conversation.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:16px"><span class="nav-pill">🎓 Courses & branches</span><span class="nav-pill">🧠 NLP intent detection</span><span class="nav-pill">⚡ Instant answers</span></div>', unsafe_allow_html=True)
with right:
    st.markdown("""
    <div class="hero-panel">
      <div class="orb"></div>
      <div style="text-align:center;font-family:'Space Grotesk';font-size:1.2rem;font-weight:700;color:#f5f3ff">Your AI campus companion</div>
      <div style="text-align:center;color:#a8abc7;font-size:.85rem;margin:6px 0 20px">Smart answers, less searching.</div>
      <div class="mini-line"><span class="mini-dot"></span> Understands student questions</div>
      <div class="mini-line"><span class="mini-dot" style="background:#c084fc;box-shadow:0 0 12px #c084fc"></span> Classifies intent with Naive Bayes</div>
      <div class="mini-line"><span class="mini-dot" style="background:#f472b6;box-shadow:0 0 12px #f472b6"></span> Responds using a rule-based system</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">Explore your next step</div><div class="section-copy">Jump straight to the information students ask for most.</div>', unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role":"assistant",
        "content":"Hey there! 👋 I'm Keystone AI, your college enquiry assistant. What would you like to know today?"
    }]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

cards = [
    ("🎓","Courses & branches","Explore engineering departments.","courses"),
    ("📝","Admissions","Understand the admission journey.","admission"),
    ("💳","Fee structure","See the fee details in our information base.","fees"),
    ("✨","Campus life","Explore facilities and student support.","facilities"),
    ("🚀","Placements","Ask about career and placement support.","placement"),
    ("📍","Find us","Get college location and contact information.","location"),
]
cols = st.columns(3, gap="medium")
for i, (emoji, title, desc, tag) in enumerate(cards):
    with cols[i % 3]:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">{emoji}</div><div class="metric-title">{title}</div><div class="metric-desc">{desc}</div></div>', unsafe_allow_html=True)
        card_questions = {
            "courses": "Which courses are available?",
            "admission": "How can I apply for admission?",
            "fees": "What are the fees?",
            "facilities": "What is campus life like?",
            "placement": "Does the college have placements?",
            "location": "Where is the college?",
        }
        st.button(
            f"Ask about {title} ↗",
            key=f"card_{tag}",
            use_container_width=True,
            on_click=queue_question,
            args=(card_questions[tag],),
        )

st.markdown('<div class="section-title" id="chat">Talk to Keystone AI</div><div class="section-copy">Type a question or choose a prompt below. The classifier predicts the intent and the response layer selects a matching answer.</div>', unsafe_allow_html=True)

quick_questions = [
    ("🎓 Courses", "Which courses are available?"),
    ("📝 Admission", "How can I apply for admission?"),
    ("💳 Fees", "What are the fees?"),
    ("✨ Facilities", "What facilities are available?"),
    ("🌿 Campus Life", "What is campus life like?"),
    ("🚀 Placements", "Does the college have placements?"),
    ("📍 Find Us", "Where is the college?"),
]
for start in range(0, len(quick_questions), 4):
    row_questions = quick_questions[start:start + 4]
    qcols = st.columns(len(row_questions), gap="small")
    for i, (label, question) in enumerate(row_questions):
        with qcols[i]:
            st.button(
                label,
                key=f"quick_{start + i}",
                use_container_width=True,
                on_click=queue_question,
                args=(question,),
            )

user_question = st.chat_input("Ask anything about Keystone...")

question_to_process = None
if user_question and user_question.strip():
    question_to_process = user_question.strip()
elif st.session_state.pending_question:
    question_to_process = st.session_state.pending_question
    st.session_state.pending_question = None

if question_to_process:
    response, tag, confidence = get_intent_response(question_to_process)
    if not response.strip():
        raise ValueError(f"Intent response is empty for predicted intent: {tag}")

    st.session_state.messages.extend([
        {"role": "user", "content": question_to_process},
        {"role": "assistant", "content": response},
    ])
    st.session_state.last_prediction = {"tag": tag, "confidence": confidence}

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
st.markdown("""
<div class="chat-head">
  <div class="bot-avatar">✦</div>
  <div style="flex:1"><div style="font-weight:700;color:#f6f4ff">Keystone AI Assistant</div><div class="small-muted"><span class="online-dot"></span> Ready to help · NLP + Naive Bayes</div></div>
  <div class="nav-pill">LIVE CHAT</div>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        safe_content = html.escape(msg["content"])
        if msg["role"] == "assistant":
            safe_content = safe_content.replace("https://www.keystone.ac.in/", '<a href="https://www.keystone.ac.in/" target="_blank">keystone.ac.in</a>')
            safe_content = safe_content.replace("https://cetcell.mahacet.org/", '<a href="https://cetcell.mahacet.org/" target="_blank">Maharashtra CET Cell</a>')
        bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        st.markdown(f'<div class="chat-bubble {bubble_class}">{safe_content}</div>', unsafe_allow_html=True)

with st.expander("🧪 NLP model details (for project demonstration)"):
    st.write("**Pipeline:** Regex tokenization → TF-IDF (unigrams + bigrams) → Multinomial Naive Bayes → intent label → rule-based response.")
    st.write("The training examples are stored in `intents.json`. Add more example phrasings per intent to improve generalization.")
    if "last_prediction" in st.session_state:
        pred = st.session_state.last_prediction
        st.write(f"Last predicted intent: `{pred['tag']}` · Model confidence: `{pred['confidence']:.2f}`")
    st.caption("This is a student project prototype. Verify admission and fee details with the official college before making decisions.")

st.markdown(f"""
<div class="footer">
  <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap">
    <span>© Keystone AI · College Enquiry Chatbot · NLP Project</span>
    <span><a href="{college['website']}" target="_blank">Official college website ↗</a> &nbsp;·&nbsp; Built with Python + Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)
