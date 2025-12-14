import streamlit as st
import requests
import fitz  # PyMuPDF
from openai import OpenAI
import tempfile
import os
import concurrent.futures
import time
import json
import re
import ftfy
import pandas as pd
from fpdf import FPDF

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="PDF Deep Examiner (Pro Edition)", layout="wide")

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("🚨 API Key missing! Add to .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=api_key)

# --- 2. CLEANING FUNCTIONS ---
def clean_text_advanced(text):
    """
    Cleans PDF artifacts to ensure high-quality AI input.
    """
    if not text: return ""
    text = ftfy.fix_text(text)
    symbol_map = {"®": " implies ", "→": " implies ", "•": "-", "◦": "-", "▪": "-", "G ": "- ", "ß": "-->", "ü": "", "ý": "", "þ": ""}
    for char, replacement in symbol_map.items():
        text = text.replace(char, replacement)
    text = re.sub(r'(?<!\S)([A-Z])\s+(?=[A-Z](?:\s|$))', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. HELPER FUNCTIONS ---
def validate_pdf(file_path):
    try:
        doc = fitz.open(file_path); toc = doc.get_toc(); doc.close()
        return (True, toc) if len(toc) > 0 else (False, [])
    except: return False, []

def extract_text_from_range(file_path, start_page, end_page):
    doc = fitz.open(file_path)
    raw_text = ""
    if start_page < 1: start_page = 1
    if end_page > doc.page_count: end_page = doc.page_count
    for page_num in range(start_page - 1, end_page):
        page = doc.load_page(page_num)
        raw_text += clean_text_advanced(page.get_text()) + "\n\n"
    doc.close()
    return raw_text

def chunk_text(text, chunk_size=2000, overlap=300):
    # Increased chunk size to allow AI to see more context for complex questions
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# --- 4. AI AGENTS (UPGRADED PROMPTS) ---

def analyze_pyq_style(pyq_text):
    """
    UPGRADED: Extracts a detailed 'Exam Blueprint' rather than a short summary.
    """
    prompt = f"""
    You are a Senior Forensic Exam Analyst. 
    Your task is to deconstruct the "DNA" of the provided Previous Year Questions (PYQs).
    
    Analyze the text and extract a detailed **'Exam Persona Profile'** covering these 3 dimensions:
    
    1. **Cognitive Depth:** Does the exam focus on Rote Memorization (dates, names, sections), Conceptual Understanding (definitions, basic logic), or Deep Analysis (application, synthesis, multi-step logic)?
    2. **Question Archetypes:** What phrasing is common? (e.g., "Which of the following is FALSE?", "Assertion-Reasoning", "Match the following", or direct "What is X?").
    3. **The 'Trap' Factor:** How does the examiner trick students? (e.g., confusingly similar options, testing obscure exceptions to rules, or very specific numerical data).

    **OUTPUT:** A detailed, instructional paragraph (approx 100 words) that commands a Question Generator exactly how to mimic this specific difficulty and style. Start with "Adopt a persona that..."
    
    PYQ SAMPLE TEXT:
    {pyq_text[:20000]}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        return response.choices[0].message.content
    except: return "Adopt a persona that focuses on core concepts, clear definitions, and standard conceptual understanding."

def generate_glossary_json(chunk_text):
    prompt = f"""
    You are a Technical Editor.
    Identify top 3-5 technical terms/concepts in this text.
    Define them **strictly based on their usage in this text**.
    RETURN JSON: {{ "terms": [ {{ "term": "...", "definition": "..." }} ] }}
    TEXT: {chunk_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            temperature=0.1, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content).get("terms", [])
    except: return []

def generate_qa_json(chunk_text, pyq_instruction=None):
    """
    UPGRADED: Detailed 'Few-Shot' style prompt logic.
    """
    
    # 1. Default "High Quality" Instructions
    core_instructions = """
    1. **ANALYZE**: Identify the most important concepts, definitions, and logic in the text.
    2. **COMBINE**: If multiple facts relate to the same topic, combine them into one comprehensive question.
    3. **DEEP ANALYSIS**: Do not ask superficial questions like "What is X?". Look for relationships: "How does X influence Y?" or "What distinguishes X from Z?".
    4. **COMPREHENSIVE ANSWERS**: The 'answer' field must be a clear, complete, standalone statement derived from the text.
    5. **NO DUPLICATES**: Ensure each question explores a unique angle.
    """
    
    # 2. Adaptive Logic Layer
    adaptive_layer = ""
    if pyq_instruction:
        adaptive_layer = f"""
    \n--- ⚠️ CRITICAL: EXAM PATTERN OVERRIDE ---
    You must align your generation logic to this specific 'Exam Persona':
    "{pyq_instruction}"
    
    *IMPLICATION:*
    - If the persona asks for **Sections/Dates**, prioritize those details over broad concepts.
    - If the persona asks for **Application**, ignore simple definitions and create scenarios.
    - If the persona asks for **"NOT" questions**, generate questions like "Which of the following is NOT...".
    ---------------------------------------------
    """

    prompt = f"""
    You are a Lead Curriculum Designer and Exam Setter.
    Your goal is to extract knowledge from the text and convert it into a targeted "Question & Answer" guide.

    --- INSTRUCTIONS ---
    {core_instructions}
    {adaptive_layer}

    --- STRICT CONSTRAINTS ---
    - NO OUTSIDE KNOWLEDGE.
    - RETURN JSON ONLY.
    
    --- JSON STRUCTURE ---
    {{
        "qa_pairs": [
            {{
                "question": "Question text?",
                "answer": "Detailed answer statement.",
                "context": "Short quote/evidence from text."
            }}
        ]
    }}
    
    INPUT TEXT:
    {chunk_text}
    """
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, # Low temp for precision
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            time.sleep(2)
            if attempt == 2: return {"qa_pairs": []}

def parse_toc_with_ai(toc_text):
    prompt = f"""
    Extract Chapter Titles and Start Page numbers.
    JSON ONLY: {{ "chapters": [ ["Chapter Name", PageNum (int)], ... ] }}
    TEXT: {toc_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            temperature=0.0, response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        if "chapters" in data: return data["chapters"]
        for k, v in data.items():
            if isinstance(v, list): return v
        return []
    except: return []

# --- 5. PDF EXPORT ---
def create_master_pdf(glossary, qa_list, topic, style_note):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, f'Mastery Guide: {topic}', 0, 1, 'C')
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f'Target Strategy: {style_note[:60]}...', 0, 1, 'C')
            self.ln(10)
    
    pdf = PDF()
    pdf.add_page()
    
    if glossary:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Part 1: Key Terminology", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=11)
        for item in glossary:
            t = item['term'].encode('latin-1', 'replace').decode('latin-1')
            d = item['definition'].encode('latin-1', 'replace').decode('latin-1')
            pdf.set_font("Arial", 'B', 11); pdf.cell(0, 8, f"{t}: ", ln=False)
            pdf.set_font("Arial", size=11); pdf.multi_cell(0, 8, d); pdf.ln(2)
        pdf.add_page()

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Part 2: Deep Dive Q&A", ln=True)
    pdf.ln(5)
    for i, item in enumerate(qa_list):
        q = f"Q{i+1}: {item.get('question', '')}".encode('latin-1', 'replace').decode('latin-1')
        a = item.get('answer', '').encode('latin-1', 'replace').decode('latin-1')
        pdf.set_font("Arial", 'B', 12); pdf.multi_cell(0, 8, q)
        pdf.set_font("Arial", size=11); pdf.set_text_color(0, 100, 0)
        pdf.multi_cell(0, 8, f"Ans: {a}")
        pdf.set_text_color(0, 0, 0); pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1')

# --- 6. GUI ---

st.title("🧠 PDF Deep Examiner (Pro Edition)")
st.caption("Glossary + Adaptive Q&A + Pattern Recognition")

# Session State
if 'pyq_style' not in st.session_state: st.session_state.pyq_style = None
if 'custom_toc' not in st.session_state: st.session_state.custom_toc = []
if 'quiz_data' not in st.session_state: st.session_state.quiz_data = []
if 'glossary_data' not in st.session_state: st.session_state.glossary_data = []

# --- SIDEBAR: PYQ UPLOAD ---
with st.sidebar:
    st.header("🎯 Adaptive Mode")
    st.info("Upload PYQs to teach the AI your exam style.")
    pyq_file = st.file_uploader("Upload Previous Year Questions (PDF)", type="pdf")
    
    if pyq_file:
        if st.button("🔍 Learn Pattern"):
            with st.spinner("Analyzing Exam DNA..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pyq_file.getvalue()); tmp_pyq = tmp.name
                
                doc = fitz.open(tmp_pyq)
                pyq_txt = ""
                for p in doc: pyq_txt += clean_text_advanced(p.get_text())
                doc.close()
                
                if len(pyq_txt) < 500:
                    st.error("Text too short. Is the PDF scanned?")
                else:
                    style = analyze_pyq_style(pyq_txt)
                    st.session_state.pyq_style = style
                    st.success("✅ Pattern Learned!")
                    with st.expander("View Strategy"):
                        st.write(style)

# --- MAIN: TEXTBOOK ---
uploaded_file = st.file_uploader("📂 Upload Textbook (PDF)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    total_pages = doc.page_count
    doc.close()

    # TOC Logic
    if not st.session_state.custom_toc:
        with st.spinner("Scanning Structure..."):
            has_toc, toc_data = validate_pdf(tmp_path)
            if has_toc:
                st.success(f"Found {len(toc_data)} bookmarks.")
                if st.radio("Chapter Source:", ["Use Bookmarks", "Scan with AI"]) == "Use Bookmarks":
                    st.session_state.custom_toc = toc_data
            
    # AI TOC Fallback
    if not st.session_state.custom_toc:
        c1, c2 = st.columns(2)
        s = c1.number_input("Start Page", 1, total_pages, 1)
        e = c2.number_input("End Page", 1, total_pages, 1)
        if st.button("Scan Table of Contents"):
            txt = extract_text_from_range(tmp_path, s, e)
            res = parse_toc_with_ai(txt)
            if res:
                st.session_state.custom_toc = [[1, x[0], x[1]] for x in res]
                st.rerun()
            else: st.error("Scan failed.")

    # GENERATION LOGIC
    if st.session_state.custom_toc:
        titles = [x[1] for x in st.session_state.custom_toc]
        choice = st.selectbox("Select Chapter:", titles)
        
        start_p, end_p = 0, 0
        for i, entry in enumerate(st.session_state.custom_toc):
            if entry[1] == choice:
                start_p = entry[2]
                end_p = st.session_state.custom_toc[i+1][2] if i+1 < len(st.session_state.custom_toc) else total_pages
                break
        
        st.info(f"📖 Chapter: **{choice}** (Pages {start_p}-{end_p})")
        
        # Mode Display
        if st.session_state.pyq_style:
            st.success(f"**Active Mode:** 🎯 Adaptive (Aligned to uploaded PYQs)")
        else:
            st.info(f"**Active Mode:** 📘 Standard Comprehensive (No PYQs uploaded)")

        if st.button("🚀 Generate Mastery Guide"):
            status = st.status("Processing...", expanded=True)
            
            raw_text = extract_text_from_range(tmp_path, int(start_p), int(end_p))
            chunks = chunk_text(raw_text)
            status.write(f"✅ Text Extracted ({len(chunks)} blocks).")
            
            qa_res, gloss_res = [], []
            prog = status.progress(0)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                # PASS STYLE (If None, it uses default strategy)
                qa_futures = {ex.submit(generate_qa_json, c, st.session_state.pyq_style): c for c in chunks}
                gloss_futures = {ex.submit(generate_glossary_json, c): c for c in chunks}
                
                done = 0
                for f in concurrent.futures.as_completed(qa_futures):
                    try:
                        res = f.result()
                        if isinstance(res, dict): res = res.get("qa_pairs", [])
                        qa_res.extend(res)
                    except: pass
                    done += 1
                    prog.progress(done / (len(chunks) * 2))
                
                for f in concurrent.futures.as_completed(gloss_futures):
                    try: gloss_res.extend(f.result())
                    except: pass
                    
            # Deduplicate Glossary
            unique_gloss = {}
            for item in gloss_res:
                if item['term'] not in unique_gloss: unique_gloss[item['term']] = item['definition']
            
            st.session_state.glossary_data = [{"term":k, "definition":v} for k,v in unique_gloss.items()]
            st.session_state.quiz_data = qa_res
            
            prog.empty()
            status.update(label="✅ Generation Complete!", state="complete")

    # --- DISPLAY ---
    if st.session_state.quiz_data:
        style_label = st.session_state.pyq_style if st.session_state.pyq_style else "Standard Comprehensive"
        pdf = create_master_pdf(st.session_state.glossary_data, st.session_state.quiz_data, choice, style_label)
        st.download_button("📄 Download Guide (PDF)", pdf, "MasteryGuide.pdf", "application/pdf")
        
        tab1, tab2 = st.tabs(["🧠 Glossary", "❓ Study Guide"])
        
        with tab1:
            if st.session_state.glossary_data:
                st.dataframe(pd.DataFrame(st.session_state.glossary_data), use_container_width=True, hide_index=True)
            else: st.warning("No glossary terms found.")
            
        with tab2:
            for i, q in enumerate(st.session_state.quiz_data):
                with st.expander(f"Q{i+1}: {q.get('question')}"):
                    st.markdown(f"**Answer:** {q.get('answer')}")
                    st.caption(f"Source: {q.get('context')}")