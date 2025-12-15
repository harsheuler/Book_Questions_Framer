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
import ftfy  # The "Permanent" text fixer
from fpdf import FPDF

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="PDF Deep Examiner (Q&A Mode)", layout="wide")

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("🚨 API Key missing! Please add it to .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=api_key)

# --- 2. ADVANCED CLEANING FUNCTIONS ---

def clean_text_advanced(text):
    """PERMANENT CLEANING SOLUTION."""
    if not text: return ""
    text = ftfy.fix_text(text)
    symbol_map = {"®": " implies ", "→": " implies ", "•": "-", "◦": "-", "▪": "-", "G ": "- ", "ß": "-->", "ü": "", "ý": "", "þ": ""}
    for char, replacement in symbol_map.items():
        text = text.replace(char, replacement)
    text = re.sub(r'(?<!\S)([A-Z])\s+(?=[A-Z](?:\s|$))', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text

# --- 3. HELPER FUNCTIONS ---

def validate_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        toc = doc.get_toc()
        doc.close()
        return (True, toc) if len(toc) > 0 else (False, [])
    except:
        return False, []

def extract_text_from_range(file_path, start_page, end_page):
    doc = fitz.open(file_path)
    raw_text = ""
    if start_page < 1: start_page = 1
    if end_page > doc.page_count: end_page = doc.page_count
    for page_num in range(start_page - 1, end_page):
        page = doc.load_page(page_num)
        page_content = page.get_text()
        clean_content = clean_text_advanced(page_content)
        raw_text += clean_content + "\n\n"
    doc.close()
    return raw_text

def chunk_text(text, chunk_size=1500, overlap=200):
    # Increased chunk size for Q&A to give more context for "Statement" answers
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# --- 4. NEW: AI ToC PARSER ---
def parse_toc_with_ai(toc_page_text):
    prompt = f"""
    You are a PDF Structure Analyzer.
    Below is text extracted from a Table of Contents page.
    Your Goal: Extract Chapter Titles and their Start Page numbers.
    RULES:
    1. Ignore Preface, Acknowledgements, etc. Focus on Chapters.
    2. Return JSON ONLY: {{ "chapters": [ ["Chapter Name", PageNumber (int)], ... ] }}
    3. If the page number is Roman (ix, x), convert to 0.
    TEXT:
    {toc_page_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        if "chapters" in data: return data["chapters"]
        for k, v in data.items():
            if isinstance(v, list): return v
        return []
    except:
        return []

# --- 5. CORE LOGIC: SMART Q&A GENERATION ---
def generate_qa_json(chunk_text):
    """
    Generates Q&A Pairs (Question + Statement Answer).
    Deduplication is now handled via Prompt instructions.
    """
    prompt = f"""
    You are a Lead Curriculum Designer.
    Your goal is to extract key knowledge from the text and convert it into a "Question & Answer" study guide.

    --- STRATEGY ---
    1. **ANALYZE**: Identify the most important concepts, definitions, and logic in the text.
    2. **COMBINE**: If multiple facts relate to the same topic, combine them into one comprehensive question.
    3. **STATEMENT ANSWER**: The answer must be a clear, complete sentence (statement) derived directly from the text.
    
    --- DEDUPLICATION RULES ---
    - Do NOT generate two questions that test the exact same fact.
    - If a concept is repeated in the text, generate only ONE high-quality question for it.
    - Focus on unique insights.

    --- STRICT CONSTRAINTS ---
    - NO OUTSIDE KNOWLEDGE.
    - RETURN JSON ONLY.
    
    --- JSON STRUCTURE ---
    {{
        "qa_pairs": [
            {{
                "question": "What is the relationship between X and Y?",
                "answer": "According to the text, X causes Y when condition Z is met.",
                "context": "Direct quote from text."
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
                temperature=0.2, 
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            time.sleep(2)
            if attempt == 2: return {"qa_pairs": []}

# --- 6. PDF EXPORT FUNCTION (Q&A Format) ---
def create_qa_pdf(qa_list, topic):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, f'Study Guide: {topic}', 0, 1, 'C')
            self.ln(10)
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for i, item in enumerate(qa_list):
        # Question
        q_text = f"Q{i+1}: {item.get('question', '')}".encode('latin-1', 'replace').decode('latin-1')
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 10, q_text)
        
        # Answer
        ans = item.get('answer', '').encode('latin-1', 'replace').decode('latin-1')
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(0, 100, 0) # Green
        pdf.multi_cell(0, 8, f"Ans: {ans}")
        
        # Context
        ctx = item.get('context', '').encode('latin-1', 'replace').decode('latin-1')
        pdf.set_text_color(100, 100, 100) # Gray
        pdf.multi_cell(0, 8, f"Ref: {ctx}")
        
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1')

# --- 7. STREAMLIT GUI ---

st.title("📚 PDF Deep Examiner (Q&A Mode)")
st.markdown("Upload PDF -> **Auto-Detect Chapters** -> **Study Guide**")

# Session State
if 'quiz_data' not in st.session_state: st.session_state.quiz_data = []
if 'quiz_generated' not in st.session_state: st.session_state.quiz_generated = False
if 'raw_text_debug' not in st.session_state: st.session_state.raw_text_debug = ""
if 'custom_toc' not in st.session_state: st.session_state.custom_toc = []
if 'use_ai_toc' not in st.session_state: st.session_state.use_ai_toc = False

uploaded_file = st.file_uploader("Upload Textbook (PDF)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    doc_check = fitz.open(tmp_path)
    total_pages_in_file = doc_check.page_count
    doc_check.close()

    with st.spinner("Scanning Metadata..."):
        has_toc, toc_data = validate_pdf(tmp_path)
    
    final_toc = []
    
    if has_toc:
        st.success(f"✅ Found {len(toc_data)} existing Digital Bookmarks.")
        mode_choice = st.radio("How would you like to select chapters?", 
                               ("Use Existing Bookmarks (Recommended)", "Ignore Bookmarks & Scan Content Page with AI"))
        if mode_choice == "Use Existing Bookmarks (Recommended)":
            final_toc = toc_data
            st.session_state.use_ai_toc = False
        else:
            st.session_state.use_ai_toc = True
            final_toc = []
    else:
        st.warning("⚠️ No Digital Bookmarks found.")
        st.session_state.use_ai_toc = True

    if st.session_state.use_ai_toc:
        if st.session_state.custom_toc:
            final_toc = st.session_state.custom_toc
            st.success(f"✅ AI-Extracted Chapters Loaded ({len(final_toc)} found).")
            if st.button("🔄 Reset / Scan Again"):
                st.session_state.custom_toc = []
                st.rerun()
        else:
            st.info("I need to read the **Table of Contents** page.")
            c1, c2 = st.columns([1, 1])
            with c1: toc_start = st.number_input("ToC Start Page", min_value=1, max_value=total_pages_in_file, value=1)
            with c2: toc_end = st.number_input("ToC End Page", min_value=1, max_value=total_pages_in_file, value=1)
            
            if st.button("🔍 Scan Table of Contents"):
                if toc_start > toc_end: st.error("Start > End")
                else:
                    with st.spinner("AI reading ToC..."):
                        raw_toc_text = extract_text_from_range(tmp_path, toc_start, toc_end)
                        ai_extracted_toc = parse_toc_with_ai(raw_toc_text)
                        if ai_extracted_toc:
                            formatted_toc = [[1, item[0], item[1]] for item in ai_extracted_toc]
                            st.session_state.custom_toc = formatted_toc
                            st.rerun()
                        else:
                            st.error("Could not read chapters. Try Manual Mode.")

    if final_toc:
        chapter_titles = [entry[1] for entry in final_toc]
        selected_chapter = st.selectbox("Select Chapter:", chapter_titles)
        
        start_p = None
        end_p = None
        for i, entry in enumerate(final_toc):
            if entry[1] == selected_chapter:
                start_p = entry[2]
                if i + 1 < len(final_toc): end_p = final_toc[i+1][2]
                else: end_p = total_pages_in_file
                break
        
        if start_p:
            st.info(f"📍 '{selected_chapter}' identified: Pages **{start_p}** to **{end_p}**.")
            st.markdown("---")
            
            c1, c2 = st.columns(2)
            with c1: user_start = st.number_input("Start Page", min_value=1, max_value=total_pages_in_file, value=int(start_p))
            with c2: user_end = st.number_input("End Page", min_value=1, max_value=total_pages_in_file, value=int(end_p))

            if st.button("🚀 Generate Study Guide"):
                status_box = st.status("Initializing Smart Deep Scan...", expanded=True)
                
                status_box.write(f"📖 Reading text from Page {user_start} to {user_end}...")
                full_text = extract_text_from_range(tmp_path, user_start, user_end)
                st.session_state.raw_text_debug = f"--- CLEANED SOURCE TEXT ---\n{full_text}"
                
                if len(full_text) < 100:
                    status_box.update(label="❌ Error: Text too short.", state="error"); st.stop()
                
                chunks = chunk_text(full_text, chunk_size=1500, overlap=200)
                status_box.write(f"🧩 Split into {len(chunks)} high-density context blocks.")
                
                status_box.write(f"⚡ Synthesizing Questions & Answers...")
                raw_qa_pairs = []
                progress_bar = status_box.progress(0)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_chunk = {executor.submit(generate_qa_json, chunk): chunk for chunk in chunks}
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        try:
                            data = future.result()
                            if isinstance(data, dict):
                                qa_list = data.get("qa_pairs", [])
                                if not qa_list:
                                    for key in data:
                                        if isinstance(data[key], list): qa_list = data[key]; break
                                raw_qa_pairs.extend(qa_list)
                            elif isinstance(data, list):
                                raw_qa_pairs.extend(data)
                        except Exception as e: print(f"Error: {e}")
                        
                        completed += 1
                        progress_bar.progress(completed / len(chunks))
                
                # NO PYTHON DEDUPLICATION (As requested)
                # We rely on the Prompt's "Do NOT generate duplicates" instruction.
                
                st.session_state.quiz_data = raw_qa_pairs
                st.session_state.quiz_generated = True
                progress_bar.empty()
                status_box.update(label=f"✅ Ready! Generated {len(raw_qa_pairs)} Q&A pairs.", state="complete", expanded=False)

        if st.session_state.raw_text_debug:
            st.download_button("📥 Download Cleaned Text (Debug)", st.session_state.raw_text_debug, "debug_text.txt")

        if st.session_state.quiz_generated and st.session_state.quiz_data:
            
            st.divider()
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown("## 🧠 Interactive Study Guide")
                st.caption(f"Topic: {selected_chapter} | Total Items: {len(st.session_state.quiz_data)}")
            with c2:
                pdf_bytes = create_qa_pdf(st.session_state.quiz_data, selected_chapter)
                st.download_button(label="📄 Download Q&A PDF", data=pdf_bytes, file_name=f"StudyGuide_{selected_chapter[:10]}.pdf", mime="application/pdf")
            
            for i, item in enumerate(st.session_state.quiz_data):
                q_text = item.get("question", "Unknown Question")
                ans_text = item.get("answer", "No answer provided.")
                context = item.get("context", "")
                
                # Q&A UI Style
                with st.expander(f"**Q{i+1}: {q_text}**"):
                    st.markdown(f"**Answer:** {ans_text}")
                    if context:
                        st.caption(f"📖 *Source:* {context}")

        elif st.session_state.quiz_generated and not st.session_state.quiz_data:
            st.warning("⚠️ No Q&A generated.")

    elif not has_toc and not st.session_state.custom_toc and not st.session_state.use_ai_toc:
        pass