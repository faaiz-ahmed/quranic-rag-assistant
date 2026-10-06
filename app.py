import io
import os
import pdfplumber
from pypdf import PdfReader
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="Quranic AI Assistant (RAG)", page_icon="📖", layout="centered"
)

st.title("📖 Quranic AI Assistant Using RAG")
st.caption("Ask questions based on the English translation of Surah Al-Fatihah.")

# 1. API Key handling
api_key = None
try:
  if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
  pass

if not api_key:
  api_key = os.getenv("GROQ_API_KEY")

if not api_key:
  api_key = st.sidebar.text_input(
      "Enter Groq API Key:",
      type="password",
      help="Get a free key from console.groq.com",
  )

if not api_key:
  st.info("👈 Please enter or configure your Groq API key to proceed.")
  st.stop()


# 2. Embedding Model
@st.cache_resource(show_spinner=False)
def get_embeddings():
  return HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2"
  )


SURAH_FATIHA_TEXT = """
Surah 1. Al-Fatihah (The Opening / The Opener)
Revealed in Mecca - 7 Verses

English Translation:
1. In the Name of Allah, the Entirely Merciful, the Especially Merciful.
2. [All] praise and thanks are due to Allah, Lord of the worlds ('Alamin: mankind, jinn, and all that exists).
3. The Entirely Merciful, the Especially Merciful.
4. Sovereign and Only Owner of the Day of Recompense (the Day of Resurrection and Judgement).
5. It is You (alone) we worship, and You (alone) we ask for help (for each and everything).
6. Guide us to the Straight Way / Path.
7. The Way / Path of those on whom You have bestowed Your Grace and favor, not of those who earned Your Anger, nor of those who went astray.

Key Explanations & Commentary:
- Meaning of Rabb (Lord): Rabb means the One and Only Lord for the entire universe, its Creator, Owner, Organizer, Provider, Master, Planner, Sustainer, Cherisher, and Giver of security. Rabb is also one of the Divine Names of Allah.
- Status of the Surah: Surat Al-Fatihah is described as the greatest Surah in the Quran, also known as As-Sab' Al-Mathani (the seven repeatedly recited verses). Reciting it is essential; prayers without reciting Surat Al-Fatihah are considered invalid.
- Two Types of Guidance:
  1. Guidance of Taufiq: Strictly from Allah, wherein He opens a person's heart to accept the truth and belief in Monotheism.
  2. Guidance of Irshad: Guidance imparted through teaching, preaching, and lessons delivered by Allah's Messengers and righteous preachers.
- The Path of the Favored: Refers to the path trodden by the Prophets, the Siddiqun (truthful believers like Abu Bakr), the martyrs, and the righteous.
- Saying Amin: When the Imam finishes reciting verse 7 ('walad-dallin'), saying Amin in unison with the angels brings forgiveness of previous minor sins.
"""


def extract_text(file_bytes):
  text_data = ""
  # Try pdfplumber first
  try:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
      for page in pdf.pages:
        txt = page.extract_text()
        if txt:
          text_data += txt + "\n"
  except Exception:
    pass

  # If pdfplumber didn't yield text, try pypdf
  if not text_data.strip():
    try:
      reader = PdfReader(io.BytesIO(file_bytes))
      for page in reader.pages:
        txt = page.extract_text()
        if txt:
          text_data += txt + "\n"
    except Exception:
      pass

  return text_data.strip()


# 3. File Uploader
uploaded_file = st.file_uploader(
    "Upload Surah PDF (e.g., surah.pdf or Surah-Fatiha.pdf):", type=["pdf"]
)

# Combine extracted document with contextual text
with st.spinner("Indexing Surah content..."):
  combined_content = SURAH_FATIHA_TEXT

  if uploaded_file:
    file_bytes = uploaded_file.read()
    extracted = extract_text(file_bytes)
    if extracted:
      combined_content = f"{extracted}\n\n---\nAdditional Context:\n{SURAH_FATIHA_TEXT}"

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=400, chunk_overlap=60
  )
  docs = [Document(page_content=combined_content)]
  splits = text_splitter.split_documents(docs)

  embeddings = get_embeddings()
  vector_db = FAISS.from_documents(splits, embeddings)
  retriever = vector_db.as_retriever(search_kwargs={"k": 4})

st.success("Surah loaded and ready for questions!")

# 4. LLM Setup (Using your active Groq model)
llm = ChatGroq(
    groq_api_key=api_key, model_name="qwen/qwen3.8-27b", temperature=0.1
)

system_prompt = (
    "You are an academic and helpful Quranic Assistant.\n"
    "Answer the user's question clearly, thoroughly, and respectfully using the provided context below.\n"
    "If the question cannot be answered from the context, state that the information is not present in the text.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# 5. Query Box
user_question = st.text_input(
    "Ask a question about the Surah:",
    placeholder="e.g., What are the attributes of Allah mentioned in the verses?",
)

if user_question:
  with st.spinner("Retrieving verses and generating answer..."):
    response = rag_chain.invoke({"input": user_question})

    st.markdown("### Answer:")
    st.write(response["answer"])

    with st.expander("View Retrieved Context Chunks"):
      for idx, doc in enumerate(response["context"]):
        st.markdown(f"**Chunk {idx+1}:**")
        st.write(doc.page_content)