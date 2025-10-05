# Import the necessary libraries
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# --- 1. Page Configuration and Title ---
st.set_page_config(page_title="PinterChat", page_icon="🤖")

# --- PERUBAHAN DI SINI ---
st.title("🤖 PinterChat: RAG & Agent Chatbot")
st.caption("Tanya apa saja, atau unggah dokumen Anda untuk jawaban yang lebih spesifik.")

# --- 3. Agent Initialization from Secrets ---
if "agent" not in st.session_state:
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        llm_agent = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )
        st.session_state.agent = create_react_agent(
            model=llm_agent,
            tools=[],
            prompt="You are a helpful, friendly assistant. Respond concisely and clearly."
        )
    except Exception as e:
        st.error(f"Failed to initialize the general agent: {e}")
        st.stop()

# --- 4. RAG Implementation & Sidebar ---
with st.sidebar:
    # --- PERUBAHAN DI SINI ---
    with st.expander("Petunjuk Pemakaian"):
        st.info(
            """
            1. **Mode Agen**: PinterChat dapat langsung berfungsi sebagai ChatAI reguler untuk menjawab pertanyaan umum.
            2. **Mode RAG**: Anda dapat berinteraksi dengan dokumen `.txt`. Unggah file Anda, maka PinterChat akan otomatis beralih ke mode RAG untuk menjawab pertanyaan berdasarkan isi dokumen.
            """
        )

    st.subheader("Pengaturan RAG")
    uploaded_file = st.file_uploader("Unggah dokumen .txt Anda", type=["txt"])
    
    # --- PERUBAHAN DI SINI ---
    st.divider()
    st.markdown("Dibuat oleh **@trianfe** ❤️  \n*Hacktiv8 Final Project*")


# Process the uploaded file and create the RAG chain
if uploaded_file is not None:
    if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
        try:
            with st.spinner("Memproses dokumen Anda... mohon tunggu."):
                temp_dir = "temp"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                temp_filepath = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(uploaded_file.getvalue())

                loader = TextLoader(temp_filepath)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                google_api_key = st.secrets["GOOGLE_API_KEY"]
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                llm_rag = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.7)
                
                prompt_template = ChatPromptTemplate.from_template("""
                Anda adalah asisten untuk tugas tanya-jawab.
                Gunakan potongan konteks yang diambil berikut ini untuk menjawab pertanyaan.
                Jika Anda tidak tahu jawabannya, katakan saja Anda tidak tahu.
                Gunakan maksimal tiga kalimat dan jaga agar jawabannya ringkas.

                Konteks: {context} 

                Pertanyaan: {input} 

                Jawaban yang Membantu:""")

                question_answer_chain = create_stuff_documents_chain(llm_rag, prompt_template)
                retriever = vectorstore.as_retriever()
                st.session_state.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                st.session_state.last_uploaded_filename = uploaded_file.name
                
                st.sidebar.success("Dokumen berhasil diproses!")
        except Exception as e:
            st.sidebar.error(f"Terjadi kesalahan: {e}")

# --- 5. Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. Handle User Input and Agent Communication ---
prompt = st.chat_input("Ketik pesan Anda di sini...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if st.session_state.get("rag_chain") and uploaded_file is not None:
            st.info("Merespons menggunakan dokumen yang diunggah...", icon="📄")
            response = st.session_state.rag_chain.invoke({"input": prompt})
            answer = response["answer"]
        elif "agent" in st.session_state:
            st.info("Merespons menggunakan agen umum...", icon="🤖")
            messages = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages]
            response = st.session_state.agent.invoke({"messages": messages})
            answer = response["messages"][-1].content if "messages" in response and response["messages"] else "Tidak ada respons yang dihasilkan."
        else:
            answer = "Chatbot tidak diinisialisasi dengan benar."
    except Exception as e:
        answer = f"Terjadi kesalahan: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
