import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain

# ========== 🎨 Streamlit Custom Styles ==========
st.markdown(
    """
    <style>
        body {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stApp {
            background-color: #0e1117;
        }
        .title {
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            color: #00ffd5;
        }
        .subtitle {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
        }
        .stTextInput>div>div>input {
            background-color: #1a1c23;
            color: white;
            border-radius: 10px;
            border: 1px solid #4e4e4e;
        }
        .stButton>button {
            background-color: #00ffd5;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #00bfa6;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== API Key ==========
openai_api_key = "API KEY"

# ========== LLM Model ==========
llm = OpenAI(temperature=0.9, openai_api_key=openai_api_key, max_tokens=500)

# ========== FAISS Storage Path ==========
faiss_index_path = "faiss_index"

# ==========  Main UI ==========
st.markdown('<p class="title">📰 AI News Research Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Extract insights from news articles with AI-powered search</p>', unsafe_allow_html=True)
st.divider()

# Sidebar for URLs
st.sidebar.title("🌍 Enter News Article URLs")
urls = [st.sidebar.text_input(f"🔗 URL {i+1}") for i in range(3)]
urls = [url.strip() for url in urls if url.strip()]  # Remove empty values

process_url_click = st.sidebar.button("🚀 Process Articles")
main_placeholder = st.empty()

# Ensure embeddings is always defined
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# ========== Processing URLs ==========
if process_url_click:
    if not urls:
        st.error("⚠️ Please enter at least one valid URL.")
    else:
        # Load documents
        main_placeholder.success("⏳ Fetching articles... Please wait.")
        try:
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            main_placeholder.success(f"✅ Successfully loaded {len(data)} articles!")
        except Exception as e:
            st.error(f"❌ Error loading URLs: {e}")
            data = []

        if len(data) == 0:
            raise ValueError("⚠️ No data found. Check the URLs.")

        # Split text
        main_placeholder.success("📖 Splitting text into sections...")
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        if len(docs) == 0:
            raise ValueError("⚠️ No documents created after splitting.")

        # Create FAISS index with embeddings
        main_placeholder.success("🧠 Generating AI embeddings...")
        try:
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
        except IndexError:
            raise ValueError("❌ Embeddings failed, possibly due to empty input.")

        # Save FAISS index correctly
        vectorstore_openai.save_local(faiss_index_path)
        main_placeholder.success("🎉 AI Model Ready! Ask me anything about the articles.")

st.divider()

# ========== Ask a Question ==========
query = st.text_input("💬 Ask a question about the articles:")

if query:
    if os.path.exists(faiss_index_path):
        # Load FAISS index correctly
        vectorIndex = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        # Create Retrieval Chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())

        # Run Query
        result = chain({"question": query}, return_only_outputs=True)

        # Display Answer
        st.subheader("🧠 AI-Powered Answer")
        st.success(result["answer"])
    else:
        st.error("⚠️ No AI model found. Please process articles first.")
