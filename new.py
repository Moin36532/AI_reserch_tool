# %%writefile app.py

import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


st.set_page_config(page_title="Web Research Assistant", page_icon="🔍")
st.title("Web Research Assistant 🔍")

url = st.text_input("Enter a webpage URL")
query = st.text_input("Enter your query")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]


if st.button("Analyze"):
    if not url or not url.startswith("http"):
        st.error("Please enter a valid URL (starting with http or https)")
    elif not query.strip():
        st.error("Please enter a query to analyze")
    else:
        with st.spinner("Fetching and analyzing the webpage..."):
            try:
                placeholder = st.empty()
                # 1️⃣ Load webpage content
                progress = st.progress(0)
                placeholder.markdown("Step 1/5: Loading the webpage...")
                progress.progress(20)
                loader = WebBaseLoader(url)
                docs = loader.load()
                res = docs[0].page_content

                # 2️⃣ Split text
                placeholder.markdown("Step 2/5: Splitting text...")
                progress.progress(40)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_text(res)

                # 3️⃣ Create embeddings and Chroma vector store
                placeholder.markdown("Step 3/5: Generating embeddings")
                progress.progress(60)
                embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
                chroma_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory="/content/db3",
                    collection_name="sample"
                )
                documents = [Document(page_content=txt) for txt in chunks]
                chroma_store.add_documents(documents)
                chroma_store.persist()

                # 4️⃣ Retrieve relevant context
                placeholder.markdown("Step 4/5: Retriving relevant result")
                progress.progress(80)
                retriever = chroma_store.as_retriever()
                retrieved_docs = retriever.get_relevant_documents(query)
                context = [doc.page_content for doc in retrieved_docs]

                # 5️⃣ Prepare prompt and LLM
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        "You are a research analyst. Answer using only the given context. If not enough information, say 'I don't know.'"
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "User query:\n{query}\n\nContext:\n{context}"
                    ),
                ])

                llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1")
                model = ChatHuggingFace(llm=llm)
                placeholder.markdown("Step 5/5 Loading results...")
                progress.progress(100)
                prompt_value = prompt.format_prompt(query=query, context=context)
                result = model.invoke(prompt_value)

                st.success("✅ Analysis complete!")
                st.subheader("📘 Answer:")
                st.write(result.content)

            except Exception as e:
                st.error(f"Something went wrong: {e}")






