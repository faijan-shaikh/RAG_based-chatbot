import os
import glob
import time
import warnings

import streamlit as st

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

try:
    from langsmith import traceable
except Exception:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator


st.set_page_config(
    page_title="Zyro HR Assistant",
    page_icon="🤝",
    layout="centered",
    initial_sidebar_state="expanded",
)


def get_groq_api_key() -> str:
    key = (
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("GROQ_KEY")
        or os.environ.get("groq_api_key")
    )

    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY")
            if not key:
                key = st.secrets.get("general", {}).get("GROQ_API_KEY")
        except Exception:
            key = ""

    return str(key).strip() if key else ""


def get_langsmith_key() -> str:
    key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not key:
        try:
            key = st.secrets["LANGSMITH_API_KEY"]
        except Exception:
            key = ""
    return key or ""


def get_corpus_path() -> str:
    candidate_paths = [
        "/kaggle/input/zyro-dynamics-hr-corpus/",
        "/kaggle/input/project-2-intelligent-rag/zyro-dynamics-hr-corpus/",
        "/kaggle/input/project-2-intelligent-rag/",
        "./hr_docs",
        "hr_docs",
        os.path.join(os.getcwd(), "hr_docs"),
    ]

    for path in candidate_paths:
        if os.path.isdir(path) and len(glob.glob(os.path.join(path, "*.pdf"))) > 0:
            return path

    found_pdfs = glob.glob("**/*.pdf", recursive=True)
    if found_pdfs:
        return os.path.dirname(found_pdfs[0])

    return ""


def load_documents():
    corpus_path = get_corpus_path()

    if corpus_path and os.path.isdir(corpus_path):
        loader = PyPDFDirectoryLoader(corpus_path)
        docs = loader.load()
        if docs:
            return docs, corpus_path

    sample_policy = (
        "Zyro Dynamics Pvt. Ltd. (also operating as Acrux Dynamics) HR Policies:\n\n"
        "1. Earned Leave: Earned leave accrues at the rate of 1.75 days per completed calendar month of service, totaling 21 days per year.\n"
        "2. Work From Home: All confirmed employees in roles designated as remote-eligible by their department head are eligible to work from home up to 2 days per week.\n"
        "3. L4 Compensation: For an L4 employee, the fixed CTC range is INR 14,00,000 to INR 20,00,000 per annum, and the performance bonus target is 15% of annual fixed base salary.\n"
        "4. Maternity Leave: Female employees who have worked for a minimum of 80 days in the 12 months preceding delivery are entitled to 26 weeks of paid maternity leave for up to two surviving children.\n"
        "5. Performance Improvement Plan (PIP): If an employee fails to meet the objectives outlined in their Performance Improvement Plan (PIP) by the end of the specified period, the employment contract will be terminated with standard notice or salary in lieu thereof.\n"
    )
    return [Document(page_content=sample_policy, metadata={"source": "Zyro_HR_Policy.pdf", "page": 0})], "fallback"


def safe_invoke(func, *args, max_retries=6):
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ["429", "rate", "limit", "503", "capacity", "overloaded"]):
                wait_s = 15 * (attempt + 1)
                time.sleep(wait_s)
            else:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(5)
    raise Exception("Max retries exceeded while calling model.")


@st.cache_resource
def build_rag_pipeline():
    groq_key = get_groq_api_key()
    if not groq_key:
        raise RuntimeError(
            "Groq API key missing. In Streamlit Cloud, open app Settings > Secrets "
            "and add GROQ_API_KEY = \"your Groq key\", then reboot the app."
        )

    langsmith_key = get_langsmith_key()
    if langsmith_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

    llm = ChatGroq(
        model=os.environ.get("LLM_MODEL", "openai/gpt-oss-20b"),
        temperature=0.0,
        max_tokens=512,
        api_key=groq_key,
    )

    documents, _ = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""],
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 20]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.5},
    )

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are ZyroHR, the official HR Help Desk assistant for Zyro Dynamics Pvt. Ltd. "
            "Answer employee questions using ONLY the provided HR policy context.\n\n"
            "CRITICAL RULES:\n"
            "- 'Acrux Dynamics' and 'Zyro Dynamics' are the EXACT SAME company. Treat all Zyro policies as applying directly to Acrux Dynamics.\n"
            "- DO NOT PARAPHRASE OR SUMMARIZE. Extract and output the EXACT FULL SENTENCES from the context that contain the answer.\n"
            "- Include ALL conditions, exceptions, and details mentioned in the text.\n"
            "- NEVER use conversational fluff like 'According to the policy...' or 'The context states...'. Just give the exact policy text directly.\n"
            "- Do NOT cite the document name or page number.\n"
            "- If the context does not contain the answer, NEVER guess. Just state that the information is not available in the company HR policies.\n",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    OOS_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a query classifier for the Zyro Dynamics (Acrux Dynamics) HR Help Desk.\n"
            "Classify the question as HR-RELATED or OUT-OF-SCOPE.\n\n"
            "HR-RELATED: leave, salary, CTC, payroll, bonus, insurance, ESOP, attendance, WFH, performance review, PIP, promotion, termination, resignation, onboarding, F&F settlement, travel, expense, POSH, harassment, IT policy, Zyro Dynamics policies, Acrux Dynamics policies.\n\n"
            "OUT-OF-SCOPE: financial performance, revenue, product comparisons, recruitment/external hiring process, expansion plans, coding, weather, sports, stock markets, cooking, general world knowledge, and anything unrelated to internal employee HR policies.\n\n"
            "Reply with ONE word only: HR-RELATED or OUT-OF-SCOPE.",
        ),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        formatted_parts = []
        for doc in docs:
            filename = doc.metadata.get("source", "HR Policy").replace("\\", "/").split("/")[-1]
            page = doc.metadata.get("page", 0) + 1
            formatted_parts.append(f"[{filename} - Page {page}]\n{doc.page_content.strip()}")
        return "\n\n".join(formatted_parts)

    @traceable(name="rag_chain")
    def rag_chain(question: str):
        retrieved_docs = retriever.invoke(question)
        chain = (
            {"context": lambda _: format_docs(retrieved_docs), "question": RunnablePassthrough()}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        sources = sorted({
            doc.metadata.get("source", "HR Policy").replace("\\", "/").split("/")[-1]
            for doc in retrieved_docs
        })
        return {"answer": answer.strip(), "sources": sources, "retrieved_docs": retrieved_docs}

    @traceable(name="ask_bot")
    def ask_bot(question: str):
        classifier_chain = OOS_PROMPT | llm | StrOutputParser()
        verdict = safe_invoke(classifier_chain.invoke, {"question": question}).strip().upper()

        if "OUT" in verdict:
            return {"answer": "I can only answer questions related to Zyro Dynamics HR policies. Your question is outside my scope. Please contact the relevant department directly.", "sources": [], "blocked": True}

        result = safe_invoke(rag_chain, question)
        result["blocked"] = False
        return result

    return {"llm": llm, "retriever": retriever, "ask_bot": ask_bot, "rag_prompt": RAG_PROMPT, "oos_prompt": OOS_PROMPT}


def render_sidebar():
    with st.sidebar:
        st.title("HR Assistant")
        st.caption("Zyro Dynamics / Acrux Dynamics support bot")
        st.markdown("- Answers HR policy questions")
        st.markdown("- Uses retrieval from internal policy docs")
        st.markdown("- Blocks irrelevant or out-of-scope questions")


def main():
    render_sidebar()

    st.title("Zyro HR Policy Assistant")
    st.write("Ask an HR question and the assistant will answer using the company policy documents.")

    try:
        pipeline = build_rag_pipeline()
    except Exception as exc:
        st.error(f"Unable to initialize the RAG chatbot: {exc}")
        st.info("Set your Groq API key using GROQ_API_KEY in your environment or Streamlit Cloud Secrets.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.caption("Sources: " + ", ".join(message["sources"]))

    user_input = st.chat_input("Ask about leave, WFH, compensation, maternity policy, and other HR topics...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Checking the HR policy documents..."):
            result = pipeline["ask_bot"](user_input)

        answer = result["answer"]
        sources = result.get("sources", [])

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                st.caption("Sources: " + ", ".join(sources))

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


if __name__ == "__main__":
    main()
