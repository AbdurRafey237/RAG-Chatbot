# ═════════════════════════════════════════════════════════════════════════════
#  Citera · Grounded answers, traced to their sources
#  ----------------------------------------------------------------------------
#  A retrieval-grounded document workspace. Drop in your own files and Citera
#  answers strictly from what they say — every reply backed by the passages it
#  was drawn from, so you can verify instead of trust.
#
#  Designed & built by Abdur Rafey.
# ═════════════════════════════════════════════════════════════════════════════

import warnings

# Silence the noisy forward-compatibility chatter from the dependency stack.
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import glob
from pathlib import Path

import streamlit as st

# ── Model providers ──────────────────────────────────────────────────────────
# Chat + embedding endpoints for the three supported back ends.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# ── Prompting, memory & orchestration ────────────────────────────────────────
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ── Ingestion: loaders, splitters & the vector index ─────────────────────────
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma

# ── Retrieval refinement: compression pipeline + reranking ───────────────────
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    CohereRerank,
)
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Configuration — providers, models, languages and on-disk locations
# ═════════════════════════════════════════════════════════════════════════════

# Order here is load-bearing: the sidebar maps the selected radio entry back to
# a provider by index, so keep these three aligned with the branches below.
LLM_PROVIDERS = [
    "OpenAI",
    "Google Gemini",
    "Hugging Face",
]

# The retriever's opening line, localised. Tone stays consistent everywhere:
# Citera works *from* your documents, so it asks what to look for in them.
WELCOME_MESSAGE = {
    "english": "What would you like to know from your documents?",
    "french": "Que souhaitez-vous savoir à partir de vos documents ?",
    "spanish": "¿Qué le gustaría saber a partir de sus documentos?",
    "german": "Was möchten Sie aus Ihren Dokumenten erfahren?",
    "russian": "Что вы хотите узнать из ваших документов?",
    "chinese": "您想从您的文档中了解什么？",
    "arabic": "ماذا تريد أن تعرف من مستنداتك؟",
    "portuguese": "O que você gostaria de saber a partir dos seus documentos?",
    "italian": "Cosa vorrebbe sapere dai suoi documenti?",
    "Japanese": "ドキュメントから何を知りたいですか？",
}

# Retrieval strategies, ordered most-to-least refined. These exact strings are
# matched downstream in build_retriever(), so edit them in lockstep there.
RETRIEVER_TYPES = [
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever",
]

# Workspace lives beside this file: scratch space for uploads + the Chroma index.
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath("data", "vector_stores")

# Make sure both directories exist before anything tries to read or write them —
# this is what lets the app run unchanged on a fresh deployment.
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Visual identity — the "ink & citation" theme
# ═════════════════════════════════════════════════════════════════════════════
#  A deep petrol base with a calm citation-teal accent and a warm highlighter
#  amber. Fraunces carries the wordmark, IBM Plex Sans the interface, and IBM
#  Plex Mono the source labels — provenance rendered like a footnote.

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink:        #0E1A24;
    --ink-raised: #14242F;
    --ink-line:   #21384A;
    --paper:      #EAF1F4;
    --muted:      #8FA6B2;
    --cite:       #4FD1C5;
    --mark:       #F2B65A;
}

/* Base surfaces ---------------------------------------------------------- */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: radial-gradient(1200px 600px at 80% -10%, #16303C 0%, var(--ink) 55%);
    color: var(--paper);
}
[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
    background: var(--ink-raised);
    border-right: 1px solid var(--ink-line);
}

html, body, [class*="css"], .stMarkdown, p, li, label, span, div {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: var(--paper);
}

h1, h2, h3, h4, h5, h6 { color: var(--paper); letter-spacing: -0.01em; }
.stMarkdown a, a { color: var(--cite); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* Hero wordmark ---------------------------------------------------------- */
.citera-hero { padding: 0.25rem 0 0.5rem 0; }
.citera-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.28em; text-transform: uppercase;
    color: var(--cite); margin-bottom: 0.35rem;
}
.citera-wordmark {
    font-family: 'Fraunces', serif; font-weight: 600;
    font-size: 2.9rem; line-height: 1; color: var(--paper); margin: 0;
}
.citera-wordmark .dot { color: var(--cite); }
.citera-tagline { color: var(--muted); font-size: 0.98rem; margin-top: 0.5rem; }
.citera-rule {
    height: 2px; width: 100%; margin: 0.9rem 0 0.4rem 0;
    background: linear-gradient(90deg, var(--cite) 0%, transparent 60%);
}
.citera-credit {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: var(--muted);
}
.citera-credit b { color: var(--mark); font-weight: 500; }

/* Buttons ---------------------------------------------------------------- */
.stButton > button {
    background: transparent; color: var(--cite);
    border: 1px solid var(--cite); border-radius: 10px;
    font-weight: 500; transition: all 0.15s ease;
}
.stButton > button:hover {
    background: var(--cite); color: var(--ink); border-color: var(--cite);
}

/* Inputs & widgets ------------------------------------------------------- */
[data-baseweb="input"], [data-baseweb="select"] > div, .stTextInput input {
    background: var(--ink) !important;
    border-color: var(--ink-line) !important;
    border-radius: 10px !important; color: var(--paper) !important;
}
.stTextInput input::placeholder { color: var(--muted) !important; }

/* Tabs ------------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--ink-line); }
.stTabs [data-baseweb="tab"] { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--cite); }

/* Chat bubbles ----------------------------------------------------------- */
[data-testid="stChatMessage"] {
    background: var(--ink-raised); border: 1px solid var(--ink-line);
    border-radius: 14px; padding: 0.4rem 0.2rem;
}

/* The signature: "Sources" panel styled like a margin of footnotes ------- */
[data-testid="stExpander"] {
    border: 1px solid var(--ink-line);
    border-left: 3px solid var(--cite);
    border-radius: 10px; background: rgba(79, 209, 197, 0.04);
}
[data-testid="stExpander"] summary { color: var(--cite); font-weight: 500; }

/* Dividers + scrollbar --------------------------------------------------- */
hr { border-color: var(--ink-line); }
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: var(--ink-line); border-radius: 8px; }
</style>
"""

HERO_HTML = """
<div class="citera-hero">
    <div class="citera-eyebrow">Every answer, traced to its source</div>
    <h1 class="citera-wordmark">Citera<span class="dot">.</span></h1>
    <div class="citera-tagline">
        A grounded document workspace — Citera reads your files and answers only
        from what they actually say, with the source passages shown alongside.
    </div>
    <div class="citera-rule"></div>
    <div class="citera-credit">Designed &amp; built by <b>Abdur Rafey</b></div>
</div>
"""


# ═════════════════════════════════════════════════════════════════════════════
#  App shell — page setup, theme injection and the hero header
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Citera · Grounded answers", page_icon="📑")

st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(HERO_HTML, unsafe_allow_html=True)

# Credentials are held only for the lifetime of the session and never persisted.
st.session_state.openai_api_key = ""
st.session_state.google_api_key = ""
st.session_state.cohere_api_key = ""
st.session_state.hf_api_key = ""


def render_provider_controls(
    provider="OpenAI",
    api_key_label="OpenAI API key — [create one](https://platform.openai.com/account/api-keys)",
    models=("gpt-4o-mini", "gpt-4o"),
):
    """Render the credential field plus the model / sampling controls for one
    provider, and stash the selections on the session."""

    st.session_state.LLM_provider = provider

    # Only the active provider keeps a live key; the others are blanked so a
    # stale credential can't leak into the next request.
    if provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            api_key_label, type="password", placeholder="paste your key"
        )
        st.session_state.google_api_key = ""
        st.session_state.hf_api_key = ""

    if provider == "Google":
        st.session_state.google_api_key = st.text_input(
            api_key_label, type="password", placeholder="paste your key"
        )
        st.session_state.openai_api_key = ""
        st.session_state.hf_api_key = ""

    if provider == "HuggingFace":
        st.session_state.hf_api_key = st.text_input(
            api_key_label, type="password", placeholder="paste your key"
        )
        st.session_state.openai_api_key = ""
        st.session_state.google_api_key = ""

    with st.expander("Model & sampling"):
        st.session_state.selected_model = st.selectbox(
            f"{provider} model", models
        )

        # temperature → how adventurous; top_p → nucleus cut-off.
        st.session_state.temperature = st.slider(
            "temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1
        )
        st.session_state.top_p = st.slider(
            "top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.05
        )


def compose_sidebar():
    """Build the control sidebar and the two-tab source panel:
    one tab forges a fresh index from uploads, the other reopens a saved one."""

    with st.sidebar:
        st.caption(
            "Grounded retrieval over 🔗 LangChain · OpenAI · Google Gemini · "
            "Hugging Face 🤗 · reranked with Cohere"
        )
        st.write("")

        # Pick the engine that will read and answer over your documents.
        provider_choice = st.radio(
            "Provider",
            LLM_PROVIDERS,
            captions=[
                "[Usage & pricing](https://openai.com/pricing)",
                "Generous free tier via Google AI Studio.",
                "**Free** inference through the 🤗 Hub.",
            ],
        )

        st.divider()
        if provider_choice == LLM_PROVIDERS[0]:
            render_provider_controls(
                provider="OpenAI",
                api_key_label="OpenAI API key — [create one](https://platform.openai.com/account/api-keys)",
                models=["gpt-4o-mini", "gpt-4o"],
            )
        if provider_choice == LLM_PROVIDERS[1]:
            render_provider_controls(
                provider="Google",
                api_key_label="Google API key — [create one](https://makersuite.google.com/app/apikey)",
                models=["gemini-2.5-flash", "gemini-2.5-pro"],
            )
        if provider_choice == LLM_PROVIDERS[2]:
            render_provider_controls(
                provider="HuggingFace",
                api_key_label="Hugging Face token — [create one](https://huggingface.co/settings/tokens)",
                models=["mistralai/Mistral-7B-Instruct-v0.3"],
            )

        # Answers are returned in the language chosen here.
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            "Answer language", list(WELCOME_MESSAGE.keys())
        )

        st.divider()
        st.subheader("Retrieval")
        st.session_state.retriever_type = st.selectbox(
            "Strategy", RETRIEVER_TYPES
        )

        # The Cohere reranker is the only strategy that needs its own key.
        st.write("")
        if st.session_state.retriever_type == RETRIEVER_TYPES[0]:
            st.session_state.cohere_api_key = st.text_input(
                "Cohere API key — [create one](https://dashboard.cohere.com/api-keys)",
                type="password",
                placeholder="paste your key",
            )

        st.write("")
        st.caption(
            f"Your {st.session_state.LLM_provider} key, the "
            f"'{st.session_state.selected_model}' settings and the "
            f"{st.session_state.retriever_type} are applied the moment you "
            "build or open an index."
        )
        st.write("")
        st.caption("— Designed & built by Abdur Rafey")

    # Two ways in: build something new, or pick up where you left off.
    tab_build, tab_open = st.tabs(["Build an index", "Open a saved index"])

    with tab_build:
        # 1 · choose the source files
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Add documents**",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "csv"],
        )
        # 2 · name the index they will be embedded into
        st.session_state.vector_store_name = st.text_input(
            label="**Name this knowledge base** — files are parsed, embedded "
            "and written to a Chroma index under this name.",
            placeholder="e.g. research-notes",
        )
        # 3 · kick off ingestion
        st.button("Build index", on_click=ingest_and_build_chain)

        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except Exception:
            pass

    with tab_open:
        # Reopen any index already sitting in the local store. Listing them with
        # a selectbox keeps this fully server-side — no native file dialog, so it
        # behaves identically whether running locally or on a hosted deployment.
        st.write("Reopen a previously built knowledge base:")

        saved_indexes = sorted(
            p.name for p in LOCAL_VECTOR_STORE_DIR.iterdir() if p.is_dir()
        )

        if not saved_indexes:
            st.info("No saved indexes yet — build one from the other tab first.")
        else:
            chosen = st.selectbox("Knowledge base", saved_indexes)
            load_clicked = st.button("Open index")
            st.session_state.selected_vectorstore_name = ""

            if load_clicked:
                # Validate credentials before touching the store.
                problems = []
                if (
                    not st.session_state.openai_api_key
                    and not st.session_state.google_api_key
                    and not st.session_state.hf_api_key
                ):
                    problems.append(f"add your {st.session_state.LLM_provider} API key")
                if (
                    st.session_state.retriever_type == RETRIEVER_TYPES[0]
                    and not st.session_state.cohere_api_key
                ):
                    problems.append("add your Cohere API key")

                if len(problems) == 1:
                    st.warning("Please " + problems[0] + ".")
                elif len(problems) > 1:
                    st.warning(
                        "Please "
                        + ", ".join(problems[:-1])
                        + ", and "
                        + problems[-1]
                        + "."
                    )
                else:
                    index_path = LOCAL_VECTOR_STORE_DIR.joinpath(chosen).as_posix()
                    with st.spinner("Opening index…"):
                        st.session_state.selected_vectorstore_name = chosen
                        try:
                            # Re-attach the embeddings, the retriever and the chain
                            # to the persisted Chroma collection.
                            embeddings = resolve_embeddings()
                            st.session_state.vector_store = Chroma(
                                embedding_function=embeddings,
                                persist_directory=index_path,
                            )
                            st.session_state.retriever = build_retriever(
                                vector_store=st.session_state.vector_store,
                                embeddings=embeddings,
                                retriever_type=st.session_state.retriever_type,
                                base_retriever_search_type="similarity",
                                base_retriever_k=16,
                                compression_retriever_k=20,
                                cohere_api_key=st.session_state.cohere_api_key,
                                cohere_model="rerank-multilingual-v3.0",
                                cohere_top_n=10,
                            )
                            (
                                st.session_state.chain,
                                st.session_state.memory,
                            ) = assemble_chain(
                                retriever=st.session_state.retriever,
                                chain_type="stuff",
                                language=st.session_state.assistant_language,
                            )
                            reset_conversation()
                            st.info(f"**{chosen}** is ready.")
                        except Exception as e:
                            st.error(e)


# ═════════════════════════════════════════════════════════════════════════════
#  Ingestion — turn raw uploads into a searchable Chroma index
# ═════════════════════════════════════════════════════════════════════════════

def purge_workspace():
    """Clear out the scratch directory so a new build starts clean."""
    for f in glob.glob(TMP_DIR.as_posix() + "/*"):
        try:
            os.remove(f)
        except Exception:
            pass


def load_source_documents():
    """Read every supported file (txt / pdf / csv / docx) out of the scratch
    directory and return them as LangChain documents."""

    documents = []

    documents.extend(
        DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.txt",
            loader_cls=TextLoader, show_progress=True,
        ).load()
    )
    documents.extend(
        DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.pdf",
            loader_cls=PyPDFLoader, show_progress=True,
        ).load()
    )
    documents.extend(
        DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.csv",
            loader_cls=CSVLoader, show_progress=True,
            loader_kwargs={"encoding": "utf8"},
        ).load()
    )
    documents.extend(
        DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.docx",
            loader_cls=Docx2txtLoader, show_progress=True,
        ).load()
    )

    return documents


def chunk_documents(documents):
    """Slice documents into overlapping, retrieval-sized chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    return splitter.split_documents(documents)


def resolve_embeddings():
    """Hand back the embedding model that matches the active provider."""

    if st.session_state.LLM_provider == "OpenAI":
        # text-embedding-3-small: current-generation, cheaper and sharper than ada-002.
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=st.session_state.openai_api_key,
        )

    if st.session_state.LLM_provider == "Google":
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=st.session_state.google_api_key,
        )

    if st.session_state.LLM_provider == "HuggingFace":
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=st.session_state.hf_api_key,
            model_name="thenlper/gte-large",
        )


def build_retriever(
    vector_store,
    embeddings,
    retriever_type="Contextual compression",
    base_retriever_search_type="similarity",
    base_retriever_k=16,
    compression_retriever_k=20,
    cohere_api_key="",
    cohere_model="rerank-multilingual-v3.0",
    cohere_top_n=10,
):
    """Assemble the retriever for the chosen strategy.

    Three options, increasingly selective:
      • Vectorstore backed retriever — plain semantic search (the base layer).
      • Contextual compression — re-chunks, de-duplicates, filters by relevance
        and reorders so the strongest evidence sits at the head and tail.
      • Cohere reranker — sends the base hits to Cohere's rerank endpoint for a
        relevance-ordered shortlist.
    """

    base_retriever = vectorstore_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    if retriever_type == "Contextual compression":
        return build_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            k=compression_retriever_k,
        )

    if retriever_type == "Cohere reranker":
        return build_cohere_reranker(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            cohere_model=cohere_model,
            top_n=cohere_top_n,
        )


def vectorstore_retriever(vectorstore, search_type="similarity", k=4, score_threshold=None):
    """Wrap a Chroma store as a retriever.

    search_type: "similarity" | "mmr" | "similarity_score_threshold".
    k: how many chunks to pull back.
    score_threshold: minimum relevance, only for the score-threshold mode.
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    return vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )


def build_compression_retriever(
    embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None
):
    """Wrap the base retriever in a four-stage compression pipeline so only the
    passages that matter survive to the prompt: split → de-dupe → relevance
    filter → long-context reorder."""

    # 1 · break hits into finer chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")

    # 2 · drop near-duplicate chunks
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3 · keep only the top-k chunks closest to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
    )

    # 4 · push the weakest chunks to the middle, strongest to the edges
    # (see https://arxiv.org/abs/2307.03172 on "lost in the middle")
    reordering = LongContextReorder()

    pipeline = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    return ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=base_retriever
    )


def build_cohere_reranker(
    base_retriever, cohere_api_key, cohere_model="rerank-multilingual-v3.0", top_n=10
):
    """Re-rank the base retriever's hits through Cohere's rerank endpoint and
    keep the top_n most relevant."""

    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, model=cohere_model, top_n=top_n
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )


def ingest_and_build_chain():
    """End-to-end build triggered by the "Build index" button:
    validate inputs → stage uploads → load → chunk → embed → index → wire up
    the retriever, memory and conversational chain."""

    with st.spinner("Building index…"):
        # Validate before doing any expensive work.
        problems = []
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
            and not st.session_state.hf_api_key
        ):
            problems.append(f"add your {st.session_state.LLM_provider} API key")
        if (
            st.session_state.retriever_type == RETRIEVER_TYPES[0]
            and not st.session_state.cohere_api_key
        ):
            problems.append("add your Cohere API key")
        if not st.session_state.uploaded_file_list:
            problems.append("add documents to upload")
        if st.session_state.vector_store_name == "":
            problems.append("name the knowledge base")

        if len(problems) == 1:
            st.session_state.error_message = "Please " + problems[0] + "."
            return
        if len(problems) > 1:
            st.session_state.error_message = (
                "Please " + ", ".join(problems[:-1]) + ", and " + problems[-1] + "."
            )
            return

        st.session_state.error_message = ""
        try:
            # 1 · clear the scratch directory
            purge_workspace()

            # 2 · stage each upload to disk so the loaders can reach it
            staging_errors = ""
            for uploaded_file in st.session_state.uploaded_file_list:
                try:
                    temp_file_path = os.path.join(TMP_DIR.as_posix(), uploaded_file.name)
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())
                except Exception as e:
                    staging_errors += str(e)
            if staging_errors:
                st.warning(f"Some files could not be staged: {staging_errors}")

            # 3 · load → 4 · chunk → 5 · embed
            documents = load_source_documents()
            chunks = chunk_documents(documents)
            embeddings = resolve_embeddings()

            # 6 · persist the Chroma index under the chosen name
            persist_directory = (
                LOCAL_VECTOR_STORE_DIR.as_posix() + "/" + st.session_state.vector_store_name
            )
            try:
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                )
                st.info(
                    f"Knowledge base **{st.session_state.vector_store_name}** is built."
                )

                # 7 · retriever → 8 · chain + memory
                st.session_state.retriever = build_retriever(
                    vector_store=st.session_state.vector_store,
                    embeddings=embeddings,
                    retriever_type=st.session_state.retriever_type,
                    base_retriever_search_type="similarity",
                    base_retriever_k=16,
                    compression_retriever_k=20,
                    cohere_api_key=st.session_state.cohere_api_key,
                    cohere_model="rerank-multilingual-v3.0",
                    cohere_top_n=10,
                )
                (
                    st.session_state.chain,
                    st.session_state.memory,
                ) = assemble_chain(
                    retriever=st.session_state.retriever,
                    chain_type="stuff",
                    language=st.session_state.assistant_language,
                )

                # 9 · fresh conversation against the new index
                reset_conversation()

            except Exception as e:
                st.error(e)

        except Exception as error:
            st.error(f"Something went wrong while building the index: {error}")


# ═════════════════════════════════════════════════════════════════════════════
#  Conversation — memory, prompts and the grounded retrieval chain
# ═════════════════════════════════════════════════════════════════════════════

def build_memory():
    """A rolling conversation buffer keyed for the retrieval chain. Modern
    long-context models make full-history buffering the simplest reliable
    choice, so that is what we use across the board."""
    return ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
        input_key="question",
    )


def grounded_answer_template(language="english"):
    """The answering prompt. It hands the model the standalone question, the
    chat history and the retrieved context, and pins it to that evidence —
    Citera answers from the sources or admits the sources fall short."""

    return f"""You are Citera. Answer the question at the end using only the
context below (delimited by <context></context>). If the context does not
contain enough to answer, say so plainly rather than guessing. Reply in the
language named at the end.

<context>
{{chat_history}}

{{context}}
</context>

Question: {{question}}

Language: {language}.
"""


def assemble_chain(retriever, chain_type="stuff", language="english"):
    """Wire up the conversational retrieval chain:
    a follow-up question is first condensed into a standalone query, the
    retriever fetches matching context, and a second model answers from it."""

    # 1 · condense follow-ups into a self-contained question
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""",
    )

    # 2 · the grounded answering prompt
    answer_prompt = ChatPromptTemplate.from_template(
        grounded_answer_template(language=language)
    )

    # 3 · conversation memory
    memory = build_memory()

    # 4 · one model to condense, one to answer (per provider)
    if st.session_state.LLM_provider == "OpenAI":
        condense_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
        )
        answer_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            model_kwargs={"top_p": st.session_state.top_p},
        )

    if st.session_state.LLM_provider == "Google":
        condense_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
            convert_system_message_to_human=True,
        )
        answer_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            convert_system_message_to_human=True,
        )

    if st.session_state.LLM_provider == "HuggingFace":
        condense_llm = HuggingFaceHub(
            repo_id=st.session_state.selected_model,
            huggingfacehub_api_token=st.session_state.hf_api_key,
            model_kwargs={
                "temperature": 0.1,
                "top_p": 0.95,
                "do_sample": True,
                "max_new_tokens": 1024,
            },
        )
        answer_llm = HuggingFaceHub(
            repo_id=st.session_state.selected_model,
            huggingfacehub_api_token=st.session_state.hf_api_key,
            model_kwargs={
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "do_sample": True,
                "max_new_tokens": 1024,
            },
        )

    # 5 · stitch it all into the retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=condense_llm,
        llm=answer_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory


def reset_conversation():
    """Drop the transcript and memory back to a clean welcome state."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE[st.session_state.assistant_language],
        }
    ]
    try:
        st.session_state.memory.clear()
    except Exception:
        pass


def respond(prompt):
    """Run one turn: send the prompt through the chain, render the answer and
    surface the exact passages it was grounded in."""
    try:
        # 1 · ask the chain
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        # Hugging Face models echo the prompt; keep only what follows "Answer:".
        if st.session_state.LLM_provider == "HuggingFace":
            answer = answer[answer.find("\nAnswer: ") + len("\nAnswer: "):]

        # 2 · record + render the exchange
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)

            # The provenance trail: where this answer actually came from.
            with st.expander("Sources"):
                provenance = ""
                for document in response["source_documents"]:
                    try:
                        page = " (page " + str(document.metadata["page"]) + ")"
                    except Exception:
                        page = ""
                    provenance += (
                        "**" + str(document.metadata["source"]) + page + "**\n\n"
                    )
                    provenance += document.page_content + "\n\n\n"
                st.markdown(provenance)

    except Exception as e:
        st.warning(e)


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_app():
    compose_sidebar()

    st.divider()
    col_title, col_action = st.columns([7, 3])
    with col_title:
        st.subheader("Ask your documents")
    with col_action:
        st.button("Clear conversation", on_click=reset_conversation)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE[st.session_state.assistant_language],
            }
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        # No key, no call — prompt the user for credentials first.
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
            and not st.session_state.hf_api_key
        ):
            st.info(f"Add your {st.session_state.LLM_provider} API key to continue.")
            st.stop()
        # No index, no chain — build or open one before chatting.
        if not st.session_state.get("chain"):
            st.info(
                "Build a new index or open a saved one first — then ask away."
            )
            st.stop()
        with st.spinner("Reading your sources…"):
            respond(prompt=prompt)


if __name__ == "__main__":
    run_app()
