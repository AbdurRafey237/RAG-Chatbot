<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&duration=3000&pause=900&color=4FD1C5&center=true&vCenter=true&width=820&lines=Citera;Chat+with+your+documents;have+the+answers+traced+to+their+source" alt="Typing SVG" />

### A source-grounded RAG chatbot — ask questions about your documents and get answers backed by the exact passages they came from.

<br/>

[![Live App](https://img.shields.io/badge/🚀_Live_App-Open-4FD1C5?style=for-the-badge&labelColor=0E1A24)](REPLACE_WITH_YOUR_STREAMLIT_URL)
&nbsp;
[![Demo Video](https://img.shields.io/badge/🎬_Demo-Watch-F2B65A?style=for-the-badge&labelColor=0E1A24)](REPLACE_WITH_YOUR_DEMO_LINK)
&nbsp;
[![Report Bug](https://img.shields.io/badge/🐛_Report-Bug-e74c3c?style=for-the-badge&labelColor=0E1A24)](https://github.com/AbdurRafey237/RAG-Chatbot/issues)

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white&labelColor=1a1a2e)
![Domain](https://img.shields.io/badge/Domain-Retrieval--Augmented%20Generation-8e44ad?style=flat&logo=databricks&logoColor=white&labelColor=1a1a2e)
![Interface](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white&labelColor=1a1a2e)
![License](https://img.shields.io/badge/License-MIT-1abc9c?style=flat&logo=opensourceinitiative&logoColor=white&labelColor=1a1a2e)
![Status](https://img.shields.io/badge/Status-Live-27ae60?style=flat&logo=statuspage&logoColor=white&labelColor=1a1a2e)

</div>

<br/>

> [!NOTE]
> **What is this?** Offline large language models answer from frozen training data, so they happily invent facts about your private files or anything recent. Citera fixes that: it retrieves the relevant passages from **your** uploaded documents and feeds them to the model, so every answer stays anchored to a source you can read for yourself.

<br/>

<div align="center">
  <img src="images/architecture.png" width="520" alt="RAG architecture" />
  <p><sub><b>How it works</b> — your question is rewritten into a standalone query, matched against your documents, and answered using only the retrieved passages.</sub></p>
</div>

<br/>

## 📑 Table of Contents

<details open>
<summary><b>Click to expand / collapse</b></summary>

- [✨ Features](#-features)
- [🧰 Tech Stack](#-tech-stack)
- [⚡ Quick Start (run it locally)](#-quick-start-run-it-locally)
- [🔑 Getting your API keys](#-getting-your-api-keys)
- [🖱️ Using the app — every control explained](#️-using-the-app--every-control-explained)
- [☁️ Deploying it online](#️-deploying-it-online)
- [📂 Project structure](#-project-structure)
- [⚠️ Good to know](#️-good-to-know)
- [🙏 Credits](#-credits)
- [📜 License](#-license)

</details>

<br/>

## ✨ Features

- 📎 **Bring your own documents** — upload **PDF, TXT, DOCX, or CSV** files and ask questions about them in plain language.
- 🎯 **Grounded answers** — replies come only from your files, with the **source passages shown** under every answer.
- 🔀 **Three AI providers** — switch freely between **OpenAI**, **Google Gemini**, and **Hugging Face** models.
- 🌍 **10 answer languages** — get responses in English, French, Spanish, German, Arabic, Chinese, and more.
- 🧠 **Smart retrieval** — choose how documents are searched, from a fast vector search to a quality-boosting reranker.
- 💾 **Reusable knowledge bases** — build an index once, then reopen it later without re-uploading.

<br/>

## 🧰 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Chroma](https://img.shields.io/badge/Chroma-FF6F61?style=for-the-badge&logo=chromatic&logoColor=white)

![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Cohere](https://img.shields.io/badge/Cohere-39594D?style=for-the-badge&logo=cohere&logoColor=white)

</div>

<table align="center"> <thead> <tr> <th align="center">Layer</th> <th align="center">What it does</th> </tr> </thead> <tbody> <tr> <td align="center"><strong>LangChain</strong></td> <td align="center">Orchestrates the whole retrieval-and-answer pipeline</td> </tr> <tr> <td align="center"><strong>Chroma</strong></td> <td align="center">Local vector database that stores and searches your documents</td> </tr> <tr> <td align="center"><strong>OpenAI / Google Gemini / Hugging Face</strong></td> <td align="center">The chat models that read the passages and write the answer</td> </tr> <tr> <td align="center"><strong>Cohere</strong></td> <td align="center">Optional reranker that sharpens which passages are used</td> </tr> <tr> <td align="center"><strong>Streamlit</strong></td> <td align="center">The web interface you click through</td> </tr> </tbody> </table>

<br/>

## ⚡ Quick Start (run it locally)

> [!IMPORTANT]
> Use **Python 3.11**. The app pins specific LangChain versions that do **not** install on Python 3.12 or newer.

**1. Get the code**
```bash
git clone https://github.com/AbdurRafey237/RAG-Chatbot.git
cd RAG-Chatbot
```

**2. Create and activate a virtual environment**
```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)
```

**3. Install the dependencies** (takes a few minutes — it's a big stack)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Launch the app**
```bash
streamlit run RAG_app.py
```

Your browser opens at **http://localhost:8501**. That's it — head to [Using the app](#️-using-the-app--every-control-explained) next.

<br/>

## 🔑 Getting your API keys

You only need a key for the **one provider you pick**. A **Cohere** key is needed only if you leave the "Cohere reranker" search strategy selected. All have a free tier.

| Provider | Get a key | Free option |
|----------|-----------|:-----------:|
| **Google Gemini** | [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey) | ✅ generous free tier |
| **OpenAI** | [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) | 💳 paid |
| **Hugging Face** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ free |
| **Cohere** *(reranking)* | [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) | ✅ free trial keys |

> 💡 **Easiest setup:** use **Google Gemini** as the provider — its key also covers the document embeddings, so a Gemini key alone is enough to run the whole app (just pick a non-Cohere search strategy).

<br/>

## 🖱️ Using the app — every control explained

<div align="center">
  <img src="images/streamlit_app.png" width="760" alt="App screenshot" />
</div>

<br/>

### 🟢 The 5-step path to your first answer

1. **Sidebar → Provider:** pick **Google Gemini** and paste your Gemini key.
2. **Sidebar → Model:** choose **gemini-2.5-flash** (fast and cheap).
3. **Sidebar → Retrieval → Strategy:** choose **Vectorstore backed retriever** (no extra key needed).
4. **Main panel → "Build an index" tab:** upload a document, give it a name, click **Build index**, and wait for it to finish.
5. **Type a question** in the chat box at the bottom. Your answer appears with a **Sources** section showing the passages it used. ✅

<br/>

### ⚙️ The sidebar (left side) — control by control

| Control | What it does |
|---------|--------------|
| **Provider** | Which AI company powers the answers: **OpenAI**, **Google Gemini**, or **Hugging Face**. |
| **API key** | Paste the key for the provider you picked (see [keys above](#-getting-your-api-keys)). |
| **Model** | The specific model. OpenAI: `gpt-4o-mini` / `gpt-4o`. Gemini: `gemini-2.5-flash` / `gemini-2.5-pro`. Hugging Face: `Mistral-7B-Instruct-v0.3`. The `-flash` / `-mini` options are faster and cheaper. |
| **temperature** | How creative the wording is. **Low (0.1)** = focused and factual, **high (0.9)** = more varied. Leave at **0.5**. |
| **top_p** | A second creativity dial (which words the model is allowed to pick from). Leave at **0.95**. |
| **Answer language** | The language the reply is written in — 10 options including English, French, Spanish, German, Arabic, and Chinese. |
| **Retrieval → Strategy** | How your documents are searched (details below). |
| **Cohere API key** | Appears **only** if you choose the "Cohere reranker" strategy. Paste a Cohere key there. |

#### 🔍 Which retrieval strategy should I pick?

| Strategy | Use it when | Needs Cohere key? |
|----------|-------------|:-----------------:|
| **Vectorstore backed retriever** | You just want it to work — fast and simple. **Start here.** | ❌ |
| **Contextual compression** | You want answers trimmed to only the most relevant text. | ❌ |
| **Cohere reranker** | You want the highest-quality passage selection. | ✅ |

<br/>

### 📄 The main panel (right side)

**Tab 1 — "Build an index"** *(create a new knowledge base from your files)*

| Step | What to do |
|------|------------|
| **Add documents** | Upload one or more **PDF / TXT / DOCX / CSV** files. |
| **Name this knowledge base** | Give it any name, e.g. `research-notes`. |
| **Build index** | Click it. Your files are read, split, and stored so they can be searched. Wait for it to finish before chatting. |

**Tab 2 — "Open a saved index"** *(reuse a knowledge base you built earlier)*

- Pick a previously built knowledge base from the **dropdown** and click **Open index** — no need to re-upload anything.

**The chat area**

- Type your question and press Enter. Each answer includes a **Sources** expander — click it to see the exact passages the answer came from.
- **Clear conversation** (top-right) wipes the current chat so you can start fresh.

<br/>

## ☁️ Deploying it online

This app is built for **[Streamlit Community Cloud](https://share.streamlit.io)** (free):

1. Push this repo to GitHub.
2. On Streamlit Community Cloud, click **Create app** and point it at your repo.
3. Set **Main file path** to `RAG_app.py`.
4. Open **Advanced settings** and set the **Python version to 3.11**.
5. Click **Deploy**.

Leave the Secrets box empty — visitors enter their own API keys in the sidebar, so usage is billed to them, not you.

<br/>

## 📂 Project structure

```
RAG-Chatbot/
├── RAG_app.py            # the Streamlit app — this is what you run
├── RAG_notebook.ipynb    # a step-by-step walkthrough of how the system is built
├── requirements.txt      # Python dependencies
├── images/               # README images (architecture diagram, screenshot)
└── README.md
```

<br/>

## ⚠️ Good to know

- **Costs:** OpenAI and Cohere bill against your account per use. Google Gemini and Hugging Face have free tiers. For a free setup, use **Gemini + a non-Cohere strategy**.
- **Scanned PDFs:** if a PDF is a scanned image with no real text, there's nothing to extract — use a document with selectable text.
- **Online storage is temporary:** on Streamlit Community Cloud, indexes you build reset whenever the app restarts. That's fine for try-it-out sessions; each visitor just builds their own.

<br/>

## 🙏 Credits

Built by **Abdur Rafey**.

<br/>

## 📜 License

Released under the **MIT License** — free to use, modify, and share. See [`LICENSE`](LICENSE) for details.

<br/>

<div align="center">
<sub>If this project helped you, consider giving it a ⭐ — it helps others find it.</sub>
</div>
