# ⚡ Global Semiconductor Intelligence Agent

An autonomous multi-agent AI pipeline built with **CrewAI** and **Google Gemini** that monitors the global semiconductor industry, analyzes supply chain disruptions and geopolitical risks, and generates executive-ready intelligence briefings in both **Markdown** and **PDF** formats.

---

## 📌 How It Works (The Big Picture)

1. **Pure-Python Data Ingestion (No LLM Cost):**
   * Automatically queries Google News RSS for key industry bellwethers (TSMC, ASML, Nvidia, Intel, etc.).
   * Filters articles within a strict 24-hour UTC window.
   * Removes duplicate news and cleans raw HTML formatting.

2. **Sequential Multi-Agent Pipeline (3 AI Experts):**
   * **Lead Triage Analyst:** Filters out marketing noise, selects the top 2-3 most critical events, and uses a web scraper tool to extract hard facts (dates, figures, investments) directly from original web pages.
   * **Supply Chain & Geopolitical Strategist:** Analyzes systemic ripple effects (lead times, fab utilization, equipment) and assigns a calibrated Risk Impact Score (1-10) to each event.
   * **Principal Equity Research Editor:** Synthesizes all findings into a structured, executive-grade briefing designed for C-suite leaders and investors.

3. **Multi-Format Publishing:**
   * Saves the clean briefing as a Markdown file (`semiconductor_intelligence_brief.md`).
   * Automatically converts the report into a styled PDF document (`semiconductor_intelligence_brief.pdf`) with safe file-lock handling.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **AI Framework:** CrewAI & CrewAI Tools
* **LLM Engine:** Google Gemini (Gemini 3.8 Flash)
* **Data Sources:** Google News RSS, `feedparser`
* **Output Formats:** Markdown, `markdown-pdf`

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/cemediboglu/semiconductor-intelligence-agent.git
cd semiconductor-intelligence-agent
```

### 2. Install dependencies
```bash
pip install crewai crewai-tools feedparser python-dotenv markdown-pdf
```

### 3. Set up your Gemini API Key
Create a `.env` file in the project folder and add your Google Gemini API key:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Agent
```bash
python semiconductor_agent.py
```

---

## 📄 Output Files

Once the run completes, you will find two files in your directory:
* `semiconductor_intelligence_brief.md`: Structured Markdown report.
* `semiconductor_intelligence_brief.pdf`: Polished PDF report ready for distribution.
