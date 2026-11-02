"""Semiconductor Intelligence Agent.

Global Semiconductor Market & Geopolitics Intelligence Monitor.

Architecture
------------
Pure-Python ingestion layer (no LLM):
    RSS fetch -> 24h UTC cutoff -> deduplication -> HTML cleanup -> truncation

Sequential CrewAI pipeline (3 specialised agents):
    1. Lead Semiconductor Triage Analyst        (scrapes the top 2-3 articles)
    2. Semiconductor Supply Chain & Geopolitical Strategist (risk scoring)
    3. Principal Technology Equity Research Editor (executive Markdown briefing)

Output:
    semiconductor_intelligence_brief.md -> semiconductor_intelligence_brief.pdf
"""

from __future__ import annotations

import calendar
import html
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import feedparser
from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv
from markdown_pdf import MarkdownPdf, Section

load_dotenv()  # Load variables from a local .env file, if present.

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
OUTPUT_DIR: Path = Path(__file__).resolve().parent
MD_FILE: Path = OUTPUT_DIR / "semiconductor_intelligence_brief.md"
PDF_FILE: Path = OUTPUT_DIR / "semiconductor_intelligence_brief.pdf"

# Override with the GEMINI_MODEL environment variable if needed.
DEFAULT_MODEL: str = "gemini/gemini-3.8-flash"

LOOKBACK_HOURS: int = 24
MAX_SUMMARY_CHARS: int = 300  # Keep RSS summaries short for token efficiency.
MAX_ARTICLES: int = 40  # Hard cap on articles handed to the LLM workflow.

# High-signal semiconductor industry & bellwether company keywords.
KEYWORDS: tuple[str, ...] = (
    "semiconductor",
    "chip",
    "TSMC",
    "ASML",
    "Nvidia",
    "Intel",
    "Samsung",
    "Infineon",
    "lithography",
)


def build_news_query() -> str:
    """Build a robust, single-group Boolean Google News RSS query.

    Google News RSS search engine cannot parse nested parentheses or excessively
    long queries. A flat OR group across key industry terms and market bellwethers
    ensures reliable, high-yield news ingestion.
    """
    or_clause = " OR ".join(KEYWORDS)
    return f"({or_clause}) when:1d"


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def get_api_key() -> str:
    """Return the Gemini API key or exit gracefully if it is missing."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: GEMINI_API_KEY is not set.\n"
            "Add it to your .env file (GEMINI_API_KEY=...) or export it in "
            "your shell, then run the script again."
        )
    # Some libraries look for GOOGLE_API_KEY instead.
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    return api_key


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_html(text: str) -> str:
    """Strip HTML tags, unescape entities, and collapse whitespace."""
    no_tags = _TAG_RE.sub(" ", text or "")
    return _WS_RE.sub(" ", html.unescape(no_tags)).strip()


def _normalise_title(title: str) -> str:
    """Normalise a headline for duplicate detection (drops ' - Source')."""
    base = title.rsplit(" - ", 1)[0] if " - " in title else title
    return re.sub(r"[^a-z0-9]+", " ", base.lower()).strip()


@dataclass(frozen=True)
class Article:
    """A cleaned, timestamped news item."""

    title: str
    link: str
    published: datetime  # Always timezone-aware (UTC).
    summary: str

    def render(self, index: int) -> str:
        """Render the article as a structured plain-text block."""
        stamp = self.published.strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"{index}. TITLE: {self.title}\n"
            f"   LINK: {self.link}\n"
            f"   PUBLISHED: {stamp}\n"
            f"   SUMMARY: {self.summary or 'N/A'}"
        )


# --------------------------------------------------------------------------- #
# Pure-Python ingestion layer (no LLM)
# --------------------------------------------------------------------------- #
def fetch_semiconductor_news(query: str | None = None) -> str:
    """Fetch, filter, deduplicate and format semiconductor news.

    Steps: Google News RSS fetch, strict 24h UTC cutoff, deduplication by
    link and normalised headline, HTML sanitisation, summary truncation,
    newest-first ordering.

    Returns:
        A numbered plain-text list ready to be embedded in a task prompt, or
        a ``NO ARTICLES FOUND`` message.
    """
    query = query or build_news_query()
    url = (
        "https://news.google.com/rss/search?q="
        f"{quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    feed = feedparser.parse(url)
    if getattr(feed, "bozo", False) and not feed.entries:
        print(f"WARNING: RSS feed could not be parsed: {feed.get('bozo_exception')}")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    seen_links: set[str] = set()
    seen_titles: set[str] = set()
    articles: list[Article] = []

    for entry in feed.entries:
        parsed = entry.get("published_parsed")
        if not parsed:
            continue  # Strict mode: undated items cannot satisfy the cutoff.
        # feedparser yields UTC struct_time; calendar.timegm is UTC-safe
        # (unlike time.mktime, which assumes local time).
        pub_dt = datetime.fromtimestamp(calendar.timegm(parsed), timezone.utc)
        if pub_dt < cutoff:
            continue

        title = clean_html(entry.get("title", "")) or "N/A"
        link = (entry.get("link") or "").strip()
        title_key = _normalise_title(title)
        if not link or link in seen_links or title_key in seen_titles:
            continue
        seen_links.add(link)
        seen_titles.add(title_key)

        summary = clean_html(entry.get("summary", ""))
        # Google News summaries often just repeat the headline; drop those.
        if _normalise_title(summary).startswith(title_key):
            summary = ""
        articles.append(
            Article(title, link, pub_dt, summary[:MAX_SUMMARY_CHARS].strip())
        )

    articles.sort(key=lambda a: a.published, reverse=True)
    articles = articles[:MAX_ARTICLES]
    if not articles:
        return f"NO ARTICLES FOUND in the last {LOOKBACK_HOURS} hours."
    return "\n\n".join(a.render(i) for i, a in enumerate(articles, 1))


# --------------------------------------------------------------------------- #
# Crew construction
# --------------------------------------------------------------------------- #
def build_crew(raw_news: str) -> Crew:
    """Assemble the three-agent sequential crew around the ingested news."""
    llm = LLM(
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        api_key=os.environ["GEMINI_API_KEY"],
    )
    scrape_tool = ScrapeWebsiteTool()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ---- Agents ----------------------------------------------------------- #
    triage_analyst = Agent(
        role="Lead Semiconductor Triage Analyst",
        goal=(
            "Discard PR fluff and marketing noise, and isolate the 2-3 "
            "highest-impact articles on geopolitical export restrictions, "
            "production bottlenecks, capacity investments, or macro supply "
            "chain risks, then extract verifiable facts from their sources."
        ),
        backstory=(
            "A former sell-side semiconductor equity analyst with 15 years "
            "covering foundries, memory and equipment names across Taiwan, "
            "Korea and the Netherlands. You have seen enough product-launch "
            "press releases to spot noise instantly, and you trust only "
            "primary facts: figures, dates, named counterparties, regulatory "
            "text. You never speculate beyond what the source states."
        ),
        tools=[scrape_tool],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    strategist = Agent(
        role="Semiconductor Supply Chain & Geopolitical Strategist",
        goal=(
            "Analyse the systemic ripple effects of the confirmed facts on "
            "global lead times, fab utilisation, equipment suppliers and "
            "downstream tech and automotive industries, and assign a "
            "calibrated Risk Impact Score (1-10) with explicit justification."
        ),
        backstory=(
            "A tech-policy strategist who advised governments and Tier-1 "
            "OEMs through the 2021 chip shortage, US-China export-control "
            "escalations and the Taiwan Strait risk debate. You think in "
            "second- and third-order effects, distinguish transitory noise "
            "from structural shifts, and always separate fact from inference."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    editor = Agent(
        role="Principal Technology Equity Research Editor",
        goal=(
            "Synthesise all insights into a polished, executive-ready "
            "Markdown briefing for C-suite executives and portfolio managers."
        ),
        backstory=(
            "A principal editor at a top-tier equity research house who has "
            "shaped morning notes read by CIOs and CEOs. You write with "
            "precision and economy: conclusion first, evidence second, "
            "actionable implications last. You never invent data and you "
            "always cite sources with timestamps."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    # ---- Tasks ------------------------------------------------------------ #
    triage_task = Task(
        description=(
            "Below are the news items from the last 24 hours, pre-filtered "
            "and deduplicated by a Python ingestion layer.\n\n"
            "1. Discard PR, product-marketing and irrelevant items.\n"
            "2. Select ONLY the top 2-3 highest-impact articles on export "
            "restrictions, production bottlenecks, capacity investments or "
            "macro supply chain risks.\n"
            "3. Use the website scraping tool on those specific URLs only, "
            "focusing on the primary paragraphs. If a page cannot be "
            "scraped, fall back to the RSS title/summary and state this.\n"
            "4. Extract hard facts (who, what, where, numbers, dates). Never "
            "invent facts. If no articles were found, say so clearly.\n"
            "STRICT RULE: for every selected article, pass on its full URL "
            "and its PUBLISHED timestamp exactly as given.\n\n"
            "=== NEWS ITEMS ===\n" + raw_news
        ),
        expected_output=(
            "For each selected article: title, full URL, exact PUBLISHED "
            "timestamp, a one-line selection rationale, and 3-5 bullet "
            "points of factual details."
        ),
        agent=triage_analyst,
    )

    risk_task = Task(
        description=(
            "Using ONLY the facts from the triage output, analyse systemic "
            "implications for: global lead times, fab utilisation, "
            "equipment suppliers (lithography/EDA/materials), and "
            "downstream tech and automotive industries. Distinguish clearly "
            "between confirmed facts and your inference. Assign each "
            "article a Risk Impact Score from 1 (negligible) to 10 "
            "(systemic disruption) with a justification covering severity, "
            "probability and time horizon, plus one overall score."
        ),
        expected_output=(
            "Per-article ripple-effect analysis, a Risk Impact Score (1-10) "
            "with justification for each article, and an overall score."
        ),
        agent=strategist,
        context=[triage_task],
    )

    briefing_task = Task(
        description=(
            "Write the final executive briefing in Markdown using ONLY the "
            "preceding task outputs. Use exactly this structure:\n"
            f"# Semiconductor Intelligence Brief\n*Date: {today}*\n"
            "## Executive Summary (high-signal overview, max 4 sentences)\n"
            "## Key Industry Events & Hard Facts (bullets with scraped "
            "data)\n"
            "## Geopolitical & Supply Chain Impact Analysis\n"
            "## Risk Assessment & Actionable Takeaways (include a Markdown "
            "table: Event | Risk Score (1-10) | Rationale, then 3-5 "
            "actionable takeaways)\n"
            "## Sources & Publication Dates\n"
            "STRICT RULE: the final section MUST list the full URL and the "
            "exact publication timestamp of every article analysed, one per "
            "line, copied unchanged. Add nothing after it. If no articles "
            "were found, state that plainly instead of inventing content. "
            "Output ONLY the Markdown, with no code fences."
        ),
        expected_output=(
            "A polished Markdown executive briefing following the "
            "required structure."
        ),
        agent=editor,
        context=[triage_task, risk_task],
    )

    return Crew(
        agents=[triage_analyst, strategist, editor],
        tasks=[triage_task, risk_task, briefing_task],
        process=Process.sequential,
        verbose=True,
    )


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _strip_code_fence(text: str) -> str:
    """Remove a wrapping ```markdown fence the LLM may have added."""
    stripped = text.strip()
    match = re.match(r"^```(?:markdown|md)?\s*\n(.*?)\n```$", stripped, re.S)
    return match.group(1).strip() if match else stripped


def save_markdown(report: str, md_path: Path = MD_FILE) -> Path:
    """Persist the Markdown report as UTF-8, with a timestamped fallback.

    If the primary path is not writable (e.g. locked by another process on
    Windows), the report is written to a timestamped file instead so that
    the output is never lost.
    """
    try:
        md_path.write_text(report, encoding="utf-8", newline="\n")
        return md_path
    except OSError as exc:
        fallback = md_path.with_name(
            f"{md_path.stem}_{datetime.now():%Y%m%d_%H%M%S}{md_path.suffix}"
        )
        print(f"WARNING: could not write {md_path} ({exc}); using {fallback}")
        fallback.write_text(report, encoding="utf-8", newline="\n")
        return fallback


def convert_md_to_pdf(md_path: Path = MD_FILE, pdf_path: Path = PDF_FILE) -> None:
    """Convert the Markdown brief into a PDF file.

    Failures are reported but never raised, so the Markdown output survives.
    """
    if not md_path.exists():
        print(f"Cannot create PDF: {md_path} not found.")
        return
    markdown_text = md_path.read_text(encoding="utf-8")

    try:
        pdf = MarkdownPdf(toc_level=2)
        pdf.meta["title"] = "Semiconductor Intelligence Brief"
        pdf.add_section(Section(markdown_text, toc=False))
        pdf.save(str(pdf_path))
        print(f"PDF saved to {pdf_path}")
    except PermissionError:
        alt = pdf_path.with_name(
            f"{pdf_path.stem}_{datetime.now():%Y%m%d_%H%M%S}{pdf_path.suffix}"
        )
        try:
            pdf.save(str(alt))
            print(f"PDF target locked; saved to {alt}")
        except Exception as exc:  # noqa: BLE001
            print(f"PDF generation failed: {exc}. Markdown is at {md_path}.")
    except Exception as exc:  # noqa: BLE001
        print(f"PDF generation failed: {exc}. Markdown is at {md_path}.")


def extract_report(result: Any) -> str:
    """Extract the final text from a CrewAI result object."""
    return _strip_code_fence(str(getattr(result, "raw", None) or result))


def main() -> None:
    """Run ingestion, the crew pipeline, and output generation."""
    get_api_key()  # Exits gracefully if GEMINI_API_KEY is missing.

    print(f"Fetching Google News RSS (last {LOOKBACK_HOURS}h)...")
    raw_news = fetch_semiconductor_news()
    print(raw_news, "\n")

    result = build_crew(raw_news).kickoff()

    saved = save_markdown(extract_report(result))
    print(f"Markdown saved to {saved}")
    convert_md_to_pdf(saved)


if __name__ == "__main__":
    main()

