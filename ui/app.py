"""
Cloud Cost Knowledge Graph Assistant
=====================================
A professional SaaS-style Streamlit dashboard for querying cloud cost data
through a Neo4j knowledge graph + RAG pipeline (Groq LLM).

NOTE ON SCOPE:
--------------
Everything under the "BACKEND LOGIC" section below is copied EXACTLY from
the original application. Nothing in that section (env loading, Neo4j
connection, embedding model, Groq client, graph_query, generate_answer,
rag_pipeline) has been modified. Only the Streamlit UI layer has been
redesigned.
"""

import os
import re
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# =========================================================================
# BACKEND LOGIC — UNCHANGED (do not modify)
# =========================================================================

# -------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------------------------------------
# Models
# -------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# -------------------------------------------------------
# Neo4j Connection
# -------------------------------------------------------

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# -------------------------------------------------------
# Embedding Model
# -------------------------------------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# -------------------------------------------------------
# Groq Client
# -------------------------------------------------------

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# -------------------------------------------------------
# Graph Query
# -------------------------------------------------------

def graph_query(question):

    keyword = ""

    q = question.lower()

    if "storage" in q:
        keyword = "storage"

    elif "compute" in q:
        keyword = "compute"

    with driver.session() as session:

        result = session.run(
            """
            MATCH (c:CostRecord)-[:INCURRED_BY]->(r:Resource)
                  -[:USES_SERVICE]->(s:Service)

            WHERE toLower(s.serviceName) CONTAINS $keyword

            RETURN
                s.serviceName AS service,
                SUM(c.billedCost) AS totalCost

            ORDER BY totalCost DESC
            """,
            keyword=keyword,
        )

        cost_data = {}
        context = ""

        for record in result:

            service = record["service"]
            total = record["totalCost"] or 0

            cost_data[service] = total

            context += (
                f"Service: {service}, "
                f"Total Cost: ${total:.2f}\n"
            )

        services = list(cost_data.keys())

        if len(services) >= 2:

            first = cost_data[services[0]]
            second = cost_data[services[1]]

            diff = first - second

            percentage = (
                (diff / second) * 100
                if second != 0
                else 0
            )

            context += "\nComparison\n"
            context += f"Difference: ${diff:.2f}\n"
            context += f"Percentage Difference: {percentage:.2f}%\n"

    return context, services


# -------------------------------------------------------
# LLM Answer
# -------------------------------------------------------

def generate_answer(question, context):

    prompt = f"""
You are an expert FinOps Cloud Cost Analyst.

Answer ONLY from the provided context.

If the context is insufficient, clearly say so.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are a senior FinOps cloud cost expert."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content


# -------------------------------------------------------
# RAG Pipeline
# -------------------------------------------------------

def rag_pipeline(question):

    context, services = graph_query(question)

    answer = generate_answer(question, context)

    return services, context, answer


# =========================================================================
# FRONTEND / UI LOGIC — REDESIGNED (SaaS-style dashboard)
# =========================================================================

st.set_page_config(
    page_title="Cloud Cost Knowledge Graph Assistant",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Injects the dashboard's visual theme (blue / white / light gray)."""
    st.markdown(
        """
        <style>
        /* ---------- Global ---------- */
        .main {
            background-color: #F8FAFC;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* ---------- Header ---------- */
        .app-header-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #1E293B;
            margin-bottom: 0;
        }
        .app-header-subtitle {
            font-size: 0.95rem;
            color: #64748B;
            margin-top: 0.15rem;
        }

        /* ---------- Hero ---------- */
        .hero-container {
            background: linear-gradient(135deg, #2563EB 0%, #3B82F6 60%, #60A5FA 100%);
            border-radius: 18px;
            padding: 2.4rem 2.5rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.18);
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            color: #E0EAFF;
            margin-bottom: 0;
        }

        /* ---------- Chips / Badges ---------- */
        .badge {
            display: inline-block;
            padding: 6px 16px;
            margin: 4px 6px 4px 0;
            border-radius: 999px;
            background-color: #EEF2FF;
            color: #3B5BDB;
            font-weight: 600;
            font-size: 0.85rem;
            border: 1px solid #C7D2FE;
        }

        /* ---------- Section titles ---------- */
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1E293B;
            margin-top: 0.4rem;
            margin-bottom: 0.6rem;
        }

        /* ---------- AI Analysis highlighted box ---------- */
        .ai-analysis-box {
            background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFF 100%);
            border-left: 5px solid #3B82F6;
            border-radius: 14px;
            padding: 1.6rem 1.8rem;
            color: #1E293B;
            font-size: 1.0rem;
            line-height: 1.6;
        }

        /* ---------- Sidebar status pill ---------- */
        .status-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.8rem;
        }
        .status-connected {
            background-color: #DCFCE7;
            color: #15803D;
        }
        .status-disconnected {
            background-color: #FEE2E2;
            color: #B91C1C;
        }

        /* ---------- Top-service medal cards ---------- */
        .medal-rank {
            font-size: 1.6rem;
        }
        .medal-service {
            font-weight: 700;
            color: #1E293B;
            font-size: 1.0rem;
        }
        .medal-cost {
            font-weight: 800;
            color: #2563EB;
            font-size: 1.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------------

def init_session_state() -> None:
    """Initializes state used for query statistics and results persistence."""
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "query_history" not in st.session_state:
        st.session_state.query_history = []  # list of dicts: {question, time}
    if "last_result" not in st.session_state:
        st.session_state.last_result = None  # dict: services/context/answer/question


# -------------------------------------------------------------------------
# Backend status helpers
# -------------------------------------------------------------------------

def check_db_status() -> bool:
    """Pings Neo4j to check connectivity. Does not alter backend logic."""
    try:
        driver.verify_connectivity()
        return True
    except Exception:
        return False


# -------------------------------------------------------------------------
# Context parsing (frontend-only — turns the raw context string into
# structured data for tables/charts, without touching graph_query itself)
# -------------------------------------------------------------------------

def parse_context(context: str):
    """
    Parses the raw context string produced by graph_query() into:
      - a pandas DataFrame with columns [Service, Total Cost]
      - a comparison dict (or None) with keys: difference, percentage
    """
    service_rows = re.findall(
        r"Service:\s*(.+?),\s*Total Cost:\s*\$?(-?[\d.]+)", context
    )

    df = pd.DataFrame(service_rows, columns=["Service", "Total Cost"])
    if not df.empty:
        df["Total Cost"] = df["Total Cost"].astype(float)
        df = df.sort_values("Total Cost", ascending=False).reset_index(drop=True)

    comparison = None
    diff_match = re.search(r"Difference:\s*\$?(-?[\d.]+)", context)
    pct_match = re.search(r"Percentage Difference:\s*(-?[\d.]+)%", context)
    if diff_match and pct_match:
        comparison = {
            "difference": float(diff_match.group(1)),
            "percentage": float(pct_match.group(1)),
        }

    return df, comparison


# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------

def render_sidebar() -> None:
    """Renders logo, navigation, system status, and query stats."""
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.2rem;">
                <span style="font-size:2rem;">☁️</span>
                <span style="font-size:1.15rem;font-weight:800;color:#1E293B;">
                    Cost Graph Assistant
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("AI-powered FinOps analytics")
        st.divider()

        # Navigation (single page today, structured for future growth)
        st.markdown("**Navigation**")
        st.radio(
            "Navigation",
            options=["📊 Dashboard"],
            label_visibility="collapsed",
        )
        st.divider()

        # Database status
        st.markdown("**Database Status**")
        db_ok = check_db_status()
        status_class = "status-connected" if db_ok else "status-disconnected"
        status_label = "🟢 Connected" if db_ok else "🔴 Disconnected"
        st.markdown(
            f'<span class="status-pill {status_class}">{status_label}</span>',
            unsafe_allow_html=True,
        )
        st.caption("Neo4j Knowledge Graph")
        st.divider()

        # LLM + embedding info
        st.markdown("**LLM Provider**")
        st.write(f"🧠 Groq — `{LLM_MODEL}`")
        st.markdown("**Embedding Model**")
        st.write(f"🔎 `{EMBEDDING_MODEL}`")
        st.divider()

        # Query statistics
        st.markdown("**Query Statistics**")
        col_a, col_b = st.columns(2)
        col_a.metric("Total Queries", st.session_state.query_count)
        last_q_time = (
            st.session_state.query_history[-1]["time"]
            if st.session_state.query_history
            else "—"
        )
        col_b.metric("Last Query", last_q_time)

        if st.session_state.query_history:
            with st.expander("Recent Questions"):
                for entry in reversed(st.session_state.query_history[-5:]):
                    st.caption(f"🕒 {entry['time']} — {entry['question']}")

        st.divider()

        # About
        with st.expander("ℹ️ About"):
            st.write(
                "This assistant answers questions about your cloud "
                "infrastructure spend by combining a **Neo4j knowledge "
                "graph** of cost records with **semantic search** and "
                "a **Groq-hosted LLM** for natural-language analysis."
            )
            st.caption("Built with Streamlit · Neo4j · Sentence-Transformers · Groq")


# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        """
        <div class="app-header-title">☁️ Cloud Cost Knowledge Graph Assistant</div>
        <div class="app-header-subtitle">
            AI-powered cloud cost analytics using Neo4j Knowledge Graph
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


# -------------------------------------------------------------------------
# Hero section (title + search box + Analyze button)
# -------------------------------------------------------------------------

def render_hero():
    """Renders the hero section and returns (question, submitted)."""
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Cloud Cost Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Ask questions about your cloud '
        'infrastructure costs using AI and a knowledge graph.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form(key="query_form", clear_on_submit=False):
        question = st.text_input(
            "Ask a question",
            placeholder=(
                "Examples: Which services cost the most? · "
                "Compare EC2 and Azure VM costs · "
                "Show my storage spending · "
                "What are my highest Azure costs?"
            ),
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return question, submitted


# -------------------------------------------------------------------------
# Relevant Services (chips)
# -------------------------------------------------------------------------

def render_service_badges(services) -> None:
    st.markdown('<div class="section-title">🏷️ Relevant Services</div>', unsafe_allow_html=True)
    if services:
        chips_html = "".join(f'<span class="badge">{s}</span>' for s in services)
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.info("No matching services found for this question.")


# -------------------------------------------------------------------------
# Cost Summary table
# -------------------------------------------------------------------------

def render_cost_summary_table(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">💵 Cost Summary</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("No cost records available for this query.")
        return

    display_df = df.copy()
    display_df["Total Cost"] = display_df["Total Cost"].map(lambda v: f"${v:,.2f}")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Service": st.column_config.TextColumn("Service"),
            "Total Cost": st.column_config.TextColumn("Total Cost"),
        },
    )
    st.caption("Tip: click a column header to sort.")


# -------------------------------------------------------------------------
# Top 5 services (medal cards)
# -------------------------------------------------------------------------

def render_top_services(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">🏆 Top 5 Services</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("No services to rank yet.")
        return

    top5 = df.head(5).reset_index(drop=True)
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    cols = st.columns(len(top5))

    for i, row in top5.iterrows():
        with cols[i]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="medal-rank">{medals[i]}</div>'
                    f'<div class="medal-service">{row["Service"]}</div>'
                    f'<div class="medal-cost">${row["Total Cost"]:,.2f}</div>',
                    unsafe_allow_html=True,
                )


# -------------------------------------------------------------------------
# Charts
# -------------------------------------------------------------------------

def render_charts(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">📈 Charts</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("Charts will appear once cost data is available.")
        return

    tab_bar, tab_pie, tab_trend = st.tabs(
        ["📊 Top 10 by Cost", "🥧 Cost Distribution", "📉 Cost Trend"]
    )

    blue_scale = px.colors.sequential.Blues[::-1]

    with tab_bar:
        top10 = df.head(10).sort_values("Total Cost", ascending=True)
        fig_bar = px.bar(
            top10,
            x="Total Cost",
            y="Service",
            orientation="h",
            text="Total Cost",
            color="Total Cost",
            color_continuous_scale="Blues",
        )
        fig_bar.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
        fig_bar.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Total Cost ($)",
            yaxis_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_pie:
        fig_pie = px.pie(
            df,
            names="Service",
            values="Total Cost",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab_trend:
        st.info(
            "📌 Historical trend data isn't tracked by the knowledge graph yet. "
            "Showing an illustrative placeholder — connect time-series cost "
            "records to enable real trend analysis."
        )
        placeholder_x = ["Week 1", "Week 2", "Week 3", "Week 4"]
        placeholder_y = [
            df["Total Cost"].sum() * f
            for f in (0.7, 0.85, 0.95, 1.0)
        ]
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=placeholder_x,
                y=placeholder_y,
                mode="lines+markers",
                line=dict(color="#3B82F6", width=3, dash="dot"),
                marker=dict(size=9, color="#2563EB"),
                name="Estimated spend",
            )
        )
        fig_trend.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Cost ($)",
        )
        st.plotly_chart(fig_trend, use_container_width=True)


# -------------------------------------------------------------------------
# AI Analysis
# -------------------------------------------------------------------------

def render_ai_analysis(answer: str) -> None:
    st.markdown('<div class="section-title">🤖 AI Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ai-analysis-box">💡 {answer}</div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------------
# Developer debug section
# -------------------------------------------------------------------------

def render_debug_section(services, context: str, comparison) -> None:
    with st.expander("🛠️ Developer Debug", expanded=False):
        st.markdown("**Relevant Services (raw)**")
        st.write(services)

        st.markdown("**Structured Context (raw)**")
        st.code(context or "(empty)", language="text")

        st.markdown("**Comparison Data**")
        if comparison:
            st.json(comparison)
        else:
            st.caption("No comparison data available for this query.")


# -------------------------------------------------------------------------
# Results orchestration
# -------------------------------------------------------------------------

def render_results(question: str, services, context: str, answer: str) -> None:
    df, comparison = parse_context(context)

    st.divider()
    st.markdown(f"##### Results for: *“{question}”*")

    render_service_badges(services)
    st.write("")

    col_left, col_right = st.columns([1.3, 1])
    with col_left:
        render_cost_summary_table(df)
    with col_right:
        render_top_services(df)

    st.write("")
    render_charts(df)

    st.write("")
    render_ai_analysis(answer)

    st.write("")
    render_debug_section(services, context, comparison)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    inject_custom_css()
    init_session_state()
    render_sidebar()
    render_header()

    question, submitted = render_hero()

    if submitted and question:
        with st.status("Running analysis...", expanded=True) as status:
            status.write("🔎 Searching the knowledge graph...")
            context, services = graph_query(question)

            status.write("📊 Structuring cost context...")
            time.sleep(0.15)  # brief pause purely for a smoother UX

            status.write("🤖 Generating AI analysis with Groq...")
            answer = generate_answer(question, context)

            status.update(label="✅ Analysis complete", state="complete")

        # Update session stats
        st.session_state.query_count += 1
        st.session_state.query_history.append(
            {"question": question, "time": datetime.now().strftime("%H:%M:%S")}
        )
        st.session_state.last_result = {
            "question": question,
            "services": services,
            "context": context,
            "answer": answer,
        }

    elif submitted and not question:
        st.warning("Please enter a question before clicking Analyze.")

    # Persist and re-render the last result across reruns (e.g. sidebar interactions)
    if st.session_state.last_result:
        r = st.session_state.last_result
        render_results(r["question"], r["services"], r["context"], r["answer"])


if __name__ == "__main__":
    main()
