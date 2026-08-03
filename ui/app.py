import os

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer

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


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.set_page_config(
    page_title="Cloud Cost Knowledge Graph",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Cloud Cost Knowledge Graph Assistant")

st.write(
    "Ask questions about cloud costs, services, and spending trends."
)

question = st.text_input(
    "Ask a Cloud Cost Question",
    placeholder="Which service has the highest cloud cost?"
)

if st.button("Analyze") and question:

    with st.spinner("Analyzing..."):

        services, context, answer = rag_pipeline(question)

    st.subheader("Relevant Services")

    if services:
        st.write(services)
    else:
        st.info("No matching services found.")

    st.subheader("Structured Cost Context")
    st.code(context)

    st.subheader("AI Analysis")
    st.write(answer)
