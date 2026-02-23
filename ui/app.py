import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ayushbb@1234"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
client = OpenAI()

def graph_query(question):
    keyword = None

    if "storage" in question.lower():
        keyword = "storage"
    elif "compute" in question.lower():
        keyword = "compute"
    else:
        keyword = ""

    with driver.session() as session:
        result = session.run("""
            MATCH (c:CostRecord)-[:INCURRED_BY]->(r:Resource)
                  -[:USES_SERVICE]->(s:Service)
            WHERE toLower(s.serviceName) CONTAINS $keyword
            RETURN s.serviceName AS service,
                   SUM(c.billedCost) AS totalCost
            ORDER BY totalCost DESC
        """, keyword=keyword)

        cost_data = {}
        context = ""

        for record in result:
            service = record["service"]
            total = record["totalCost"]
            cost_data[service] = total
            context += f"Service: {service}, TotalCost: {total}\n"

        services = list(cost_data.keys())

        if len(services) >= 2:
            diff = cost_data[services[0]] - cost_data[services[1]]
            percentage = (diff / cost_data[services[1]]) * 100 if cost_data[services[1]] != 0 else 0

            context += f"\nCost Difference: {diff}\n"
            context += f"Percentage Difference: {percentage:.2f}%\n"

        return context, services

def generate_answer(question, context):
    prompt = f"""
You are a senior FinOps cloud cost analyst.

Use ONLY the structured context below.

Structured Cost Context:
{context}

User Question:
{question}

Provide a professional explanation.
"""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content

def rag_pipeline(question):
    context, services = graph_query(question)
    answer = generate_answer(question, context)
    return services, context, answer

st.set_page_config(page_title="Cloud Cost Knowledge Graph", layout="wide")

st.title("Cloud Cost Knowledge Graph Assistant")

question = st.text_input("Ask a Cloud Cost Question")

if st.button("Analyze") and question:
    with st.spinner("Processing..."):
        services, context, answer = rag_pipeline(question)

    st.subheader("Relevant Services")
    st.write(services)

    st.subheader("Structured Cost Context")
    st.text(context)

    st.subheader("Final Answer")
    st.write(answer)