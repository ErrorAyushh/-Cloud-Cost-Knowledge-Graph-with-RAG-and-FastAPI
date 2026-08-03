from fastapi import FastAPI
from pydantic import BaseModel
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------------------------------
# Initialize Clients
# -----------------------------------------------------

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

app = FastAPI(title="Cloud Cost Knowledge Graph API")


# -----------------------------------------------------
# Request Model
# -----------------------------------------------------

class QueryRequest(BaseModel):
    question: str


# -----------------------------------------------------
# Intent Detection
# -----------------------------------------------------

def detect_intent(question: str):
    q = question.lower()

    if "compare" in q:
        return "comparison"

    elif "top" in q:
        return "ranking"

    elif "commitment" in q:
        return "commitment_analysis"

    else:
        return "general"


# -----------------------------------------------------
# Semantic Search
# -----------------------------------------------------

def semantic_service_search(question: str):

    embedding = embedding_model.encode(question).tolist()

    with driver.session() as session:

        result = session.run(
            """
            CALL db.index.vector.queryNodes(
                'service_embedding_index',
                3,
                $embedding
            )
            YIELD node, score
            RETURN node.serviceName AS service, score
            """,
            embedding=embedding,
        )

        services = [record["service"] for record in result]

    return services


# -----------------------------------------------------
# Graph Context Retrieval
# -----------------------------------------------------

def graph_context(keyword):

    with driver.session() as session:

        result = session.run(
            """
            MATCH (c:CostRecord)-[:INCURRED_BY]->(r:Resource)
                  -[:USES_SERVICE]->(s:Service)

            OPTIONAL MATCH (c)-[:HAS_CHARGE]->(ch:Charge)

            WHERE toLower(s.serviceName) CONTAINS $keyword
              AND (
                    ch.category IS NULL
                    OR NOT ch.category IN
                    ["CommitmentPurchase","CommitmentFee"]
                  )

            RETURN
                s.serviceName AS service,
                r.resourceName AS resource,
                c.billedCost AS cost
            """,
            keyword=keyword,
        )

        rows = list(result)

    aggregation = {}

    for row in rows:

        service = row["service"]
        cost = row["cost"] or 0

        aggregation[service] = aggregation.get(service, 0) + cost

    context = ""

    context += "=== Cost Summary ===\n"

    for service, total in aggregation.items():
        context += f"Service: {service}, Total Cost: {total}\n"

    context += "\n=== Supporting Records ===\n"

    for row in rows[:15]:

        context += (
            f"Service: {row['service']}, "
            f"Resource: {row['resource']}, "
            f"Cost: {row['cost']}\n"
        )

    return context


# -----------------------------------------------------
# LLM Answer Generation (Groq)
# -----------------------------------------------------

def generate_answer(question, context):

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert FinOps Cloud Cost Analyst. "
                    "Answer ONLY using the supplied context. "
                    "If information is missing, clearly state that."
                ),
            },
            {
                "role": "user",
                "content": f"""
Context:

{context}

Question:

{question}
""",
            },
        ],
    )

    return response.choices[0].message.content


# -----------------------------------------------------
# Health
# -----------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------------------------------
# Main Query Endpoint
# -----------------------------------------------------

@app.post("/query")
def query(request: QueryRequest):

    question = request.question

    intent = detect_intent(question)

    services = semantic_service_search(question)

    if not services:

        return {
            "answer": "No relevant services found.",
            "concepts": [],
            "paths": [],
            "confidence": 0.20,
        }

    keyword = services[0].lower()

    context = graph_context(keyword)

    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "intent": intent,
        "concepts": services,
        "paths": [
            "CostRecord → Resource → Service"
        ],
        "confidence": 0.85,
    }


# -----------------------------------------------------
# Concept Details
# -----------------------------------------------------

@app.get("/concept/{name}")
def concept_details(name: str):

    with driver.session() as session:

        result = session.run(
            """
            MATCH (s:Service {serviceName:$name})

            OPTIONAL MATCH
            (s)<-[:USES_SERVICE]-(r:Resource)

            RETURN
                s.serviceName AS service,
                collect(DISTINCT r.resourceName)
                AS resources
            """,
            name=name,
        )

        record = result.single()

    if not record:
        return {"message": "Concept not found"}

    return {
        "service": record["service"],
        "related_resources": record["resources"],
    }


# -----------------------------------------------------
# Graph Statistics
# -----------------------------------------------------

@app.get("/stats")
def stats():

    with driver.session() as session:

        node_count = session.run(
            "MATCH (n) RETURN count(n) AS count"
        ).single()["count"]

        rel_count = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS count"
        ).single()["count"]

    return {
        "total_nodes": node_count,
        "total_relationships": rel_count,
        "vector_index": "service_embedding_index",
    }
