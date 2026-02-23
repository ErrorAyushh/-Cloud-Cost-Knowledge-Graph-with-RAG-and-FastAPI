from fastapi import FastAPI
from pydantic import BaseModel
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()



NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ayushbb@1234"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

client = OpenAI(api_key=OPENAI_API_KEY)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Cloud Cost Knowledge Graph API")



class QueryRequest(BaseModel):
    question: str


def detect_intent(question):
    q = question.lower()
    if "compare" in q:
        return "comparison"
    elif "top" in q:
        return "ranking"
    elif "commitment" in q:
        return "commitment_analysis"
    else:
        return "general"



def semantic_service_search(question):
    embedding = model.encode(question).tolist()

    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes(
                'service_embedding_index',
                3,
                $embedding
            )
            YIELD node, score
            RETURN node.serviceName AS service, score
        """, embedding=embedding)

        services = [record["service"] for record in result]

    return services



def graph_context(keyword):

    with driver.session() as session:

        query = """
        MATCH (c:CostRecord)-[:INCURRED_BY]->(r:Resource)
              -[:USES_SERVICE]->(s:Service)
        OPTIONAL MATCH (c)-[:HAS_CHARGE]->(ch:Charge)
        WHERE toLower(s.serviceName) CONTAINS $keyword
          AND (ch.category IS NULL OR NOT (ch.category IN ["CommitmentPurchase","CommitmentFee"]))
        RETURN s.serviceName AS service,
               r.resourceName AS resource,
               c.billedCost AS cost
        """

        result = session.run(query, keyword=keyword)

        rows = list(result)

        aggregation = {}
        for row in rows:
            service = row["service"]
            cost = row["cost"] if row["cost"] else 0

            if service not in aggregation:
                aggregation[service] = 0

            aggregation[service] += cost

        context = ""

        for service, total in aggregation.items():
            context += f"Service: {service}, TotalCost: {total}\n"

        context += "\nProvenance:\n"

        for row in rows[:15]:
            context += (
                f"Service: {row['service']}, "
                f"Resource: {row['resource']}, "
                f"Cost: {row['cost']}\n"
            )

    return context



def generate_answer(question, context):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a FinOps cloud cost expert."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )

    return response.choices[0].message.content



@app.get("/health")
def health():
    return {"status": "ok"}

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
            "confidence": 0.2
        }

    keyword = services[0].lower()

    context = graph_context(keyword)
    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "concepts": services,
        "paths": ["CostRecord -> Resource -> Service"],
        "confidence": 0.85
    }

@app.get("/concept/{name}")
def concept_details(name: str):

    with driver.session() as session:
        result = session.run("""
            MATCH (s:Service {serviceName:$name})
            OPTIONAL MATCH (s)<-[:USES_SERVICE]-(r:Resource)
            RETURN s.serviceName AS service,
                   collect(DISTINCT r.resourceName) AS resources
        """, name=name)

        record = result.single()

    if not record:
        return {"message": "Concept not found"}

    return {
        "service": record["service"],
        "related_resources": record["resources"]
    }

@app.get("/stats")
def stats():

    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

    return {
        "total_nodes": node_count,
        "total_relationships": rel_count,
        "vector_index": "service_embedding_index"
    }