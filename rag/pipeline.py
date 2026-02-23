import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI


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


def detect_intent(question):
    q = question.lower()

    if "compare" in q:
        return "comparison"
    elif "top" in q:
        return "ranking"
    elif "commitment" in q or "utilization" in q:
        return "commitment_analysis"
    elif "cost type" in q:
        return "cost_type"
    else:
        return "general"


def extract_time_filter(question):
    q = question.lower()

    if "january" in q:
        return "2024-01-01", "2024-01-31"

    return None, None



def vendor_equivalent(service_name):
    mapping = {
        "amazon simple storage service": "Storage Accounts",
        "s3": "Azure Blob"
    }

    return mapping.get(service_name.lower(), None)



def semantic_service_search(question):

    embedding = model.encode(question).tolist()

    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes(
                'service_embedding_index',
                5,
                $embedding
            )
            YIELD node, score
            RETURN node.serviceName AS service, score
        """, embedding=embedding)

        services = [record["service"] for record in result]

    return services



def graph_analysis(keyword, intent, start_date=None, end_date=None):

    with driver.session() as session:

        query = """
        MATCH (c:CostRecord)-[:INCURRED_BY]->(r:Resource)
              -[:USES_SERVICE]->(s:Service)
        OPTIONAL MATCH (c)-[:HAS_CHARGE]->(ch:Charge)
        OPTIONAL MATCH (c)-[:IN_BILLING_PERIOD]->(bp:BillingPeriod)
        WHERE toLower(s.serviceName) CONTAINS $keyword
          AND (ch.category IS NULL OR NOT (ch.category IN ["CommitmentPurchase","CommitmentFee"]))
          AND ($start IS NULL OR bp.start >= $start)
          AND ($end IS NULL OR bp.end <= $end)
        RETURN s.serviceName AS service,
               r.resourceName AS resource,
               c.billedCost AS cost
        """

        result = session.run(
            query,
            keyword=keyword,
            start=start_date,
            end=end_date
        )

        rows = list(result)

        # Aggregation
        aggregation = {}
        for row in rows:
            service = row["service"]
            cost = row["cost"] if row["cost"] else 0

            if service not in aggregation:
                aggregation[service] = 0

            aggregation[service] += cost

        structured_context = ""

        # Ranking if needed
        sorted_services = sorted(
            aggregation.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if intent == "ranking":
            sorted_services = sorted_services[:5]

        for service, total in sorted_services:
            structured_context += f"Service: {service}, TotalCost: {total}\n"

        structured_context += "\nProvenance:\n"

        for row in rows[:20]:
            structured_context += (
                f"Service: {row['service']}, "
                f"Resource: {row['resource']}, "
                f"Cost: {row['cost']}\n"
            )

    return structured_context



def generate_answer(question, context, intent):

    system_prompt = """
You are a FinOps cloud cost expert.
Use financial logic.
Avoid assumptions.
Use only provided context.
"""

    if intent == "cost_type":
        context += "\nRecommended Cost Type: EffectiveCost\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )

    return response.choices[0].message.content



def run_pipeline(question):

    print("\nStep 1: Detecting Intent...")
    intent = detect_intent(question)
    print("Intent:", intent)

    print("\nStep 2: Semantic Service Search...")
    services = semantic_service_search(question)
    print("Relevant Services:", services)

    if not services:
        print("No relevant services found.")
        return

    keyword = services[0].lower()

    start_date, end_date = extract_time_filter(question)

    print("\nStep 3: Graph Retrieval & Aggregation...")
    context = graph_analysis(keyword, intent, start_date, end_date)
    print(context)

    print("\nStep 4: Generating LLM Answer...\n")
    answer = generate_answer(question, context, intent)

    print("Final Answer:\n")
    print(answer)



if __name__ == "__main__":
    user_question = input("Ask a Cloud Cost Question: ")
    run_pipeline(user_question)