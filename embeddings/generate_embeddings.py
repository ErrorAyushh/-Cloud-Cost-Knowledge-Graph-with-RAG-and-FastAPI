from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer



URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "Ayushbb@1234"  

VECTOR_INDEX_NAME = "service_embedding_index"
MODEL_NAME = "all-MiniLM-L6-v2"



print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully.")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))




def generate_and_store_embeddings():
    print("Generating embeddings for Service nodes...")

    with driver.session() as session:
        result = session.run("MATCH (s:Service) RETURN s.serviceName AS name")

        for record in result:
            service_name = record["name"]

            embedding = model.encode(service_name).tolist()

            session.run("""
                MATCH (s:Service {serviceName: $name})
                SET s.embedding = $embedding
            """, name=service_name, embedding=embedding)

            print(f"Embedding stored for: {service_name}")

    print("All embeddings stored successfully.\n")



def create_vector_index():
    print("Creating vector index (if not exists)...")

    with driver.session() as session:
        session.run(f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME}
            IF NOT EXISTS
            FOR (s:Service)
            ON (s.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)

    print("Vector index ready.\n")




def search_similar_services(query, top_k=3):
    print(f"\nSearching for services similar to: '{query}'")

    query_embedding = model.encode(query).tolist()

    with driver.session() as session:
        result = session.run(f"""
            CALL db.index.vector.queryNodes(
                '{VECTOR_INDEX_NAME}',
                $topK,
                $embedding
            )
            YIELD node, score
            RETURN node.serviceName AS service, score
            ORDER BY score DESC
        """, embedding=query_embedding, topK=top_k)

        print("\nTop Matches:")
        for record in result:
            print(f"{record['service']}  |  Score: {record['score']:.4f}")

    print()




if __name__ == "__main__":

    generate_and_store_embeddings()

    create_vector_index()

    search_similar_services("storage optimization")