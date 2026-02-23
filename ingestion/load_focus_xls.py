import pandas as pd
from neo4j import GraphDatabase
import math

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ayushbb@1234"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

def clean_value(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val

def ingest(file_path, vendor):
    df = pd.read_excel(file_path)
    df.columns = [col.lower() for col in df.columns]

    with driver.session() as session:
        for index, row in df.iterrows():

            service_name = clean_value(row.get("servicename"))
            resource_id = clean_value(row.get("resourceid"))
            billing_account = clean_value(row.get("billingaccountid"))

            if service_name is None or resource_id is None:
                continue

            cost_id = f"{vendor}_{index}"

            session.run("""
                MERGE (s:Service {serviceName: $serviceName})
                SET s.serviceCategory = $serviceCategory

                MERGE (r:Resource {resourceId: $resourceId})
                SET r.resourceName = $resourceName,
                    r.resourceType = $resourceType

                MERGE (r)-[:USES_SERVICE]->(s)

                MERGE (c:CostRecord {id: $costId})
                SET c.billedCost = $billedCost,
                    c.effectiveCost = $effectiveCost,
                    c.currency = $currency,
                    c.chargeCategory = $chargeCategory,
                    c.vendor = $vendor

                MERGE (c)-[:INCURRED_BY]->(r)
            """,
            serviceName=service_name,
            serviceCategory=clean_value(row.get("servicecategory")),
            resourceId=resource_id,
            resourceName=clean_value(row.get("resourcename")),
            resourceType=clean_value(row.get("resourcetype")),
            costId=cost_id,
            billedCost=clean_value(row.get("billedcost")),
            effectiveCost=clean_value(row.get("effectivecost")),
            currency=clean_value(row.get("billingcurrency")),
            chargeCategory=clean_value(row.get("chargecategory")),
            vendor=vendor
            )

    print(f"{vendor} file ingested successfully")

if __name__ == "__main__":
    ingest("data/aws_test-focus-00001.snappy_transformed.xls", "AWS")
    ingest("data/focusazure_anon_transformed.xls", "Azure")
    driver.close()