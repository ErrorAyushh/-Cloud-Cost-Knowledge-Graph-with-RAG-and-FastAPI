#  Cloud Cost Knowledge Graph with Hybrid RAG & FastAPI

> Ontology-Driven FinOps Intelligence System  
> Built with Python 3.11, Neo4j, Vector Embeddings & LLM-powered Hybrid RAG

---

# 1️⃣ System Architecture

## High-Level Architecture


AWS & Azure FOCUS XLS Files
↓
Data Ingestion Layer (Pandas + Validation)
↓
Neo4j Knowledge Graph (Ontology Modeled)
↓
Vector Embedding Layer (Sentence Transformers)
↓
Hybrid Retrieval (Vector + Graph Traversal)
↓
LLM Grounded Response (OpenAI)
↓
FastAPI REST Interface


## Runtime Query Flow


User Question
↓
Intent Detection
↓
Vector Search (Service Embeddings)
↓
Graph Traversal (Structured Cost Aggregation)
↓
Context Assembly + Provenance
↓
LLM Response Generation
↓
API Output


This architecture separates:

- Data modeling
- Storage
- Retrieval
- Reasoning
- Presentation

Ensuring modular scalability.

---

# 2️⃣ Ontology Design Rationale

The ontology is based on **FOCUS 1.0 FinOps specification** to ensure semantic standardization across AWS and Azure billing exports.

## Core Classes

- CostRecord
- BillingAccount
- SubAccount
- Service
- Resource
- Charge
- Location
- VendorSpecificAttributes (AWS / Azure)
- CostAllocation

### Why Ontology?

Cloud billing data is:
- Vendor-specific
- Semi-structured
- Difficult to compare across providers

Ontology ensures:

- Unified schema across AWS & Azure
- Explicit financial semantics
- Traceable cost lineage
- Accurate commitment calculations

---

## Derived Financial Logic

Cost analysis relies on proper cost selection:


EffectiveCost = BilledCost + AmortizedCost


Commitment utilization excludes purchase & fee charges to prevent double counting.

---

# 3️⃣ Graph Schema Design

## Node Types

| Node | Purpose |
|------|---------|
| CostRecord | Atomic billing entry |
| Service | Cloud service (EC2, S3, Azure Blob) |
| Resource | Specific deployed resource |
| Charge | Usage, CommitmentPurchase, Fee |
| BillingAccount | Top-level account |
| Location | Region mapping |
| VendorSpecificAttributes | x_* vendor fields |

---

## Core Relationships

- (CostRecord)-[:INCURRED_BY]->(Resource)
- (Resource)-[:USES_SERVICE]->(Service)
- (CostRecord)-[:HAS_CHARGE]->(Charge)
- (CostRecord)-[:IN_BILLING_PERIOD]->(BillingPeriod)
- (CostRecord)-[:BELONGS_TO_BILLING_ACCOUNT]->(BillingAccount)
- (CostRecord)-[:HAS_VENDOR_ATTRS]->(VendorSpecificAttributes)
- (CostRecord)-[:ALLOCATED_VIA]->(CostAllocation)

---

## Indexing Strategy

- Uniqueness constraints (billingAccountId, resourceId)
- Standard indexes (serviceName)
- Vector index (384-dimension cosine similarity)

---

# 4️⃣ Vector Store Design

Embedding Model:
- sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional vectors

Stored:
- Directly inside Neo4j nodes as `embedding` property

Vector Index:
- Cosine similarity
- Used for semantic retrieval of services and financial concepts

Purpose:
- Map natural language queries to graph concepts
- Enable semantic service discovery

---

# 5️⃣ RAG Pipeline Architecture

This system implements **Hybrid RAG**.

## Step 1: Semantic Retrieval

Vector search finds relevant services based on query meaning.

Example:
"Optimize storage cost"
→ Matches S3, Azure Blob, Storage Accounts

---

## Step 2: Graph Traversal

Cypher query retrieves structured financial data:

- Cost aggregation
- Resource linkage
- Charge category filtering
- Commitment exclusion logic

Example aggregation:


TotalCost = SUM(billedCost)


---

## Step 3: Context Assembly

System assembles:

- Aggregated cost totals
- Resource-level provenance
- Financial logic applied
- Relationship paths

---

## Step 4: LLM Generation

LLM receives structured context only.

No raw CSV.
No hallucinated data.
Grounded answer only.

---

# 6️⃣ Relationship Types & Mapping Methodology

## Mapping Strategy

AWS and Azure use vendor-specific columns:

- AWS: x_ServiceCode, x_UsageType
- Azure: x_skumetercategory, x_skudescription

These are normalized via:

- ServiceCategory
- PricingCategory
- CommitmentDiscountType

Cross-cloud equivalence is determined via:
- Service semantic similarity
- Functional category mapping

---

## Allocation Logic

CostAllocation nodes define:

- allocationMethod
- allocationTargetType
- allocationBasis
- isSharedCost

Supports:
- Proportional allocation
- Even split
- Weighted distribution

---

# 7️⃣ Data Ingestion

Datasets Used:

- AWS FOCUS billing export (~900+ records)
- Azure FOCUS billing export (~1000+ records)

Process:

1. Null-safe parsing
2. Column normalization
3. Node creation
4. Relationship linking
5. Vendor attribute classification

Total nodes created: ~1900+ CostRecords

---

# 8️⃣ Setup & Installation

## Clone Repository


git clone https://github.com/yourusername/cloud-cost-kg.git

cd cloud-cost-kg


## Create Virtual Environment


python -m venv venv
venv\Scripts\activate


## Install Dependencies


pip install -r requirements.txt


## Configure Environment Variables

Create `.env` file:


OPENAI_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password




---

## Run Application


uvicorn api.app:app --reload


Swagger UI:


http://127.0.0.1:8000/docs


---

# 9️⃣ Example Supported Queries

- Compare storage costs between AWS and Azure
- Find top 5 most expensive services
- Why does total increase when including commitment purchases?
- Which cost type should be used for spend analysis?
- What is Azure equivalent of AWS S3?
- Calculate commitment utilization correctly

---

---

# 🏁 Conclusion

This project demonstrates:

- Ontology engineering
- Knowledge graph modeling
- Financial cost reasoning
- Hybrid RAG architecture
- LLM grounding
- Production-style API design

It represents a portfolio-grade AI Engineering system aligned with FinOps standards.

---

## Author

Ayush Kumar  
B.Tech AIML  
AI Engineer | Knowledge Graph | FinOps Intelligence
