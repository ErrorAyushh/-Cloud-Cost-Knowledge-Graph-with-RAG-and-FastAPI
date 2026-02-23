#  Cloud Cost Knowledge Graph with Hybrid RAG & FastAPI

> Ontology-Driven Cloud FinOps Intelligence System  
> Built with Python 3.11, Neo4j, Vector Embeddings, and LLM-powered RAG

---

##  Overview

This project implements a **Cloud Cost Knowledge Base** using:

- ✅ FOCUS 1.0 FinOps specification  
- ✅ Ontology-driven Knowledge Graph (Neo4j)  
- ✅ Vector embeddings for semantic retrieval  
- ✅ Hybrid RAG (Graph + Vector Search)  
- ✅ FastAPI REST layer  
- ✅ OpenAI grounded financial explanations  

The system ingests AWS and Azure FOCUS billing datasets (1900+ records), models them semantically, and enables intelligent cost analysis queries.

---


---

##  Core Components

### 1️⃣ Ontology Layer (FOCUS 1.0 Based)

Implemented classes:

- CostRecord  
- BillingAccount  
- SubAccount  
- Service  
- Resource  
- Charge  
- Location  
- VendorSpecificAttributes (AWS / Azure)  
- CostAllocation  

Includes:

- Cardinality constraints  
- Validation rules  
- Derived fields  
- Commitment exclusion logic  
- Cost type reasoning  

---

### 2️⃣ Knowledge Graph (Neo4j)

Key Relationships:

- `(CostRecord)-[:INCURRED_BY]->(Resource)`
- `(Resource)-[:USES_SERVICE]->(Service)`
- `(CostRecord)-[:HAS_CHARGE]->(Charge)`
- `(CostRecord)-[:IN_BILLING_PERIOD]->(BillingPeriod)`
- `(CostRecord)-[:BELONGS_TO_BILLING_ACCOUNT]->(BillingAccount)`
- `(CostRecord)-[:HAS_VENDOR_ATTRS]->(VendorSpecificAttributes)`

Indexes:

- Uniqueness constraints
- Service index
- Vector index (384 dimensions, cosine similarity)

---

### 3️⃣ Hybrid RAG Pipeline

The system combines:

-  Vector similarity search  
-  Structured graph traversal  
-  Cost aggregation logic  
-  Provenance tracking  
-  LLM-based grounded explanation  

Financial reasoning example:
TotalSpend = UsageCost + CommitmentPurchaseCost


To prevent double counting, commitment purchases and fees are excluded during utilization calculations.

---

##  Data Ingestion

- AWS FOCUS dataset (~900+ records)
- Azure FOCUS dataset (~1000+ records)
- Null-safe ingestion
- Vendor attribute normalization
- 1900+ CostRecords modeled in graph

---

## 🔌 REST API (FastAPI)




### Endpoints

| Endpoint | Method | Description |
|-----------|--------|------------|
| `/health` | GET | Health check |
| `/query` | POST | Ask cloud cost question |
| `/concept/{name}` | GET | Inspect service node |
| `/stats` | GET | Graph statistics |

### Swagger URL
http://127.0.0.1:8000/docs
---

##  Example Queries

- Compare storage costs between AWS and Azure  
- Find top 5 most expensive services  
- Why does total increase when including commitment purchases?  
- Which cost type should be used to analyze spend?  
- What is Azure equivalent of AWS S3?  
- Calculate commitment utilization correctly  

---

##  Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/cloud-cost-kg.git
cd cloud-cost-kg

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt



