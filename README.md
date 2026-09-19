# Production E-Commerce Recommendation & Ranking Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD Pipeline](https://github.com/shubham-murtadak/recomendation_system/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/shubham-murtadak/recomendation_system/actions/workflows/ci_cd.yml)
[![AWS](https://img.shields.io/badge/AWS-us--east--1-FF9900.svg?logo=amazon-aws)](https://aws.amazon.com/)
[![Terraform](https://img.shields.io/badge/Terraform-IaC-844FBA.svg?logo=terraform)](https://www.terraform.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-red.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)

An end-to-end, enterprise-grade multi-stage recommendation and ranking platform trained on the **RetailRocket Implicit-Feedback Dataset** (~2.75M user interactions across 1.15M visitors and 215K items). 

Implements the industry-standard **3-Stage Production Recommender Funnel** (Retrieval → Ranking → Re-Ranking) with offline temporal evaluation, low-latency REST API with Swagger documentation, automated GitHub Actions CI/CD, and Terraform Infrastructure as Code (IaC) deployable to AWS.

---

## 🏛️ System Architecture

```text
                                USER REQUEST
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │ STAGE 1: Candidate Retrieval (~150 items)             │
         │ • Item-Item Collaborative Filtering (Cosine Matrix)   │
         │ • Popularity Fallback & Backfill                      │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │ STAGE 2: Precision Ranking (Top 30 items)             │
         │ • V1: Logistic Regression Ranker                      │
         │ • V2: XGBoost Gradient Boosted Decision Trees (GBDT)  │
         │ • V3: DeepFM (Deep Factorization Machine in PyTorch)  │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────────┐
         │ STAGE 3: Re-Ranking & Diversity Layer (Final Top 10)  │
         │ • Maximal Marginal Relevance (MMR, λ=0.7)             │
         │ • Category Capping Constraint (Max 2 items/category)  │
         │ • Freshness & De-biasing Exploration                  │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                          FINAL TOP-10 CAROUSEL
```

---

## 📊 Evolutionary Benchmark Scorecard (5,000 Test Cohort)

All models evaluated strictly on a **temporal holdout test set** to prevent data leakage:

| Model Generation | Architecture / Model | NDCG@10 | Recall@10 | Precision@10 | Hit Rate@10 | Intra-List Diversity (ILD) | Add-to-Cart Lift |
|---|---|---|---|---|---|---|---|
| **Baseline** | Weighted Popularity | `0.003897` | `0.004887` | `0.000720` | `0.006800` | `0.9974` | Base |
| **V1** | Item-CF + Logistic Regression | `0.012829` | `0.018677` | `0.003460` | `0.029200` | `0.9843` | +229% vs Base |
| **V2** | Item-CF + **XGBoost GBDT** | **`0.019941`** | **`0.026372`** | **`0.005100`** | **`0.043200`** | `0.9843` | **+55.4% vs V1** |
| **V3** | Item-CF + **DeepFM (PyTorch)** | `0.013628` | `0.019542` | `0.003880` | `0.032800` | `0.9843` | +6.2% vs V1 |
| **V4** | **Two-Stage + MMR Re-Ranking**| `0.017863` | `0.021762` | `0.004140` | `0.037800` | **`0.9918`** | **+16.7% Cart Lift** 🛒 |

### Key Technical Insights:
1. **Why GBDT beats Logistic Regression (+55% Lift)**: Tree ensembles capture non-linear thresholds on activity and recency, plus cross-feature interactions (`src_item_cf` $\times$ `item_cart_count`).
2. **Why Re-Ranking boosts Cart Conversions (+16.7%)**: Greedy rankers suffer from category monopolization. Applying **Maximal Marginal Relevance (MMR)** and **Category Capping** surfaces complementary products, driving higher intent actions.
3. **500x Vectorized Speedup**: Batch inference scores 5,000 users (327,666 candidate items) in **1.8 seconds**.

---

## 🌐 Real-Time REST API & Interactive Swagger UI

The service includes a production-grade FastAPI recommendation server with sub-50ms inference latency:

### Core Endpoints:
- `GET /docs` — Interactive Swagger UI documentation.
- `GET /health` — Service health, active models, catalog size, and uptime statistics.
- `GET /popular?limit=10` — Global top trending items for cold-start exploration.
- `GET /similar/{item_id}?limit=10` — Real-time Item-to-Item Collaborative Filtering similarity.
- `GET /recommend/{user_id}?k=10&model_version=v4` — Personalized multi-stage recommendations with MMR diversity.

```bash
# Start API locally on port 8000
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to test the live endpoints.

---

## ☁️ Cloud Deployment: AWS Infrastructure as Code (Terraform)

All cloud infrastructure is codified in the [`infrastructure/`](infrastructure/) directory using HashiCorp Terraform:

### Cloud Architecture (`us-east-1`):
- **Compute**: AWS EC2 (`t3.micro` — 100% AWS Free Tier eligible) with automated 3GB swap allocation.
- **Object Storage**: Amazon S3 bucket (`recsys-artifacts-*`) storing feature stores and pre-trained model artifacts with `force_destroy = true`.
- **Security & IAM**: EC2 Instance Profile with least-privilege S3 read policy, AWS Systems Manager (SSM) integration, and Security Group restricting ingress to HTTP port 8000.
- **Service Orchestration**: Systemd daemon (`recsys.service`) automatically bootstrapping virtual environments, syncing model artifacts from S3, and launching Uvicorn on boot.

### Deploy to AWS:
```powershell
cd infrastructure
.\terraform.exe init
.\terraform.exe apply -auto-approve
```

### Clean Teardown ($0.00 Ongoing Cost):
```powershell
.\terraform.exe destroy -auto-approve
```

---

## 🔄 CI/CD Automation (GitHub Actions)

The repository features a GitHub Actions pipeline ([`.github/workflows/ci_cd.yml`](.github/workflows/ci_cd.yml)):
1. **Continuous Integration**: Runs algorithmic and pipeline test suites on Python 3.11 via `pytest`.
2. **Infrastructure Validation**: Performs `terraform fmt`, `terraform init`, and `terraform validate`.
3. **Continuous Deployment**: Automated plan generation and 1-click apply/destroy execution directly from GitHub Actions with repository secrets.

---

## 🚀 Quick Start (Local Reproduction)

### 1. Installation
```bash
git clone https://github.com/shubham-murtadak/recomendation_system.git
cd recomendation_system
pip install -r requirements.txt
```

### 2. Run Pipeline Versions
```bash
# V1: Baseline + Item-CF + Logistic Regression
python run_model_pipeline.py

# V2: Upgrade to XGBoost GBDT Ranking
python run_v2_pipeline.py

# V3: Deep Recommendation with PyTorch DeepFM
python run_v3_pipeline.py

# V4: Full 3-Stage Recommender with MMR Diversity & Category Capping
python run_v4_pipeline.py
```

### 3. Run Automated Tests
```bash
pytest tests/
```

### 4. Launch MLflow Experiment Dashboard
```bash
mlflow ui --port 5000
```
Open **[http://localhost:5000](http://localhost:5000)** to compare parameters, ROC/Accuracy, NDCG@10, and metrics across all pipeline iterations.

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ci_cd.yml             # Automated CI/CD pipeline (Pytest + Terraform)
├── configs/
│   └── config.yaml               # Centralized configuration (weights, hyperparameters)
├── infrastructure/
│   ├── main.tf                   # EC2, S3, IAM, and Security Group definitions
│   ├── provider.tf               # AWS Provider configuration
│   ├── variables.tf              # Input variables and Free Tier defaults
│   └── outputs.tf                # Live endpoints (Swagger, Health, Recommendations)
├── src/
│   ├── api/
│   │   └── main.py               # FastAPI recommendation service with Swagger UI
│   ├── features/                 # User, item, and interaction feature extractors
│   ├── candidate_generation/     # Item-CF and Popularity candidate pooling
│   ├── ranking/                  # LR, XGBoost, and DeepFM ranking engines
│   ├── reranking/                # Stage 3 MMR diversity & category capping
│   └── evaluation/               # NDCG, Recall, ILD diversity, and business conversion
├── tests/
│   ├── test_algorithms.py        # Algorithmic tests for MMR & Item-CF
│   └── test_api.py               # API integration tests
├── run_model_pipeline.py         # V1 pipeline runner
├── run_v2_pipeline.py            # V2 (XGBoost) runner
├── run_v3_pipeline.py            # V3 (DeepFM) runner
├── run_v4_pipeline.py            # V4 (3-Stage Re-ranking) runner
├── pytest.ini                    # Pytest configuration
├── requirements.txt
└── README.md
```
