Yes bro. Now we should **freeze a V1 architecture** that your AI agent can implement quickly, without trying to build Google/Netflix-level recommender infrastructure on day one.

I checked the RetailRocket dataset details again: it has ~2.76M events from ~1.4M visitors over ~4.5 months, with `view`, `addtocart`, and `transaction` events, plus ~417K items and time-varying item properties. ([Kaggle][1])

# 🚀 Project V1: Production-Style E-Commerce Recommendation System

### Final objective

Build this first:

```text
                 RETAILROCKET DATA
                        │
                        ▼
              Data Engineering Layer
                        │
                        ▼
              User/Item Feature Store
                        │
                        ▼
             ┌─────────────────────┐
             │ Candidate Generation│
             └──────────┬──────────┘
                        │
                    ~100 items
                        │
                        ▼
             ┌─────────────────────┐
             │   Ranking Model     │
             └──────────┬──────────┘
                        │
                     Top 10
                        │
                        ▼
              Recommendation API
                        │
                        ▼
                Evaluation Layer
                        │
                        ▼
              A/B Testing Layer
```

**Do NOT start with DeepFM, transformers, agents, LLMs, etc.**

First make the entire pipeline work.

Then we improve each component.

---

# 1. Dataset

Use the **RetailRocket Recommender System Dataset**.

[RetailRocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?utm_source=chatgpt.com)

It contains:

### `events.csv`

```text
timestamp
visitorid
event
itemid
transactionid
```

Events:

```text
view
addtocart
transaction
```

The official dataset describes 2,756,101 events, including 2,664,312 views, 69,332 carts and 22,457 transactions from 1,407,580 visitors. ([Kaggle][1])

### `item_properties_part1.csv`

### `item_properties_part2.csv`

Contains time-dependent product properties.

### `category_tree.csv`

Product category hierarchy.

---

# 2. First thing the AI agent should do

Don't immediately train a model.

Create:

```text
data/
├── raw/
│   ├── events.csv
│   ├── item_properties_part1.csv
│   ├── item_properties_part2.csv
│   └── category_tree.csv
│
└── processed/
```

Then profile:

```text
number of users
number of items
number of events
event distribution
events/user
events/item
unique categories
missing values
duplicate events
time range
```

And create an EDA report.

---

# 3. Data cleaning

The agent should handle:

### Timestamp

Convert:

```text
Unix milliseconds
        ↓
datetime
```

Extract:

```text
date
hour
day_of_week
week
```

### Remove impossible/invalid records

Check:

* null visitor IDs
* null item IDs
* invalid timestamps
* duplicate records
* inconsistent transaction IDs

### Important

**Do not randomly split the data.**

Recommendation systems are temporal.

We need:

```text
PAST                         FUTURE

───────────────┬────────────────────
     TRAIN     │       TEST
───────────────┴────────────────────
```

Otherwise we risk temporal leakage.

---

# 4. Create sessions

This is important.

A visitor can have:

```text
10:01 → item A
10:04 → item B
10:07 → item C
```

These belong to one browsing session.

If there is a sufficiently long inactivity gap, start a new session.

Use a configurable threshold, e.g. **30 minutes**, rather than hard-coding the assumption throughout the system.

Create:

```text
session_id
visitorid
timestamp
itemid
event
```

This gives us:

```text
User
 ↓
Session
 ↓
Sequence of items
```

This becomes extremely useful later for **session-based recommendation**.

---

# 5. Interaction strength

Because this is implicit-feedback data, we shouldn't treat:

```text
view == purchase
```

A purchase is much stronger evidence of preference.

Start with:

```text
view          = 1
addtocart     = 3
transaction   = 5
```

These are **engineering weights**, not values supplied by RetailRocket. They should be configurable and later tuned/ablation-tested.

Create:

```text
interaction_score
```

Example:

```text
user 101
item 500

view       → 1
view       → 1
addtocart  → 3

total interaction score = 5
```

The dataset itself is explicitly designed for implicit-feedback recommender research. ([Kaggle][1])

---

# 6. Build the first recommender

## Model 0 — Popularity

Before ML, build:

```text
Most viewed products
Most added-to-cart products
Most purchased products
```

Then create a weighted popularity score:

```text
popularity_score =
    1 × views
  + 3 × carts
  + 5 × transactions
```

Normalize if necessary.

This is our **baseline**.

This is extremely important.

Every sophisticated model must beat a simple baseline.

---

# 7. Candidate Generation

Now build the first actual personalized component.

For a user:

```text
User 123
   │
   ├── viewed A
   ├── viewed B
   ├── purchased C
   └── viewed D
```

Generate candidates using multiple strategies.

### Candidate source 1 — Item-item similarity

If:

```text
User viewed A
```

find products frequently interacted with by users who also interacted with A.

```text
A → B
A → D
A → F
```

### Candidate source 2 — User history

Recommend similar products to:

```text
recently viewed
recently carted
recently purchased
```

### Candidate source 3 — Popularity

Always maintain:

```text
global popular products
```

### Candidate source 4 — Category

Later:

```text
user likes category X
        ↓
products from category X
```

Then combine them.

For V1:

```text
Candidate Pool
      ↓
100–200 items
```

---

# 8. Ranking

This is where the project becomes interesting.

For every:

```text
(user, candidate_item)
```

create a feature vector.

Example:

```text
user_item_view_count
user_item_cart_count
user_item_purchase_count
item_total_views
item_total_carts
item_total_purchases
user_total_interactions
user_category_interactions
item_category
recency
time_since_last_interaction
session_position
item_popularity
```

Potential feature table:

```text
user_id
item_id

user_view_count
user_cart_count
user_purchase_count

item_view_count
item_cart_count
item_purchase_count

user_category_count
item_category

days_since_user_last_interaction
days_since_item_last_interaction

candidate_source
```

---

# 9. V1 Ranking Model

Don't start with DeepFM.

Start:

### Logistic Regression

Target:

```text
Did user interact with this candidate in the future?
```

For example:

```text
X = user/item/context features

y =
1 → user interacted
0 → user didn't
```

Output:

```text
P(interaction | user,item,context)
```

Then rank:

```text
candidate
   ↓
probability
   ↓
sort descending
   ↓
Top 10
```

This gives us:

```text
Candidate Generation
        ↓
Logistic Ranking
        ↓
Top-K
```

---

# 10. Evaluation

We need **two different types of evaluation**.

## ML recommendation metrics

### Recall@K

Of the items the user actually interacted with:

> how many did we retrieve?

### Precision@K

Of the items we recommended:

> how many were relevant?

### NDCG@K

Rewards putting relevant items higher in the ranking.

Also track:

```text
MAP@K
MRR@K
HitRate@K
Coverage
```

Don't blindly optimize everything.

For V1:

> **Primary offline metric = NDCG@10**

> **Secondary = Recall@10**

---

# 11. Business metrics

This is where your Statistics/A-B testing preparation comes into the project.

Track:

```text
View Rate
Add-to-Cart Rate
Purchase Rate
```

Define them clearly.

For example:

```text
Purchase Rate =
users purchasing
----------------
users exposed
```

Later we can add:

```text
Revenue/User
Average Order Value
```

But **don't invent revenue in V1** unless we explicitly create a documented synthetic assumption, because RetailRocket doesn't provide actual product prices/revenue in the event log.

---

# 12. Recommendation API

Once the model works:

```text
GET /recommend/{user_id}
```

Response:

```json
{
  "user_id": 123,
  "recommendations": [
    {"item_id": 5001, "score": 0.91},
    {"item_id": 8321, "score": 0.87},
    {"item_id": 1022, "score": 0.83}
  ]
}
```

Use:

```text
FastAPI
```

Later:

```text
Docker
```

---

---

# 15. Architecture your AI agent should implement

Tell your agent to create this structure:

```text
recommendation-system/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_candidate_generation.ipynb
│   ├── 05_ranking.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── cleaning.py
│   │   └── sessionization.py
│   │
│   ├── features/
│   │   ├── user_features.py
│   │   ├── item_features.py
│   │   └── interaction_features.py
│   │
│   ├── candidate_generation/
│   │   ├── popularity.py
│   │   ├── item_cf.py
│   │   └── candidate_service.py
│   │
│   ├── ranking/
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   ├── ranking_metrics.py
│   │   └── business_metrics.py
│   │
│   ├── experimentation/
│   │   ├── assignment.py
│   │   ├── metrics.py
│   │   └── statistical_tests.py
│   │
│   └── api/
│       └── main.py
│
├── models/
│
├── tests/
│
├── configs/
│
├── requirements.txt
├── Dockerfile
├── README.md
└── Makefile
```

---

# 16. AI Agent workflow

Give your coding agent **phases**, not one giant prompt.

### Agent 1 — Data Engineer

```text
Download/load RetailRocket
→ validate
→ clean
→ sessionize
→ temporal split
→ save Parquet
```

### Agent 2 — Recommendation Engineer

```text
Popularity baseline
→ item-item CF
→ candidate generation
→ evaluate Recall@K
```

### Agent 3 — ML Engineer

```text
Build ranking dataset
→ negative sampling
→ feature engineering
→ Logistic Regression
→ evaluate NDCG@K
```

### Agent 4 — Backend Engineer

```text
FastAPI
→ load models
→ /recommend endpoint
→ recommendation response
```

### Agent 5 — Experimentation Engineer

```text
Control/Treatment assignment
→ metric calculation
→ statistical testing
→ experiment report
```

### Agent 6 — QA/Reviewer

Check:

```text
data leakage
temporal leakage
duplicate handling
cold-start
unseen users
unseen items
metric correctness
API correctness
reproducibility
```

---

# 17. Then V2 → V5

This is where we make the project genuinely strong.

### V1 — Working system

```text
Popularity
+
Item-CF
+
Logistic ranking
+
FastAPI
+
Offline evaluation
```

### V2 — Better ranking

Replace Logistic Regression:

```text
Logistic Regression
       ↓
XGBoost / LightGBM
```

Then compare:

```text
LR vs GBDT
```

This gives you a great interview discussion:

> Why did gradient boosting outperform linear ranking?

---

### V3 — Deep Recommendation

Implement:

**DeepFM**

```text
User features
Item features
Context features
       ↓
Embedding
       ↓
FM interactions
       +
Deep network
       ↓
Prediction
```

Now you have a proper deep ranking model.

---

### V4 — Two-stage recommender

Make the architecture more realistic:

```text
             User
              │
              ▼
       Candidate Retrieval
              │
          100–500
              │
              ▼
           Ranking
              │
            20
              │
              ▼
          Re-ranking
              │
             10
```

---

### V5 — Production/Research layer

Then add:

```text
Cold Start
Session-based recommendation
Recency weighting
Diversity
Exploration vs exploitation
CUPED
Sequential testing
Guardrail metrics
Feature store
Model monitoring
Drift detection
```

The original dataset itself specifically calls out abnormal traffic as a concern for both recommender quality and split-test bias, so anomaly/traffic-quality analysis is actually a nice project extension rather than artificial complexity. ([Kaggle][1])

---

# 🎯 The most important thing

**Do not tell your AI agent:**

> "Build a sophisticated production recommendation system."

That will cause it to generate 50 files, random algorithms and probably a half-working project.

Tell it:

```text
V1 objective:

Build a complete working two-stage e-commerce
recommendation system using the RetailRocket dataset.

Pipeline:

Raw data
→ cleaning
→ sessionization
→ temporal train/test split
→ popularity baseline
→ item-item candidate generation
→ candidate feature engineering
→ Logistic Regression ranking
→ Top-K recommendations
→ Recall@K / Precision@K / NDCG@K
→ FastAPI recommendation endpoint
→ basic offline experiment comparing
  popularity vs personalized recommendations.

Prioritize correctness, reproducibility,
data-leakage prevention and modular architecture.

Do NOT implement DeepFM, transformers,
LLMs, agents, complex MLOps or advanced A/B
testing yet.

Leave clean interfaces so these can be added later.
```

**That's the V1 I would lock.**

Then, once your agent finishes it, **we don't throw it away**. We progressively upgrade the same system — ranking → DeepFM → re-ranking → proper experimentation → monitoring — so by the end you have one serious end-to-end project rather than five disconnected projects.

[1]: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?utm_source=chatgpt.com "Retailrocket recommender system dataset"
