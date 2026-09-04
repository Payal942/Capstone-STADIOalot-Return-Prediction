# Capstone Project: Predictive Demand Forecasting for STADIOalot
A Data Science capstone project focused on developing predictive demand forecasting models for STADIOalot to identify demand patterns, improve forecasting accuracy and support data-driven inventory and operational decisions.

**Student Name:** Payal.S. Morar  
**Student Number:** 23301266  
**Institution:** School of Information Technology, STADIO Higher Education  
**Due Date:** 7 September 2026  

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation for the Study](#motivation-for-the-study)
3. [Problem Statement](#problem-statement)
4. [Data Request](#data-request)
5. [Repository Structure](#repository-structure)
6. [RAAIDD Log](#raaidd-log)
   - [Risks](#risks)
   - [Actions](#actions)
   - [Assumptions](#assumptions)
   - [Issues](#issues)
   - [Decisions](#decisions)
   - [Dependencies](#dependencies)
7. [Project Scope](#project-scope)

## Project Overview

This Capstone project focuses on STADIOalot and investigates how Data Science can be used to support the company's objective of reducing avoidable product returns. STADIOalot operates at significant scale, with approximately 4.1 million active customers and 58 million orders per year. However, the company faces profitability and operational challenges, with revenue of approximately R38 billion and an operating margin of only 1.9%. The project therefore focuses on using historical customer, order, product, seller and operational data to identify patterns associated with product returns and develop a predictive approach that can identify orders that are more likely to be returned.

---

## Motivation for the Study

STADIOalot is a large South African e-commerce business with approximately 4.1 million active customers and 58 million orders annually. Despite generating approximately R38 billion in revenue, the organisation has an operating margin of only 1.9%, making operational efficiency critical to profitability. A key challenge is the increasing rate of product returns, which has risen from 11% to 15% overall and from 22% to 29% for apparel and shoes.


The increasing return rate creates additional fulfilment, logistics, processing and administrative costs and places pressure on STADIOalot's already thin margins. The issue also affects customers, marketplace sellers and the efficiency of the organisation's logistics operations. Importantly, the briefing pack indicates that return behaviour may be predictable from factors such as products, sellers, size-and-colour combinations and customer purchasing patterns. However these patterns are not currently being used proactively before an order is confirmed. This creates an opportunity to shift from primarily managing returns after they occur towards identifying potentially high-risk orders before the cost of a return is incurred. Addressing this issue is important as continued growth in avoidable returns could further reduce profitability and undermine STADIOalot's 2030 strategy to break the returns spiral and improve the economics of each parcel.


Data Science methods can assist by using STADIOalot's existing transactional, product, customer-behaviour, seller and historical returns data to identify patterns associated with return risk. A supervised machine learning can be used to estimate the probability that an order will be returned using information available at or before the point of purchase. These predictions could potentially support targeted interventions, such as improving product or sizing information, identifying products or sellers with higher return risk and supporting seller-performance reviews.


Therefore, this study seeks to generate evidence that may support predictive and proactive decision-making around customer returns. The findings could help STADIOalot identify high-risk orders and the factors associated with returns, providing evidence to support strategies aimed at reducing potentially avoidable returns, improving operational efficiency and contributing to greater profitability.

---

## Problem Statement

Despite STADIOalot's substantial annual order volume, the organisation is confronted with an escalating product return rate, currently standing at 15% overall and 29% for apparel and shoes, which places further pressure on its already thin 1.9% operating margin. Consequently, it remains analytically unclear whether historical transaction records, product catalogue attributes and customer behavioural signals can be systematically leveraged to predict the probability of a product return at the point of purchase. Although STADIOalot records extensive information about customers, orders, products, sellers and historical returns, potentially avoidable returns are currently identified primarily after the purchase has occurred. The briefing pack indicates that return behaviour is influenced by factors such as product characteristics, sellers, size-and-colour combinations and customer purchasing patterns, suggesting that return risk may be predictable before an order is completed.

Therefore, this study aims to develop and evaluate a predictive machine learning classification model using STADIOalot's order, product catalogue and clickstream data to identify transactions at a high risk of return. The project will investigate which customer, product, seller, behavioural and transactional characteristics are associated with return risk and determine whether these characteristics can be used to produce reliable return-risk predictions. The intended outcome is to generate evidence that can support proactive, data-driven interventions, enabling STADIOalot to identify high-risk orders earlier, investigate appropriate preventative strategies and potentially reduce avoidable returns and their associated fulfilment and logistics costs.

---

## Data Request

The data request consists of eight related tables. Each table provides information required to understand factors associated with product returns and to develop predictive features for the machine learning models.

| Table | Purpose |
|---|---|
| **Orders and Transactions** | Provides information about customer orders, products purchased, sellers, prices, promotions, payment methods and delivery options. This forms the core transactional data for the prediction model. |
| **Returns** | Provides information about returned orders, including return reasons, returned quantities, refund outcomes and return costs. This table is primarily used to define the target variable (`Returned = 1/0`) and evaluate the model. |
| **Product Catalogue** | Provides product characteristics such as category, subcategory, brand, size, colour, price, ratings and review information. These attributes can help identify products associated with higher return risk. |
| **Marketplace Sellers** | Provides seller-related information such as seller ratings, tenure, shipping performance, late shipment rates and complaints. This can help determine whether seller characteristics are associated with product returns. |
| **Customer Information** | Provides customer-level characteristics and historical purchasing information, including customer tenure, average order value, historical spending and previous returns. These variables can help identify customer-level return patterns. |
| **Customer Behaviour** | Provides information about customer shopping activity, such as sessions, product views, add-to-basket activity, previous orders and previous returns. This can help identify behavioural patterns associated with return risk. |
| **Inventory and Warehouse** | Provides information about stock availability, stock age, distribution centres and historical fulfilment performance. These variables can help investigate whether inventory and operational factors are associated with returns. |
| **Delivery and Logistics** | Provides information about delivery options, delivery location, route distance and historical delivery performance. These variables can help assess whether logistics and delivery factors contribute to return risk. |

The complete data request, including **column names, data types, examples and additional requirements**, is provided in the Part C Data Request document located in the `docs/` folder.

**Data Request Document:** `docs/Part_C_Data_Request.pdf`

---

## Repository Structure

The GitHub repository is organised to clearly separate datasets, models, experimental configuration and results, statistical analysis, visualisations, notebooks, scripts and supporting documentation. This structure supports reproducibility and makes it easy to locate the different artefacts produced throughout the project.


```text
STADIOalot-Return-Prediction/
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
│
├── models/
│   └── README.md
│
├── experimental_setup/
│   ├── README.md
│   ├── requirements.txt
│   └── configuration.md
│
├── experimental_results/
│   └── README.md
│
├── statistical_helpers/
│   └── README.md
│
├── visualisations/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── scripts/
│   └── README.md
│
└── docs/
    ├── README.md
    └── Data_Request.pdf
```
    
## Repository Structure

| Folder | What goes there |
|---|---|
| `data/raw/` | Stores original datasets received from the client. Raw data is kept unchanged to preserve the original source data.|
| `data/processed/` | Stores cleaned, transformed and prepared datasets used for analysis and model development.|
| `models/` | Stores trained machine learning models, model artefacts and related files.|
| `experimental_setup/` | Contains the project environment and experimental configuration, including requirements.txt and configuration documentation. |
| `experimental_results/` | Stores model evaluation results, performance metrics, experiment outputs and related findings. |
| `statistical_helpers/` | Contains statistical helper functions, statistical tests and scripts used to support model analysis and comparison. |
| `visualisations/` | Contains visualisation scripts and generated graphs, charts and other visual outputs used for exploratory analysis and model evaluation.|
| `notebooks/` | Contains Jupyter notebooks used for exploratory data analysis, preprocessing, feature engineering, modelling and evaluation. |
| `scripts/` | Contains reusable Python scripts used for data processing, modelling, evaluation and other project tasks. |
| `docs/` | Contains supporting project documentation. |

---

## RAAIDD Log

The RAAIDD log identifies the key **Risks, Actions, Assumptions, Issues, Decisions and Dependencies** associated with the proposed STADIOalot customer-return prediction project. Each entry is specifically linked to the objective of using Data Science to predict the probability that an order will be returned before the order is confirmed.

## Risks

| Risk | Description | Potential Impact | Mitigation |
|---|---|---|---|
| **Data leakage** | Information that becomes available only after an order is placed, such as return outcomes, refund information or post-delivery behaviour, may accidentally be included as predictive features. | The model could appear highly accurate during testing but fail to perform realistically on new orders. | Define the prediction point as the point of order confirmation and conduct a feature-by-feature leakage review before model training. |
| **Inconsistent product data** | STADIOalot has approximately 38 million SKUs and the briefing pack identifies inconsistent product categorisation. | Incorrect or inconsistent categories could reduce model quality and make product-level patterns difficult to interpret. | Profile product data, standardise relevant categories and document all transformations during preprocessing. |
| **Unreliable seller information** | Some seller-supplied fields are identified as unreliable in the briefing pack. | Poor-quality seller information could introduce noise or bias into the model. | Assess seller-data completeness and reliability and exclude or transform variables that do not meet defined quality standards. |
| **Class imbalance** | Returned orders may represent a smaller proportion of total orders than non-returned orders. | A model could achieve high accuracy while performing poorly when identifying actual returns. | Evaluate the target distribution and use precision, recall, F1-score and ROC-AUC in addition to accuracy. |
| **Insufficient predictive performance** | Customer, product and seller characteristics may not provide enough information to reliably predict individual returns. | The model may not provide sufficient value for operational decision-making. | Establish a baseline model and compare multiple suitable classification approaches. |
| **Concept drift** | Customer purchasing behaviour, product trends and seller performance may change over time. | A model trained on older data may become less accurate when applied to newer orders. | Use time-aware validation where appropriate and monitor performance across different periods. |
| **High data volume** | STADIOalot processes approximately 58 million orders per year and maintains extensive behavioural and product data. | Large datasets may increase processing time and computational requirements. | Determine an appropriate modelling sample, optimise data processing and document sampling decisions. |
| **Privacy and data governance** | Customer-level behavioural data may contain personally identifiable information that is not necessary for modelling. | Inappropriate use of personal information could create privacy and governance concerns. | Request only data necessary for the project, minimise personally identifiable information and follow applicable data-governance requirements. |

---

## Actions

| Action | Description | Expected Outcome |
|---|---|---|
| **Define prediction target** | Define the return status of an order as the target variable indicating whether the order is subsequently returned. | A clear and measurable classification target. |
| **Define prediction point** | Establish that predictions must use information available at or before order confirmation. | A realistic modelling scenario and reduced risk of data leakage. |
| **Request required data** | Request order, product, customer-behaviour, seller and historical returns data from the client. | Sufficient data to investigate the return-prediction problem. |
| **Profile the data** | Assess data types, missing values, duplicates, outliers, inconsistent categories and identifier quality. | A clear understanding of data quality and preprocessing requirements. |
| **Integrate datasets** | Join relevant datasets using identifiers such as Order ID, Customer ID, SKU and Seller ID. | An integrated analytical dataset suitable for modelling. |
| **Clean the data** | Address missing, inconsistent, duplicated and invalid records. | A reliable dataset for analysis and modelling. |
| **Engineer features** | Create pre-order variables such as historical customer return rate, product return rate and seller performance indicators. | Predictive variables that represent return-risk patterns. |
| **Conduct exploratory analysis** | Investigate relationships between return behaviour and customer, product, seller and transaction characteristics. | Identification of potentially important predictive patterns. |
| **Develop baseline model** | Establish a simple classification model against which more advanced models can be compared. | A benchmark for evaluating model improvements. |
| **Develop and compare models** | Train multiple suitable supervised classification models. | Evidence of which modelling approach provides the most useful predictions. |
| **Evaluate models** | Assess accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices. | A comprehensive understanding of model performance. |
| **Interpret model results** | Identify variables and patterns associated with higher return risk. | Business insights that complement individual predictions. |
| **Develop recommendations** | Translate findings into potential interventions for STADIOalot. | Evidence-based recommendations for proactive return management. |

---

## Assumptions

| Assumption | Description | Why It Matters |
|---|---|---|
| **Historical returns are accurately recorded** | It is assumed that historical data correctly identifies returned and non-returned orders. | Incorrect target labels could significantly affect model training. |
| **Orders can be linked across datasets** | It is assumed that consistent identifiers such as Order ID, Customer ID, SKU and Seller ID exist across relevant datasets. | Dataset integration depends on reliable identifiers. |
| **Historical data is representative** | It is assumed that historical customer and product behaviour contains patterns relevant to future return behaviour. | The model depends on historical patterns being informative about future outcomes. |
| **Sufficient historical data exists** | It is assumed that enough returned and non-returned orders are available to develop and evaluate a model. | A sufficiently large sample is required for reliable machine learning. |
| **The prediction point can be established** | It is assumed that the business can identify the exact point at which a return-risk prediction would be generated. | This determines which variables can legitimately be used by the model. |
| **Pre-order information is available** | It is assumed that relevant customer, product, seller and transaction information is available before order confirmation. | These variables form the basis of the prediction. |
| **Customer behaviour contains predictive signals** | It is assumed that browsing, purchasing and previous-return behaviour contains information related to future returns. | If meaningful patterns do not exist, predictive performance may be limited. |
| **Seller performance is relevant** | It is assumed that seller characteristics may contribute to return risk. | Seller-level information may help explain differences in return behaviour. |
| **Data can be appropriately standardised** | It is assumed that inconsistent product categories and seller information can be cleaned sufficiently for analysis. | Poor standardisation could reduce the usefulness of product and seller features. |

---

## Issues

At the proposal stage, several potential issues have been identified but cannot yet be confirmed until the requested data is received and assessed.

| Issue | Description | Current Status / Proposed Response |
|---|---|---|
| **Data fragmentation** | Browsing, ordering, warehouse, delivery and returns data may exist across separate systems and may not be fully integrated. | To be investigated during data integration and profiling. |
| **Inconsistent product categorisation** | Product categories may be inconsistent across STADIOalot's large product catalogue. | To be assessed during data cleaning and standardisation. |
| **Unreliable seller-supplied fields** | Some marketplace seller information may not be sufficiently reliable for modelling. | Data-quality checks will determine which fields are suitable. |
| **Unknown feature availability** | It is not yet confirmed which variables are consistently available at the exact point of purchase. | The data request and stakeholder discussions will establish feature availability. |
| **Unknown model performance** | It is not yet known whether the available data will provide sufficient predictive power. | Baseline and comparative modelling will be used to assess feasibility. |

---

## Decisions

| Decision | Rationale |
|---|---|
| **Focus the project on customer-return prediction** | Returns represent a significant business challenge and are a priority within STADIOalot's strategic objectives. |
| **Use return status as the primary target variable** | The business objective is to identify whether an order is likely to result in a return. |
| **Predict at or before order confirmation** | This ensures that the project addresses the problem proactively rather than simply explaining returns after they occur. |
| **Use supervised machine learning** | Historical orders can be labelled according to whether they were subsequently returned, making this suitable for a classification problem. |
| **Exclude post-order information from predictive features** | Post-order variables would create data leakage and would not be available when the prediction is required. |
| **Compare multiple classification models** | Comparing models provides stronger evidence than relying on a single modelling approach. |
| **Use multiple evaluation metrics** | Return prediction requires consideration of the model's ability to identify actual returns, so accuracy alone is insufficient. |
| **Prioritise interpretability alongside predictive performance** | STADIOalot needs to understand factors associated with return risk so that predictions can potentially inform business interventions. |
| **Treat the model as decision support** | Predictions should provide evidence to support business decisions rather than automatically determine how every high-risk order is handled. |

---

## Dependencies

| Dependency | Description | Relationship |
|---|---|---|
| **Client data access → Data profiling** | The required datasets must be supplied before data quality can be assessed. | Data profiling depends on receiving the client data. |
| **Data profiling → Data cleaning** | Missing, inconsistent and invalid data must first be identified. | Cleaning actions depend on profiling results. |
| **Data cleaning → Data integration** | Key identifiers and data-quality issues need to be understood before datasets are combined. | Reliable integration depends on appropriately prepared datasets. |
| **Data integration → Feature engineering** | Predictive features may require information from multiple datasets. | Feature engineering depends on successful dataset integration. |
| **Feature engineering → Exploratory analysis** | Predictive variables need to be constructed before their relationships with return status can be investigated. | Analysis depends on suitable features. |
| **Exploratory analysis → Model development** | Initial data patterns and characteristics inform the modelling approach. | Modelling decisions depend on understanding the data. |
| **Model development → Model evaluation** | Models must first be trained before predictive performance can be measured. | Evaluation depends on completed model training. |
| **Model evaluation → Model selection** | The final model should be selected using evidence from comparative performance. | Model selection depends on evaluation results. |
| **Model interpretation → Business recommendations** | Recommendations should be based on demonstrated predictive patterns and model findings. | Business recommendations depend on interpretable results. |
| **Stakeholder validation → Final recommendations** | Business stakeholders need to confirm whether proposed interventions are operationally realistic. | Final recommendations depend on stakeholder feedback. |

---

## RAAIDD Summary

The RAAIDD analysis demonstrates that the proposed project depends on more than simply developing a machine learning model. Reliable and appropriately structured data, careful prevention of data leakage, suitable feature engineering, comparative model evaluation and interpretation of results are all required to determine whether return prediction can provide meaningful business value.

The RAAIDD log will be reviewed and updated throughout the project lifecycle as new information becomes available, particularly once the requested datasets have been received and assessed.

---

## Project Scope

### In Scope

The project will investigate whether Data Science and supervised machine learning methods can be used to predict the probability that an STADIOalot order will subsequently be returned.

The project will focus on:

- Collecting and understanding historical order and return data.
- Analysing customer purchasing behaviour relevant to return risk.
- Analysing product and category characteristics associated with returns.
- Analysing seller-level characteristics and return patterns.
- Investigating size-and-colour combinations where relevant to return behaviour.
- Cleaning, integrating and preparing the required datasets.
- Performing exploratory data analysis to identify patterns associated with returns.
- Engineering features using information available at or before the point of order confirmation.
- Developing supervised machine learning classification models.
- Comparing model performance using appropriate evaluation metrics.
- Identifying the factors that contribute most strongly to return risk.
- Providing evidence-based recommendations that could support proactive return-management strategies.

### Out of Scope

The project will not:

- Automatically cancel or modify customer orders.
- Automatically prevent customers from purchasing high-risk products.
- Implement a production machine learning system within STADIOalot's live e-commerce platform.
- Make final business decisions on behalf of STADIOalot.
- Determine whether an individual customer should be penalised because of their predicted return risk.
- Use information that would only become available after the order or return has occurred as predictive input.
- Guarantee that predicted high-risk orders will result in returns.
- Attempt to eliminate all product returns, as some returns are unavoidable and may be legitimate.

The project is intended to provide a predictive and decision-support capability rather than a fully automated operational solution.

---