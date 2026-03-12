# Lab: Transformer Query Contextual Support Reader

## Feature Overview
This sub-branch introduces a **Neural Support Reader** that replaces static neighborhood moments (mean, variance) with a learned representation derived via Transformer Cross-Attention.

## Rationale
While the "Challenger" model improved performance using higher-order moments (Skewness/Kurtosis), it still relied on human-engineered features. The **Support Reader** allows the model to:
1.  **Learn the Manifold**: Automatically identify which neighbors in the support set are most informative for the current query.
2.  **In-Context Learning**: Adopt the TabPFN/TabR philosophy where the "model" is partially defined by its local data context at inference time.
3.  **Cross-View Synergy**: By using the same embedding space for support and query, the router can detect subtle regime shifts that scalar variance cannot capture.

## Architecture: Query-Contextual vs. Task-Level
Unlike TabPFN, which consumes the entire dataset as a "task," our reader is **Query-Contextual**. 
- **Query ($Q$):** The current row embedding $x_q$.
- **Keys/Values ($K, V$):** The top-K nearest neighbors from each expert's feature subspace.
- **Output:** A fixed-length "Support Token" appended to the expert's contextual token.

## Research & Supporting Evidence
- **TabR (2023):** Demonstrated that "Retrieval-based" tabular models outperform standard MLPs by attending to similar training examples.
- **Perceiver (DeepMind):** Supports using a small latent query (our $x_q$) to attend to a large context (our neighborhood) to maintain computational efficiency.
- **Attention is All You Need (Vaswani et al.):** The fundamental Cross-Attention mechanism allows for permutation-invariant processing of the support set.

## Tracibility
- **Branch:** `lab/support-reader`
- **Component:** `TransformerSupportReader`
- **Integrated With:** `CrossAttentionSetRouter`
- **Status:** Experimental (Validated on OpenML Blood/Credit-G)
