# Theory: Prior-Awareness in GraphDrone

## 1. The PFN Paradigm
PFNs learn by being exposed to millions of synthetic datasets. The "Prior" is the generator of these datasets. TabPFN 2.5 improves this by:
- **Increasing Context Resolution**: Better handling of how the "support set" (neighbors) interacts with the query point.
- **Hierarchical Priors**: Combining simple linear priors with complex non-linear ones.

## 2. Leveraging Priors for Specialist Views
GraphDrone can now be conceptualized as a **Multi-Prior Mixture of Experts**.

### A. View-Specific Priors
- **GEO Expert**: Assumes a **Gaussian Process Prior** (points close in space have similar values).
- **DOMAIN Expert**: Assumes a **Tabular/Decision-Tree Prior** (points with similar categorical attributes have similar values).
- **LOWRANK Expert**: Assumes a **Linear/Manifold Prior**.

### B. Prior-Matching Tokens
We add a **Prior-Likelihood Field** to our Token Builder.
$$L_{prior} = \text{Softmax}(\text{Alignment between query and view-manifold})$$
This tells the router: "This query point looks like it came from the GEO prior."

### C. Differential kNN Dynamics
Instead of a fixed $K=15$ for all views, we use **Prior-Informed K**:
- If the manifold is smooth (GEO), we use a larger $K$ to reduce noise.
- If the manifold is sparse/discrete (DOMAIN), we use a smaller, tighter $K$.

## 3. Implementation Plan
1. **Dynamic-K Expert Factory**: Allow experts to define their own neighborhood size.
2. **Prior-Alignment Token**: Compute local manifold density as a proxy for prior-matching.
3. **ICL-Router**: Feed the Transformer Support Reader the "Prior Type" as an embedding.
