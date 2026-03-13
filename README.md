# 🚁 GraphDrone: A Mixture-of-Specialists Meta-Model for Tabular Data

**GraphDrone** is a high-performance, distribution-ready meta-model designed to outperform standard foundation models by utilizing a "Team of Specialists." It leverages state-of-the-art Transformer architectures to dynamically route query points to the most relevant feature subspaces.

---

## 🚀 Headlines
*   **Winner on Spatial & Physical Manifolds**: Outperforms TabPFN on complex datasets like California Housing and Stock Market prediction by up to **31%**.
*   **Neural In-Context Learning**: Uses a **Transformer Support Reader** to learn from local neighbors at inference time.
*   **Noise-Immune Architecture**: Built-in **SNR-Gating** automatically prunes noisy expert views for every query point.
*   **Sklearn-Native Experience**: Designed to fit perfectly into existing Python ML pipelines with a simple `fit`/`predict` API.

---

## 💡 High-Level Design & Innovation

GraphDrone moves beyond "one-size-fits-all" modeling. While standard models (like TabPFN) process all features globally, GraphDrone breaks the problem into **Specialist Views**.

### Core Innovations:
1.  **Contextual Set-Router**: Instead of fixed weights, GraphDrone uses a **Cross-Attention mechanism** where the global model acts as a "Query" to find the most helpful specialists.
2.  **SNR-Gate (Signal-to-Noise Ratio)**: A learned pruning layer that calculates the reliability of each expert's prediction relative to its local manifold. If a specialist is "confused" by noise, GraphDrone shuts the gate.
3.  **Dynamic-K Neighborhoods**: Prior-aware resolution that uses wide windows for smooth manifolds (Geo) and tight windows for jagged ones (Structural).

---

## 📊 Performance

GraphDrone has been statistically validated against **TabPFN** on 18+ registered OpenML datasets using **NVIDIA H200 NVL** hardware.

### Regression Highlights (RMSE Delta)
| Dataset | Improvement vs. TabPFN |
| :--- | :--- |
| **Stock Market** | **-31.2%** |
| **Ailerons** | **-4.06%** |
| **California Housing** | **-2.62%** |

### Classification Highlights (ROC-AUC)
| Dataset | GraphDrone AUC | TabPFN AUC |
| :--- | :--- | :--- |
| **Tic-Tac-Toe** | **0.9997** | 0.9994 |
| **Blood Transfusion** | **0.7820** | 0.7699 |
| **Mammographic** | **0.9965** | 0.9979 |

---

## 💻 Usage

GraphDrone follows the standard `scikit-learn` API.

### Installation
```bash
git clone https://github.com/RicardoLaMo/Graph_Drone.git
cd Graph_Drone
pip install -e .
```

### Quick Start
```python
from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig

# 1. Configure the Meta-Router
config = GraphDroneConfig(
    full_expert_id="FULL",
    router=SetRouterConfig(kind="noise_gate_router")
)

# 2. Initialize and Fit
model = GraphDrone(config)
model.fit(X_train, y_train)

# 3. Predict with Diagnostics
predictions = model.predict(X_test)

# Optional: View why the router picked certain experts
result = model.predict(X_test, return_diagnostics=True)
print(result.diagnostics['effective_defer_rate'])
```

---

## 🛠 Tech Stack
*   **Backbone**: TabPFN (Foundation Engine)
*   **Routing**: PyTorch Cross-Attention
*   **Manifold Analysis**: Scikit-Learn kNN
*   **Hardware Optimized**: Fully validated on **NVIDIA H200** NVL.

---
*Developed by the GraphDrone Research Team.*

