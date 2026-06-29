---
license: apache-2.0
language:
- en
tags:
- deep-research
- agent
- reinforcement-learning
- web-search
- local-tools
---

# LiteResearcher-4B

**LiteResearcher-4B** is a low-cost, scalable 4B deep research agent trained with the LiteResearcher agentic RL framework. The RL stage is performed entirely in a local search/browse environment, requiring no external search or browsing APIs during RL and incurring zero marginal API cost.

Despite its compact scale, LiteResearcher-4B achieves **71.3%** on GAIA and **78.0%** on Xbench-DeepSearch, surpassing open-source deep research agents up to 8× larger and matching strong commercial systems on representative benchmarks.

## Highlights

- **Compact 4B agent:** designed for efficient deep research rather than brute-force model scale.
- **Low-cost RL scaling:** the RL stage runs fully locally with no external API consumption.
- **Stable local tools:** local search and browsing tools mirror real-world search dynamics while avoiding live web variance and per-call API cost.
- **Strong benchmark performance:** 71.3% on GAIA and 78.0% on Xbench-DeepSearch.

## Links

- Paper: https://arxiv.org/abs/2604.17931
- Code: https://github.com/simplexai-labs/LiteResearcher
- Project page: https://simplexai-labs.github.io/LiteResearcher/

## Citation

```bibtex
@article{li2026literesearcher,
  title={LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent},
  author={Li, Wanli and Qu, Bince and Pan, Bo and Zhang, Jianyu and Liu, Zheng and Zhang, Pan and Chen, Wei and Zhang, Bo},
  journal={arXiv preprint arXiv:2604.17931},
  year={2026}
}
```
