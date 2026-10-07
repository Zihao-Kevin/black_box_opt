# Black-box optimization with Q + Delta^*

We train a model whose output is scored by a black box (a solver, a tool-using agent, a scheduler). Plain REINFORCE gradients are noisy. We fit a reward model from the paid calls and use it as a control variate: Q removes most of the noise, and the row-wise residual Δ* removes more. The gradient stays unbiased.

## Setup

```bash
conda env create -f environment.yml
conda activate black_box_opt
cp mcp_dfl_prototype/.env.example mcp_dfl_prototype/.env   # add OPENAI_API_KEY and HF_TOKEN
```

## Experiments

| Folder | Task | Start here |
|---|---|---|
| `toy_example/` | Decision-focused learning on a small QP | `DFL_QP_Black_Box_new.ipynb` |
| `mcp_dfl_prototype/` | An LLM picks MCP tool servers for a live agent | `Live_agent_rowwise.ipynb` |
| `experiments/jobshop/` | An LLM picks job-shop scheduling rules | its own `README.md` |

`baselines.py` holds RLOO, GRPO, OTB and LAX/RELAX.

## Figures

```bash
python paper_figs_src/make_paper_figs.py   # writes paper_figs/
```

Run outputs are not in git. Each figure reads its run folder (for example `toy_example/_qp_adam_runs/`). Run the matching notebook or launch script first. The script docstring lists which runs each figure needs.
