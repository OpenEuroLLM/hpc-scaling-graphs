# OpenEuroLLM scaling graphs

Throughput scaling of OpenEuroLLM pretraining runs (Megatron-LM) on the EuroHPC clusters we train on:

| cluster | hosted by | GPUs per node | interconnect | topology | network per node |
|---|---|---|---|---|---|
| [JUPITER](#jupiter) | [JSC, Jülich (DE)](https://www.fz-juelich.de/en/jsc/jupiter) | 4× NVIDIA GH200 (96 GB) | InfiniBand NDR (NVIDIA Quantum-2) | Dragonfly+ | 4 × 200 Gb/s (800 Gb/s) |
| [MareNostrum 5](#marenostrum-5) | [BSC, Barcelona (ES)](https://www.bsc.es/marenostrum/marenostrum-5) | 4× NVIDIA H100 (64 GB HBM2) | InfiniBand NDR | Fat tree | 4 × 200 Gb/s (800 Gb/s) |
| [Leonardo](#leonardo) | [CINECA, Bologna (IT)](https://leonardo-supercomputer.cineca.eu) | 4× NVIDIA A100 (64 GB) | InfiniBand HDR (NVIDIA Mellanox) | Dragonfly++ | 2 × dual-port HDR (400 Gb/s) |
| [LUMI](#lumi) | [CSC, Kajaani (FI)](https://lumi-supercomputer.eu/) | 4× AMD MI250 (8 GPU dies) | HPE Cray Slingshot-11 | Dragonfly | 4 × 200 Gb/s (800 Gb/s), each endpoint on a different switch |
| [Snellius](#snellius) | [SURF, Amsterdam (NL)](https://www.surf.nl/en/services/compute/snellius-the-national-supercomputer) | 4× NVIDIA H100 (94 GB) | InfiniBand NDR | Fat tree | 4 × 200 Gb/s (800 Gb/s) |

Each graph comes in two forms:
- **Standardized**
  ([`figures/standardized/`](figures/standardized/)): re-plotted in one layout with one naming scheme, so
  clusters and models can be compared side by side.
- **Original**
  ([`figures/original/`](figures/original/)): the figures as their authors posted them, mostly on the
  OpenEuroLLM Taskboard.

The config column links to the [oellm-autoexp](https://github.com/OpenEuroLLM/oellm-autoexp) file at a pinned commit, where
one could be found. "Candidate" means the source does not say which file produced the plotted runs.

**Jump to:** [JUPITER](#jupiter) · [MareNostrum 5](#marenostrum-5) · [Leonardo](#leonardo) · [LUMI](#lumi) ·
[Snellius](#snellius) · [by model size](#by-model-size) · [conventions](#conventions)

## Strong vs weak scaling

- **Strong scaling** keeps the global batch size (GBS) fixed while GPUs are added, so each GPU gets less work.
  Per-GPU throughput eventually falls.
- **Weak scaling** grows the GBS with the GPUs, so the work per GPU stays constant (fixed GAS, `GBS n/GPU`).
  Ideally per-GPU throughput stays flat.

Every standardized figure shows TFLOP/s per GPU (left) and tokens/s per GPU (right) against the GPU count. The
green dashed line is ideal scaling. Strong-scaling figures add points at other GBS values of the same cluster
and model as marker-only extras.

## By cluster

### JUPITER

JSC, 4× NVIDIA GH200 (96 GB) per node, 680 W power cap (peak achieved TFLOPs and thus MFU will be lower compared to 900W cards, see [here](https://github.com/OpenEuroLLM/Taskboard/issues/203) for more information on the power caps effect on TFLOPs)

| model | strong | weak | precision, parallelism | GPUs | source | config |
|---|---|---|---|---|---|---|
| Qwen3 9B dense | [GBS 1024 (+2048)](figures/standardized/jupiter/jupiter_qwen3-9b-dense_strong.png) | — | BF16, TP 4, PP 1, MBS 4 | 4–1024 | [Taskboard#213](https://github.com/OpenEuroLLM/Taskboard/issues/213#issuecomment-4442082570) (tvosch) | [config](https://github.com/OpenEuroLLM/oellm-autoexp/blob/6839afb76133/config/experiments/thomas/qwen3_dense_jupiter.yaml) |
| Qwen3.5 9B GDN hybrid | [GBS 2048](figures/standardized/jupiter/jupiter_qwen3.5-9b-gdn-hybrid_strong.png) | [GAS 16, GBS 16/GPU](figures/standardized/jupiter/jupiter_qwen3.5-9b-gdn-hybrid_weak.png) | BF16, TP 1, PP 1, MBS 1, overlap grad/param | 4–2048 | [Taskboard#208](https://github.com/OpenEuroLLM/Taskboard/issues/208) (dharmsen): [strong](https://github.com/OpenEuroLLM/Taskboard/issues/208#issuecomment-4359697835), [weak](https://github.com/OpenEuroLLM/Taskboard/issues/208#issuecomment-4361950922) | [base config](https://github.com/OpenEuroLLM/oellm-autoexp/blob/b13ac89243a5/config/experiments/exp_dalton/speed_hybrid/megatron_gdn_qwen3.5-9B.yaml) |
| Qwen3 30B-A3B MoE | [GBS 4096 (+2048)](figures/standardized/jupiter/jupiter_qwen3-30b-a3b-moe_strong.png) | [GAS 64, GBS 32/GPU](figures/standardized/jupiter/jupiter_qwen3-30b-a3b-moe_weak.png) | strong: FP8, TP 1, PP 2, VPP 2, EP 4, MBS 4; weak: BF16 implied, TP 1, PP 2, EP 4, MBS 1 | 8–1024 | [Taskboard#211](https://github.com/OpenEuroLLM/Taskboard/issues/211) (Vmjkom): [strong](https://github.com/OpenEuroLLM/Taskboard/issues/211#issuecomment-4372282006), [weak](https://github.com/OpenEuroLLM/Taskboard/issues/211#issuecomment-4359614377) | — (branch deleted) |
| Qwen3 32B dense | [GBS 4096 (+2048, 8192)](figures/standardized/jupiter/jupiter_qwen3-32b-dense_strong.png) | [GAS 8, GBS 1/GPU](figures/standardized/jupiter/jupiter_qwen3-32b-dense_weak.png) | FP8, TP 4, PP 4, VPP 2, MBS 2 | 32–4096 | korbi's speed sweep, 2026-08 (run logs; not posted). Extras: [Taskboard#217](https://github.com/OpenEuroLLM/Taskboard/issues/217#issuecomment-5271958100) (Vmjkom) | [strong](https://github.com/OpenEuroLLM/oellm-autoexp/blob/0f55a1562272/config/experiments/korbi/speed_scaling/fp8_strong_scaling.yaml), [weak](https://github.com/OpenEuroLLM/oellm-autoexp/blob/0f55a1562272/config/experiments/korbi/speed_scaling/fp8_weak_batch.yaml), [results](https://github.com/OpenEuroLLM/oellm-autoexp/blob/0f55a1562272/config/experiments/korbi/speed_scaling/RESULTS.md) |
| Qwen3 235B-A22B MoE | — | [GAS 128, GBS 8/GPU](figures/standardized/jupiter/jupiter_qwen3-235b-a22b-moe_weak.png) | FP8, TP 2, PP 8, EP 16, MBS 1 | 256–2048 | [Taskboard#206](https://github.com/OpenEuroLLM/Taskboard/issues/206) (tvosch), parallelism from [#207](https://github.com/OpenEuroLLM/Taskboard/issues/207) | [sweep](https://github.com/OpenEuroLLM/oellm-autoexp/blob/3774a0ff8141/config/experiments/thomas/megatron_moe_qwen235BA22B-sweep.yaml), [parallelism](https://github.com/OpenEuroLLM/oellm-autoexp/blob/2264e10d20c8/config/experiments/ville/jupiter/jupiter_qwen-235B_sweep-parallelism.yaml) |

### MareNostrum 5

BSC, 4× NVIDIA H100 (64 GB HBM2) per node.

| model | strong | weak | precision, parallelism | GPUs | source | config |
|---|---|---|---|---|---|---|
| Qwen3 9B dense | [GBS 1024](figures/standardized/marenostrum5/marenostrum5_qwen3-9b-dense_strong.png) | — | BF16, TP 2, PP 1, MBS 1 | 8–1024 | [Taskboard#213](https://github.com/OpenEuroLLM/Taskboard/issues/213#issuecomment-4442307161) (JoanLlop) | — |
| Qwen3 30B-A3B MoE | [GBS 4096](figures/standardized/marenostrum5/marenostrum5_qwen3-30b-a3b-moe_strong.png) | — | BF16, TP 1, PP 4, VPP 2, EP 4, MBS 4 | 32–128 | [Taskboard#211](https://github.com/OpenEuroLLM/Taskboard/issues/211#issuecomment-6057044334) (ibaucells) | candidates: [bf16](https://github.com/OpenEuroLLM/oellm-autoexp/blob/6bd37fd7aab7/config/experiments/irene/megatron_moe_qwen30BA3B_marenostrum_hybridep_replication_exps_nemo260801_bf16.yaml), [bf16 layout](https://github.com/OpenEuroLLM/oellm-autoexp/blob/6bd37fd7aab7/config/experiments/irene/megatron_moe_qwen30BA3B_marenostrum_hybridep_nemo260801_bf16_layout.yaml) |

### Leonardo

CINECA, 4× NVIDIA A100 (64 GB) per node.

| model | strong | weak | precision, parallelism | GPUs | source | config |
|---|---|---|---|---|---|---|
| Qwen3 9B dense ("baby 9B") | [GBS 1024 (+2048, 4096)](figures/standardized/leonardo/leonardo_qwen3-9b-dense_strong.png) | — | BF16, TP 4, PP 1, MBS 4 | 8–2048 | ecozzoli, oellm-autoexp branch `prod/baby_9b_dense` (not posted on the Taskboard) | candidate: [model + parallelism](https://github.com/OpenEuroLLM/oellm-autoexp/blob/c710f94fc008/config/backend/megatron/baby/models/qwen3_dense_9b.yaml) |

### LUMI

TBA.

### Snellius

TBA.

## By model size

| size | model | JUPITER | MareNostrum 5 | Leonardo |
|---|---|---|---|---|
| 9B | Qwen3 9B dense | [strong](figures/standardized/jupiter/jupiter_qwen3-9b-dense_strong.png) | [strong](figures/standardized/marenostrum5/marenostrum5_qwen3-9b-dense_strong.png) | [strong](figures/standardized/leonardo/leonardo_qwen3-9b-dense_strong.png) |
| 9B | Qwen3.5 9B GDN hybrid | [strong](figures/standardized/jupiter/jupiter_qwen3.5-9b-gdn-hybrid_strong.png), [weak](figures/standardized/jupiter/jupiter_qwen3.5-9b-gdn-hybrid_weak.png) | — | — |
| 30B (3B active) | Qwen3 30B-A3B MoE | [strong](figures/standardized/jupiter/jupiter_qwen3-30b-a3b-moe_strong.png), [weak](figures/standardized/jupiter/jupiter_qwen3-30b-a3b-moe_weak.png) | [strong](figures/standardized/marenostrum5/marenostrum5_qwen3-30b-a3b-moe_strong.png) | — |
| 32B | Qwen3 32B dense | [strong](figures/standardized/jupiter/jupiter_qwen3-32b-dense_strong.png), [weak](figures/standardized/jupiter/jupiter_qwen3-32b-dense_weak.png) | — | — |
| 235B (22B active) | Qwen3 235B-A22B MoE | [weak](figures/standardized/jupiter/jupiter_qwen3-235b-a22b-moe_weak.png) | — | — |

## Conventions

- **File names:** `<cluster>_<model>_<strong|weak>.png`.
- **Titles:** `<Strong|Weak> Scaling Megatron <model> on <cluster> (<precision>, TP n, PP n[, VPP n][, EP n], MBS n, GBS n)`.
  Weak-scaling titles give `GAS n, GBS n/GPU` instead.
- **Sequence length** is 4096 for all runs. TFLOP/s per GPU is Megatron's own throughput figure.
- **Date** (bottom right): when the data was posted or measured.
- **Tokens/s** for JUPITER 9B dense, MareNostrum 5 9B dense and JUPITER 30B-A3B is computed from TFLOP/s, because
  those sources reported TFLOP/s only.
- **Extra markers in the JUPITER 32B strong figure** are BF16 runs with PP 2, so they differ from the FP8 main
  curve in more than GBS.

## Layout

```
figures/
  standardized/<cluster>/<cluster>_<model>_<strong|weak>.png
  original/<cluster>/<model>/<source>/<file as posted>
```

## Adding a graph

1. Put the author's figure in `figures/original/<cluster>/<model>/<source>/`, where `<source>` is
   `tb<issue>-<github-login>` or `<person>-<yyyy-mm>`.
2. Re-plot it in the standard layout and title format. Save it as
   `figures/standardized/<cluster>/<cluster>_<model>_<strong|weak>.png`.
3. Link the config at a pinned commit (not a branch) if you have one.
4. Add a row to the cluster table and the model-size table above, with the issue link.
