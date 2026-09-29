# Sobolev Novelty for Symbolic Regression

Implementations of **SN-MCTS, SN-GP, SN-IGSR, and Tensor-SN-DSRRANS** with their corresponding Base controls, plus **SN-E2ESR** pretraining.

## Directory structure

```text
src/           Method implementations, Base controls, and SN computation
scripts/       Run entrypoints and source/data checks
configs/       Method parameters
data/          Task lists and turbulence arrays
tests/         Mathematical, pairing, and implementation tests
requirements/  Environment-specific dependencies
licenses/      Third-party licenses
```

## Data preparation

| Dataset | Size | Methods |
|---|---|---|
| SRBench whitebox | 133 tasks | MCTS, GP |
| SRBench blackbox | 122 tasks | MCTS, GP, IGSR |
| Turbulence `case_0p8` | 9,600 rows | Tensor-SN-DSRRANS |
| Random formulas and numerical samples | Generated online | SN-E2ESR pretraining |

Place the SRBench data at:

```text
data/pmlb/datasets/<task>/<task>.tsv.gz
```

```bash
# Check bundled data
python scripts/verify_data.py

# Validate SRBench data
python scripts/verify_data.py --require-external
```

Task lists are in `data/srbench/` and method parameters are in `configs/`. Data splits and GP initial populations are generated from the seed at runtime.

| Method | Data split and evaluation |
|---|---|
| MCTS / GP whitebox | Seeded 75/25 search-pool/test split after dropping incomplete rows; sample up to 200 search rows and report R² on the disjoint test set |
| MCTS / GP blackbox | 75/25 search-pool/test split, with up to 200 search rows; MCTS uses all features, GP uses the first 10 |
| IGSR blackbox | 25% test set; up to 4,000 remaining rows, split 60/40 for training and validation |
| Turbulence | Default 75/25 row split within the same flow field; full-data fitting is also supported |

GP/MCTS and their Base/SN variants share the same task/seed
partition. At least two test rows are reserved; datasets with fewer than four
complete rows are rejected.

## Installation

Use Python 3.12 and run from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/core.txt
```

MCTS and GP run on CPU. Turbulence dependencies are listed in `requirements/dsrrans.txt`. Install `requirements/igsr.txt` in a separate environment for IGSR, with a separately deployed model service.

## SN-MCTS

SN represents additive terms using function values and input derivatives, measuring each term's normalized projection residual against the reference span. MCTS reranks a quality shortlist using SN and attempts low-novelty-term removal under a fit-quality constraint. Caching and parent-child incremental updates reuse geometry information.

```bash
python scripts/run_mcts.py --method base --scope blackbox --task 1027_ESL --seed 20260808 --smoke --output outputs/mcts_base_smoke
python scripts/run_mcts.py --method sn --scope blackbox --task 1027_ESL --seed 20260808 --smoke --output outputs/mcts_sn_smoke
```

Remove `--smoke` for the full configuration: export the last completed-iteration checkpoint within 900 seconds, with a process limit of 1,800 seconds. Whitebox example: `--scope whitebox --task strogatz_lv1`.

Search parameters: `sample_num=200`, `child_num=50`, `n_playout=100`, `d_playout=10`, `max_len=30`, and `ratio=1`. SN parameters: `alpha=0.01`, `tau=1/sqrt(10)`, value and gradient weights of 1, geometry sample and shortlist sizes of 64, at most one removal per step, and acceptance tolerance of 0.

## SN-GP

SN uses a cross-generation structure archive, conditional novelty, and residual information to guide recombination and structured candidate proposals. The exported expression is selected by Base reward `0.999**complexity/(2-search_internal_r2)`.

```bash
python scripts/run_gp.py --method base --scope blackbox --task 1027_ESL --seed 20260808 --smoke --output outputs/gp_base_smoke
python scripts/run_gp.py --method sn --scope blackbox --task 1027_ESL --seed 20260808 --smoke --output outputs/gp_sn_smoke
```

Remove `--smoke` for the full configuration: 1,000 individuals and 60 generations. Whitebox example: `--scope whitebox --task strogatz_lv1`.

Base and SN share initial populations and search rows. The elite count is 10, tournament size is 20, crossover probability is 0.9, and subtree, hoist, and point mutation probabilities are each 0.01. Runtime soft and hard limits are 12 and 16 hours. Full parameters are in `configs/`.

## SN-IGSR

IGSR uses SRBench blackbox data. SN diagnoses parent expressions on training rows and turns up to three low-novelty terms and two reference terms into textual feedback for the next LLM proposal. Model selection uses validation NMSE.

Deploy an OpenAI-compatible service for **Llama-3.1-8B-Instruct, BF16, without quantization**, with the served model name `llama-3.1-8b-instruct`. Use vLLM `0.6.6.post1` and PyTorch `2.5.1+cu121` in the serving environment. Replace the model paths below with local paths and configure the GPU and port as needed:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8010 \
  --model /path/to/Llama-3.1-8B-Instruct \
  --tokenizer /path/to/Llama-3.1-8B-Instruct \
  --served-model-name llama-3.1-8b-instruct \
  --api-key igsr-local-vllm \
  --load-format safetensors --dtype bfloat16 \
  --max-model-len 16384 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.88 --max-num-seqs 1 \
  --enable-prefix-caching --swap-space 4 --seed 20260805 \
  --enforce-eager --disable-frontend-multiprocessing --disable-log-requests
```

Set the client's `IGSR_API_KEY` to match the server key; the default is `igsr-local-vllm`. Run in the IGSR environment:

```bash
# Check data configuration
python scripts/run_igsr.py --scope blackbox --task 1027_ESL --seed 42 --check-data --output outputs/unused

# Run Base then SN for the same task and seed, sharing a completion tape
python scripts/run_igsr.py --method pair --scope blackbox --task 1027_ESL --seed 42 --endpoint http://127.0.0.1:8010/v1 --output outputs/igsr_pair
```

Each method uses a budget of 300,000 logical tokens, checked at node boundaries; cache hits count toward the budget. Each pair uses the same endpoint, model, and seed. Temperature is 1, output is capped at 2,048 tokens per call, the successor count is 5, search depth is 10, and the UCT coefficient is 1.41421356. Training uses `Ridge(alpha=1e-8)`; SN geometry uses up to 64 training rows.

## Tensor-SN-DSRRANS

Deterministic basis selection operates on the DSRRANS tensor representation. Base selects the term with the greatest SSE reduction after joint refitting. SN selects the highest same-channel conditional novelty among candidates achieving at least 94% of the best reduction.

```bash
python scripts/run_dsrrans.py --method base --seed 1000003 --smoke --output outputs/tensor_base_smoke
python scripts/run_dsrrans.py --method sn --seed 1000003 --smoke --output outputs/tensor_sn_smoke
```

Remove `--smoke` for 100 forward-selected terms. Add `--prune-to 20` to apply the same exact-loss backward pruning rule to both methods. The default row split is 75/25; `--fit-fraction 1.0` uses the full-data fitting protocol.

Inputs are transformed as `x=tanh(Lambda[:,:2]/2)`, using the first three tensor bases and components `[0,1,4,8]`. Maximum polynomial degree is 12, yielding 273 channel-specific candidate terms. SN value and gradient weights are 1 and 0.005; derivatives hold the tensor bases fixed.

## SN-E2ESR

E2ESR generates random formulas and numerical inputs online. SN applies leave-one-out novelty filtering to expanded top-level additive terms, and accepted samples enter Transformer pretraining. Terms share finite-value rows, standardized input coordinates, and the candidate formula's RMS output scale. Every term in a multi-term formula must have novelty strictly greater than `1/sqrt(10)`; a single term has novelty 1. The fast path uses condition-checked Gram/Cholesky calculations, with a projection-based reference path.

E2ESR sources are in `src/generative/e2esr/`, with independent SN dependencies in `src/e2esr_frozen/sobolev/`. Create the Python 3.10 environment:

```bash
conda env create --file requirements/e2esr.yml
conda activate e2esr-sobolev-novelty
```

The CPU smoke uses 1 epoch and 2 training steps:

```bash
bash scripts/run_e2esr.sh --cpu true --debug \
  --max_epoch 1 --n_steps_per_epoch 2 --batch_size 2 --num_workers 0 --collate_queue_size 4 \
  --enc_emb_dim 64 --dec_emb_dim 64 --n_enc_layers 1 --n_dec_layers 1 \
  --n_enc_heads 4 --n_dec_heads 4 --eval_size 8 \
  --eval_on_pmlb false --eval_on_pmlb_blackbox false --eval_in_domain false \
  --dump_path outputs/e2esr_smoke
```

SN parameters are `sn_threshold=0.31622776601683794`, `sn_geometry_rows=200`, `sn_min_valid_rows=32`, value and gradient weights of 1, and `sn_geometry_seed=20260806`. The launcher supports `PYTHON_BIN` and forwards training arguments. Set model size, batch size, and training steps for full pretraining runs.

```bash
python -m src.generative.e2esr.train --help
python -m src.generative.e2esr.evaluate --help
python -B -m pytest -q -p no:cacheprovider tests/test_e2esr_sobolev_novelty.py
```

Use `--reload_checkpoint` to specify the directory containing `checkpoint.pth`. For PMLB evaluation, set `--pmlb_data_path` to the dataset directory and `--pmlb_metadata_path` to the directory containing `all_summary_stats.tsv` and `feynman_equations.tsv`, then enable `--eval_on_pmlb true`.

## Output metrics

MCTS/GP report the expression, R², and expression complexity. IGSR independently evaluates the exported model after search, reporting `NMSE=MSE/Var(y)`, `R²=1-NMSE`, `accuracy_tol`, `accuracy_tol_max`, and additive term count excluding the intercept. Turbulence metrics include invNRMSE and energy R², which uses target squared energy as its denominator. Each run writes results to its specified output directory.

## Checks and documentation

Specify a new directory for `--output` on each run; E2ESR uses `--dump_path`. Method parameters are in `configs/` and each entrypoint's `--help`.

```bash
python scripts/verify_package.py
python -B -m pytest -q -p no:cacheprovider tests
```

## Sources and licenses

MCTS/GP build on EIC and nd2py, IGSR builds on IGSR, the turbulence representation builds on DSRRANS, and E2ESR builds on Meta's E2ESR implementation. SRBench data uses the PMLB distribution. Third-party licenses are provided for [nd2py](licenses/nd2py.txt), [PMLB](licenses/pmlb.txt), [DSRRANS](licenses/dsrrans.txt), and [E2ESR](licenses/e2esr.txt). Llama models follow their upstream model license.
