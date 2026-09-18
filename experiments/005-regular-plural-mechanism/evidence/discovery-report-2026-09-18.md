# Experiment 005 Report

- Run ID: `050405f9f4f7e629`
- Manifest sha256: `1c50c2e8bbf95f9aa4331ac57899f77c2a6be6b17d4cb2a5ac4ecc332c5e703c`
- Extension sha256: `1c6852547f1c2d8ecfe8a736599ba3670fac51e43c93f04057091c56ba179da5`
- Protocol/code commit at discover: `83d1ae435e05451666b1ca955d1939aa740789a1`
- Model: `EleutherAI/pythia-70m-deduped` @ `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`

## Phases

- `calibrate`: `not_started`
- `confirm`: `not_started`
- `discover`: `complete`
- `lock`: `not_started`
- `report`: `not_started`
- `revise`: `not_started`

- A0 contract test: passed
- A1 baseline: 240 cases compared to the screen (max gap 0.00e+00); development mean d_full 5.231; prompt-level gap 4.52e-03
- Hypothesis tree: `H1` from q1=0.816, q2=0.865, q3=0.987, q4=0.606, q5=0.953 (top head `L03.H04`)
- Encoding branch: `L00.MLP` with E = ['L00.MLP'] (E share 0.535, embedding share 0.009)

### A2 rankings

- p_t (all templates): L00.MLP, L03.H04, L05.MLP, L04.MLP, L02.MLP, L05.H07, L00.H05, L01.MLP
- p_t (coordinated): L03.H04, L04.MLP, L05.MLP, L05.H07, L01.MLP, L00.H05, L01.H05, L01.H03
- p_c (coordinated): L00.MLP, L02.MLP, L04.MLP, L00.H07, L03.H04, L00.H06, L01.H00, L01.H05

### A3 residual profile at p_c (coordinated)

- L0: recovery 1.000
- L1: recovery 0.965
- L2: recovery 0.938
- L3: recovery 0.956
- L4: recovery 0.125
- L5: recovery 0.125

### A4 attention to the cue from p_t

- L05.H07: 0.999
- L02.H07: 0.893
- L00.H05: 0.829
- L03.H04: 0.747
- L05.H02: 0.534
- L01.H05: 0.519

### A6 chain and path

- E alone: recovery 0.885, m_T 0.900, m_R 0.987
- residual patch at p_c: unfrozen 0.956; blocked fractions L00.H05=0.000, L03.H04=0.865, L05.H07=0.128, top3=0.986
- T alone: L00.H05=0.028, L03.H04=0.816, L05.H07=0.122; cumulative top1=0.816, top2=0.935, top3=0.953
- R alone: 0.606

## Mechanism version M1 — rejected

- Outcome: NO_COMPACT_MECHANISM 

## Execution ledger

- Executed prompt keys: 12
- Executed noun keys: 40
- Invalidated runs: 0
