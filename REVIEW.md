# ResQRoute – review summary

**Goal:** route each satellite image of a disaster-hit building to the cheapest damage classifier that is safe
for it, taking into account uncertainty, image quality, device resources and the cost of a dangerous mistake.
The aim is accuracy close to the large models at much lower compute.

## What was built

| Stage | What | Where |
|---|---|---|
| Data | xBD (Maxar imagery, CC BY-NC-SA). 162,787 buildings indexed. A balanced subset of 6,000 pre/post crops (2,000 per class) from Hurricane Michael, Hurricane Matthew and the Santa Rosa wildfire. 70/15/15 split, grouped by satellite tile so there is no leakage. | `scripts/00–03` |
| Base models | MobileNetV3-Small, ResNet-18 and EfficientNet-B0, fine-tuned from ImageNet. Checkpoints selected by val macro-F1, with label smoothing so the probabilities are better calibrated for routing. | `scripts/04`, `models/` |
| Log | All 3 models run on every crop on the laptop CPU (1 thread): probabilities, correctness, latency, memory. | `outputs/model_log.csv` |
| Router features | Small-model confidence, entropy and margin; image quality (blur, brightness, contrast, building size, pre/post change); simulated device state (latency budget, free memory, network, battery, queue). | `scripts/07`, `08` |
| Controller | Gradient-boosted classifier choosing one of 5 actions: **accept small / escalate to EfficientNet / escalate to ResNet-18 / defer to a priority queue / human review**. Severe-damage cases weigh 3–5× more in training (asymmetric risk). Trained on val, tested on test. | `scripts/08_router.py` |
| Evaluation | 852 test crops × 8 device states = 6,816 decisions, against 5 baselines. | `outputs/router_results.csv`, `outputs/figures/` |

## Results (test split)

| Policy | Macro-F1 | Severe → "no-damage" | Avg CPU ms | Over budget |
|---|---:|---:|---:|---:|
| Always small (MobileNetV3-S) | 0.597 | 6.3% | 23 | 0.5% |
| Always EfficientNet-B0 | 0.610 | 4.2% | 126 | 37% |
| Always ResNet-18 | 0.610 | 9.5% | 235 | 78% |
| Fixed-confidence cascade (t = 0.74) | 0.612 | 7.4% | 113 | 33% |
| Resource-aware cascade, no risk term | 0.605 | 4.6% | 33 | 0.4% |
| **ResQRoute controller** | **0.631** | **4.6%** | **27** | **0.5%** |

- **Same accuracy as the big models at about 1/5 of the computation.** The controller averages 27 ms per image,
  against 126 ms for EfficientNet and 235 ms for ResNet-18.
- **It respects the device.** It exceeds the latency or memory budget in 0.5% of decisions. Always-EfficientNet
  exceeds it in 37%, and the fixed-confidence cascade in 33%.
- **It is safer than the cheap options.** Dangerous misses fall from 6.3% (always small) and 7.4% (fixed
  cascade) to 4.6%, close to the best model (4.2%).
- **It adapts to resources.** With a tight 60 ms budget it defers 13% of images. With a relaxed 400 ms budget
  it escalates instead and defers only 6%. Accuracy rises from 0.623 to 0.652 as the budget relaxes
  (`figures/actions_by_budget.png`).
- **Deferral prioritises danger.** In the deferred queue, ordered by priority P = U × R × D × A
  (uncertainty × risk × severity × area), 43.7% of the top half are truly severe, against 34.5% of the bottom half.
- **What the controller relies on:** network status, the small model's P(severe), top-2 margin, whether
  each model fits the budget, and image darkness and brightness. So it really does combine uncertainty,
  resources and image quality.

## Honest limitations (say these before you are asked)

1. **The +0.02 macro-F1 over the big models is not statistically significant.** A bootstrap over test crops gives
   a 95% CI of [−0.003, +0.044]. Most of the gain comes from the 4% of images sent to human review, which the
   evaluation assumes a human labels correctly. Without that credit the controller scores 0.607, the same as
   EfficientNet's 0.610. **The defensible claim is equal accuracy at much lower cost and safer than a
   confidence cascade, not higher accuracy.**
2. **Device states are simulated.** Budgets of 60/180/400 ms, free memory of 150/250/500 MB, 70% network
   availability and battery levels are drawn at random. The model latencies and memory are real measurements.
3. **Bigger is not better here.** ResNet-18 is the slowest model but not the most accurate, so the "medium"
   and "advanced" labels were re-ordered by measured cost.
4. **Test is harder than val** (macro-F1 0.61 vs 0.78), because half the severe test buildings are from Hurricane
   Matthew, whose imagery is blurry. All base models struggle to separate minor from severe damage.
5. **The controller has only 849 val crops to learn from** (×8 device states).

## Next steps

- Add the pre-disaster crop to the base models' input; minor vs severe damage is mostly a before/after
  difference.
- Use k-fold out-of-fold predictions to give the controller about 5,000 honest training rows.
- Build a Streamlit demo: upload an image, set the device state, and see the routing decision.

## How to reproduce the router results

```powershell
python scripts/07_quality_features.py   # ~1 min
python scripts/08_router.py             # ~1 min: trains the controller, prints the results table
python scripts/09_figures.py            # outputs/figures/*.png
```
