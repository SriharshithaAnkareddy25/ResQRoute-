"""ResQRoute demo (Person C).  Run from the repo root:  streamlit run app/demo.py

Replays logged model outputs (no torch needed). Crop images are shown if data/crops/ is present.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import streamlit as st

from pipeline import ResQRoutePipeline, Device, ACTIONS  # noqa: E402
from common import CLASSES_3, OUTPUTS  # noqa: E402
import resqroute_sim as sim  # noqa: E402

st.set_page_config(page_title="ResQRoute", layout="wide")


@st.cache_resource(show_spinner="Loading models log and routing controller (first run trains it, ~30 s)...")
def pipe():
    return ResQRoutePipeline()


P = pipe()
tab_demo, tab_eval = st.tabs(["Route one building", "Evaluation results"])

with tab_demo:
    st.title("ResQRoute: risk- and resource-aware damage triage")
    st.caption("Small model first; the controller decides from model uncertainty, image quality, device state "
               "and the cost of a dangerous miss whether to accept, escalate, defer or ask a human.")
    side = st.sidebar
    side.header("Device state")
    preset = side.selectbox("Quick preset", ["(custom)"] + list(sim.PRESETS))
    d0 = Device()
    if preset != "(custom)":
        p = sim.PRESETS[preset]
        d0 = Device(p.budgets[0], p.mems[0], int(p.p_net > 0.5), round(sum(p.battery) / 2, 2), sum(p.queue) // 2)
    dev = Device(
        side.select_slider("Latency budget (ms)", [60, 100, 180, 300, 400], min(
            [60, 100, 180, 300, 400], key=lambda v: abs(v - d0.budget_ms))),
        side.select_slider("Free memory (MB)", [100, 150, 200, 250, 500], min(
            [100, 150, 200, 250, 500], key=lambda v: abs(v - d0.mem_free_mb))),
        int(side.checkbox("Network up", bool(d0.network_ok))),
        side.slider("Battery", 0.0, 1.0, float(d0.battery), 0.05),
        side.slider("Queue length", 0, 100, int(d0.queue_len)),
    )

    test = P.log[P.log.split == "test"]
    c1, c2, c3 = st.columns(3)
    event = c1.selectbox("Disaster", ["all"] + sorted(test.disaster_event.unique()))
    truth = c2.selectbox("True class", ["all"] + CLASSES_3)
    hard = c3.checkbox("Only crops the small model gets wrong", False)
    sub = test
    if event != "all":
        sub = sub[sub.disaster_event == event]
    if truth != "all":
        sub = sub[sub.damage_class_3 == truth]
    if hard:
        sub = sub[sub.mobilenet_correct == 0]
    ids = list(sub.image_id)
    if not ids:
        st.warning("No crop matches these filters.")
        st.stop()
    idx = st.number_input(f"Crop (1 to {len(ids)})", 1, len(ids), 1) - 1
    iid = ids[idx]
    out = P.route(iid, dev)
    row = out["row"]

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Input")
        pre, post = ROOT / row.pre_crop_path, ROOT / row.post_crop_path
        if post.exists():
            st.image([str(pre), str(post)], caption=["before", "after"], width=150)
        else:
            st.info("Crop images not on this machine (data/crops/ missing); showing model outputs only.")
        st.write(f"**Event:** {row.disaster_event}")
        st.write(f"**Ground truth:** {row.damage_class_3}")
        st.write(f"blur {row.blur:.0f} · contrast {row.contrast:.0f} · visible {row.visible_frac:.0%} · "
                 f"pre/post diff {row.prepost_diff:.0f}")

    with right:
        st.subheader("Cascade")
        for s in out["steps"]:
            with st.container(border=True):
                st.write(f"**Stage {s['stage']}: {s['model']}** · {s['time_ms']:.0f} ms · {s['mem_mb']:.0f} MB")
                st.bar_chart(pd.Series([float(v) for v in s["probs"]], index=CLASSES_3), height=140, horizontal=True)
        if out["action"] == 3:
            st.warning(f"Deferred. Queue priority P = U×R×D×A = {out['priority']:.3f}")
        elif out["action"] == 4:
            st.error("Flagged for human review")
        else:
            ok = out["final_pred"] == row.damage_class_3
            (st.success if ok else st.error)(
                f"Final: **{out['final_pred']}** ({'correct' if ok else 'wrong'}) · {out['immediate_ms']:.0f} ms total")
        st.caption(out["status"])

    st.subheader("Why this decision")
    cc1, cc2 = st.columns([1, 1])
    cc1.write("\n".join(f"- {r}" for r in out["reasons"]))
    cc2.bar_chart(pd.Series(out["controller_probs"]).reindex(ACTIONS).fillna(0), height=220, horizontal=True)

with tab_eval:
    st.header("Six-system comparison (test split, 852 crops × 8 simulated device states)")
    f = OUTPUTS / "eval_summary.csv"
    if f.exists():
        r = pd.read_csv(f, index_col=0)
        st.dataframe(r[["accuracy", "macro_F1", "severe_recall", "dangerous_FN_%", "lat_mean_ms", "lat_p95_ms",
                        "peak_mem_mb", "violations_%", "stage1_only_%", "stage2_effnet_%", "stage3_resnet_%",
                        "deferred_%", "human_review_%"]].round(3), width="stretch")
        for name, cap in [("eval_systems", "Six systems"), ("eval_scenarios", "By scenario (incl. degraded network)"),
                          ("eval_queue", "Deferred queue: priority vs FIFO vs random")]:
            img = OUTPUTS / "figures" / f"{name}.png"
            if img.exists():
                st.image(str(img), caption=cap)
    else:
        st.info("Run `python scripts/10_evaluate.py` and `python scripts/11_eval_figures.py` first.")
