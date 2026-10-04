"""Generate a report draft from measured results and official evaluator output."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from experiments import PREDICTIONS, MECHANISMS


def observation(result, reference, noise):
    cfg, summary = result["cfg"], result["summary"]
    if summary["diverged"]:
        return "Diverged: " + summary["divergence_reason"] + " This run is excluded from final selection."
    delta = summary["val_macro_f1"] - reference["summary"]["val_macro_f1"]
    exceeded = abs(delta) > noise
    message = (f"Best-checkpoint F1={summary['val_macro_f1']:.6f}, "
               f"accuracy={summary['val_acc']:.6f}, best epoch={summary['best_epoch']}; "
               f"F1 difference versus {reference['cfg']['exp_id']}={delta:+.6f}. "
               f"Absolute difference {'exceeds' if exceeded else 'does not exceed'} "
               f"the baseline 2-sigma reference ({noise:.6f}).")
    if cfg["group"] == "clipping":
        fractions = result["history"].get("clip_fraction", [])
        message += f" Mean epoch clipping fraction={np.mean(fractions):.4f}."
    elif cfg["group"] == "dropout":
        gap = summary["final_val_loss"] - summary["final_train_loss"]
        basegap = reference["summary"]["final_val_loss"] - reference["summary"]["final_train_loss"]
        message += f" Final loss gap={gap:.6f} versus reference gap={basegap:.6f}."
    elif cfg["group"] == "amp":
        ratio = summary["time_per_epoch_s"] / reference["summary"]["time_per_epoch_s"]
        message += f" Epoch time ratio to FP32={ratio:.3f}; peak allocated GPU MiB={summary['peak_mem_MB']:.2f}."
    elif cfg["group"] == "init":
        message += f" Initial activation standard deviations={summary['activation_std']}."
    elif cfg["group"] == "hparam":
        message += f" Batch={cfg['batch']}; time per epoch={summary['time_per_epoch_s']:.3f}s."
    return message + " " + MECHANISMS[cfg["group"]]


def write_report(path, results, baseline, final, seed_stats, official_scores,
                 diagnostics, environment, student_id):
    noise = seed_stats["val_macro_f1"]["noise_2sigma"]
    base_score = official_scores[baseline["cfg"]["exp_id"]]
    final_score = official_scores[final["cfg"]["exp_id"]]
    d = diagnostics
    lines = [f"# Lab Day 1 report - {student_id}", "",
             "Generated from measured runs. Review the interpretations and add your name before submission.", "",
             "## 1. Setup", "",
             f"Python {environment['python']}; PyTorch {environment['torch']}; GPU: {environment.get('gpu') or 'CPU'}.",
             f"Official 80/20 train/eval metadata; stratified validation 20% of training rows, seed 42. "
             f"Training rows={d['n_train']}; validation rows={d['n_val']}. Only the first ten columns are standardized using training statistics.",
             f"M-base: 54-256-128-7, {d['parameters']} parameters. Baseline `{baseline['cfg']['exp_id']}`: "
             f"SGD momentum 0.9, CE, He, lr={baseline['cfg']['lr']}, batch=512, "
             f"epochs={baseline['cfg']['epochs']}, FP32, no dropout or clipping. "
             f"Majority-class validation accuracy={d['majority_val_acc']:.6f}.",
             "Every completed configuration uses the same epoch budget and validation split. "
             "Checkpoint metrics come from each run's minimum-validation-loss epoch. "
             "Final configuration ranking uses validation macro-F1 at those checkpoints.", "",
             "## 2. Initial checks and seed noise", "",
             f"Logits shape: {d['logits_shape']}; initial CE={d['step0_loss']:.6f}; ln(7)={d['ln7']:.6f}. "
             "Random logits need not be uniform, so the loss need not equal ln(7) exactly. "
             f"Every parameter had a finite nonzero gradient. The 20-example check reached loss={d['tiny_loss']:.6f}, "
             f"accuracy={d['tiny_acc']:.6f} after {d['tiny_steps']} updates.",
             f"Baseline validation accuracy: {seed_stats['val_acc']['mean']:.6f} +/- {seed_stats['val_acc']['std']:.6f}. "
             f"Baseline validation macro-F1: {seed_stats['val_macro_f1']['mean']:.6f} +/- "
             f"{seed_stats['val_macro_f1']['std']:.6f}; 2-sigma={noise:.6f}. "
             "This sample standard deviation over three seeds is a rough noise reference, not a statistical significance test.",
             "Diagnostics are recorded in the workbook Legend sheet; baseline seed IDs are in Seeds.", "",
             "![Tiny-batch diagnostic](figures/diagnostic_overfit.png)", "",
             "## 3. Results by topic", "",
             "| Experiment ID | Group | Validation accuracy | Validation macro-F1 | Best epoch | Diverged |",
             "|---|---|---:|---:|---:|---|"]
    for r in results:
        s = r["summary"]
        numbers = [f"{s[k]:.6f}" if s.get(k) is not None else "unavailable" for k in ("val_acc", "val_macro_f1")]
        lines.append(f"| {r['cfg']['exp_id']} | {r['cfg']['group']} | {numbers[0]} | {numbers[1]} | {s['best_epoch']} | {s['diverged']} |")
    for group in PREDICTIONS:
        members = [r for r in results if r["cfg"]["group"] == group]
        if not members:
            continue
        lines.extend(["", f"### {group}", "", "Prediction: " + PREDICTIONS[group]])
        if group == "baseline":
            lines.append("Learning rate was selected from 0.01, 0.03 and 0.1 using validation; only the selected rate's seeds enter the noise estimate.")
        for r in members:
            if group == "baseline" and r is not baseline:
                continue
            lines.append(f"`{r['cfg']['exp_id']}`: " + r["summary"].get("observation", "See workbook notes."))
        lines.extend(["", f"![{group} comparison](figures/compare_{group}.png)"])
    if not any(r["cfg"]["group"] == "amp" for r in results):
        lines.extend(["", "Mixed precision was skipped because the runtime has no CUDA GPU."])
    elif not any(r["cfg"]["precision"] == "bf16" for r in results):
        lines.extend(["", "BF16 was skipped because the GPU does not support it; FP16 remains the tested precision comparison."])
    lines.extend(["", "## 4. Final evaluation", "",
                  "Configuration and seed were frozen in selection.json before reading eval scores. Eval was used only for the designated baseline and final configuration.", "",
                  "| Configuration | Experiment ID | Seed | Eval accuracy | Eval macro-F1 |",
                  "|---|---|---:|---:|---:|"])
    for name, r, scores in (("Baseline", baseline, base_score), ("Final", final, final_score)):
        lines.append(f"| {name} | {r['cfg']['exp_id']} | {r['cfg']['seed']} | {scores['accuracy']:.6f} | {scores['macro_f1']:.6f} |")
    lines.extend(["", f"Final minus baseline eval F1={final_score['macro_f1'] - base_score['macro_f1']:+.6f}. "
                  "Eval seed uncertainty was not measured; validation seed noise cannot establish eval significance.", "",
                  "### Per-class errors", "",
                  "| Class | Support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"])
    for item in final_score["per_class"]:
        lines.append(f"| {item['cls']} | {item['support']} | {item['precision']:.6f} | {item['recall']:.6f} | {item['f1']:.6f} |")
    worst = min(final_score["per_class"], key=lambda item: item["f1"])
    cm = np.array(final_score["confusion_matrix"])
    errors = cm[worst["cls"]].copy()
    errors[worst["cls"]] = 0
    confusable = f"class {int(errors.argmax())}" if errors.max() else "no other class"
    lines.extend(["", f"The lowest F1 is class {worst['cls']} ({worst['f1']:.6f}), most often confused with {confusable}. "
                  "Class imbalance and overlapping features are plausible causes. Feature analysis or a controlled class-weight experiment would test these hypotheses.",
                  "", "![Eval confusion matrix](figures/eval_confusion.png)", "",
                  "## 5. Guiding questions", "",
                  "Optimizer conclusions apply to the tested rates, epoch budget and seeds. Dropout is useful only when the measured gap and validation scores support it. "
                  "Clipping limits update spikes; its activation fraction and matched high-rate curves are the evidence. "
                  "The precision time ratios above determine whether AMP was faster. Zero initialization blocks hidden ReLU gradients; He and Xavier use different variances.", "",
                  "If loss does not decrease after 2,000 updates, first inspect feature scale, label range and initial CE. "
                  "Second, overfit 20 samples with regularization disabled to test the training pipeline. "
                  "Third, inspect each parameter's gradient and verify finite updates, zero_grad, learning rate and AMP unscaling. "
                  "These checks distinguish data problems, disconnected or dead activations, and update-loop problems before changing architecture.", "",
                  "## 6. Limitations and unexpected outcomes", "",
                  "Only baseline noise is measured across three seeds; most topic trials use one seed. "
                  "The 2-sigma rule is descriptive and multiple configuration searches can overfit validation. "
                  "MSE and CE require different learning-rate tuning; here only loss changes, so this is a controlled comparison at the baseline rate. "
                  "Batch sizes change update count at fixed epochs. GPU timings include evaluation and checkpoint copies and vary with runtime load. "
                  "Memory is peak allocated MiB, including resident data, rather than total GPU reservation. "
                  "A 10x-rate clipping stress test may remain stable or fail despite clipping; neither outcome should be misreported. "
                  "Additional repeated topic seeds and broader independent learning-rate searches would strengthen conclusions.", "",
                  "## 7. Files and traceability", "",
                  "experiments.xlsx preserves the four official sheets; results/<exp_id>.json stores measured histories; "
                  "figures/<exp_id>.png records every run. predictions_eval.csv and eval_result.json use the official scorer. "
                  "The code folder contains the executable notebook and all implementation modules. "
                  "Model checkpoints and data are excluded from the submission.", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
