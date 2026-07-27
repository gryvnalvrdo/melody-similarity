import os, sys, argparse, pickle, csv, time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

try:
    from script.config import INDEX_DIR, SIMILARITY_THRESHOLD
except ImportError:
    try:
        from config import INDEX_DIR, SIMILARITY_THRESHOLD
    except ImportError:
        INDEX_DIR = os.path.join(BASE_DIR, "index")
        SIMILARITY_THRESHOLD = 0.60

def load_index(path: str) -> dict:
    print(f"[Eval] Memuat indeks dari: {path}")
    with open(path, "rb") as f:
        idx = pickle.load(f)
    n_versions = len(idx["song_metadata"])
    n_songs    = len(set(str(m["song_id"]) for m in idx["song_metadata"]))
    dim        = idx["song_embeddings"].shape[1]
    print(f"[Eval] Total song-version : {n_versions:,}")
    print(f"[Eval] Unique song_id     : {n_songs:,}")
    print(f"[Eval] Embedding dim      : {dim}")
    return idx

def build_test_split(song_meta: list, train_ratio: float = 0.8) -> dict:
    all_sids  = sorted(set(str(m["song_id"]) for m in song_meta))
    split_idx = int(len(all_sids) * train_ratio)
    test_sids = set(all_sids[split_idx:])

    sid_map = defaultdict(list)
    for i, m in enumerate(song_meta):
        sid = str(m["song_id"])
        if sid in test_sids:
            sid_map[sid].append(i)

    evaluable = {sid: idxs for sid, idxs in sid_map.items() if len(idxs) >= 2}
    print(f"[Eval] Total song_id      : {len(all_sids):,}")
    print(f"[Eval] Test song_id (20%) : {len(test_sids):,}")
    print(f"[Eval] Evaluable (>=2 ver): {len(evaluable):,}")
    return evaluable

def build_sim_matrix(embs: np.ndarray) -> np.ndarray:
    return (embs @ embs.T).astype(np.float32)

def collect_pairs(sim_mat, evaluable, sample, neg_per_query=50, seed=42):
    N   = sim_mat.shape[0]
    rng = np.random.default_rng(seed)

    all_queries = []
    for sid, idxs in evaluable.items():
        for q_idx in idxs:
            all_queries.append((q_idx, set(idxs) - {q_idx}))

    if sample and sample < len(all_queries):
        chosen = rng.choice(len(all_queries), size=sample, replace=False)
        all_queries = [all_queries[i] for i in chosen]

    pos_sims, neg_sims = [], []
    for q_idx, pos_set in tqdm(all_queries, desc="  [Eval] Mengumpulkan pasangan"):
        sims        = sim_mat[q_idx].copy()
        sims[q_idx] = -1.0
        for pi in pos_set:
            pos_sims.append(float(sims[pi]))
        neg_pool = [i for i in range(N) if i not in pos_set and i != q_idx]
        n_neg    = min(neg_per_query, len(neg_pool))
        for ni in rng.choice(neg_pool, size=n_neg, replace=False):
            neg_sims.append(float(sims[ni]))

    return pos_sims, neg_sims

def compute_metrics(pos_sims: list, neg_sims: list, threshold: float) -> dict:
    pos = np.array(pos_sims, dtype=np.float32)
    neg = np.array(neg_sims, dtype=np.float32)

    n_pos_total = float(len(pos))
    n_neg_total = float(len(neg))

    y_true  = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
    y_score = np.concatenate([pos, neg]).astype(np.float32)
    y_pred  = (y_score >= threshold).astype(np.float32)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy  = (tp + tn) / (tp + fp + tn + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    thresholds_roc = np.sort(np.unique(y_score))[::-1]
    tprs, fprs = [0.0], [0.0]
    op_tpr, op_fpr = None, None
    for thr in thresholds_roc:
        preds = (y_score >= thr).astype(np.float32)
        tpr = float(np.sum((preds == 1) & (y_true == 1))) / (n_pos_total + 1e-9)
        fpr = float(np.sum((preds == 1) & (y_true == 0))) / (n_neg_total + 1e-9)
        tprs.append(tpr)
        fprs.append(fpr)
        if op_tpr is None and abs(thr - threshold) < 0.005:
            op_tpr, op_fpr = tpr, fpr
    tprs.append(1.0); fprs.append(1.0)
    _trapz  = getattr(np, "trapezoid", np.trapz)
    roc_auc = abs(float(_trapz(np.array(tprs), np.array(fprs))))

    sorted_labels = y_true[np.argsort(y_score)[::-1]]
    n_retrieved, n_relevant, ap_sum = 0, 0, 0.0
    for label in sorted_labels:
        n_retrieved += 1
        if label == 1:
            n_relevant += 1
            ap_sum     += n_relevant / n_retrieved
    map_score = ap_sum / (n_pos_total + 1e-9)

    return {
        "n_pos_pairs":     len(pos_sims),
        "n_neg_pairs":     len(neg_sims),
        "imbalance_ratio": n_neg_total / (n_pos_total + 1e-9),
        "threshold":       threshold,
        "_roc_fprs":       fprs,
        "_roc_tprs":       tprs,
        "_op_fpr":         op_fpr,
        "_op_tpr":         op_tpr,
        "_y_true":         y_true,
        "_y_score":        y_score,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy":        accuracy,
        "precision":       precision,
        "recall":          recall,
        "f1_score":        f1,
        "balanced_accuracy": (tp / (n_pos_total + 1e-9) + tn / (n_neg_total + 1e-9)) / 2.0,
        "mcc":             _compute_mcc(tp, fp, tn, fn),
        "roc_auc":         roc_auc,
        "map":             map_score,
        "mean_pos_sim":    float(np.mean(pos)),
        "mean_neg_sim":    float(np.mean(neg)),
        "std_pos_sim":     float(np.std(pos)),
        "std_neg_sim":     float(np.std(neg)),
        "margin":          float(np.mean(pos)) - float(np.mean(neg)),
    }

def _compute_mcc(tp, fp, tn, fn):
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / (denom + 1e-9))

def find_optimal_threshold(res: dict) -> dict:
    y_true  = res["_y_true"]
    y_score = res["_y_score"]

    thresholds = np.sort(np.unique(y_score))[::-1]
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))

    best_j_thr, best_j, best_j_f1 = 0.5, -1.0, 0.0
    best_f1_thr, best_f1 = 0.5, 0.0

    for thr in thresholds:
        preds = (y_score >= thr).astype(np.float32)
        tp = float(np.sum((preds == 1) & (y_true == 1)))
        fp = float(np.sum((preds == 1) & (y_true == 0)))
        tn = float(np.sum((preds == 0) & (y_true == 0)))
        fn = float(np.sum((preds == 0) & (y_true == 1)))

        tpr = tp / (n_pos + 1e-9)
        fpr = fp / (n_neg + 1e-9)
        j   = tpr - fpr

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        if j > best_j:
            best_j, best_j_thr, best_j_f1 = j, float(thr), f1
        if f1 > best_f1:
            best_f1, best_f1_thr = f1, float(thr)

    def _metrics_at(thr):
        preds = (y_score >= thr).astype(np.float32)
        tp = float(np.sum((preds == 1) & (y_true == 1)))
        fp = float(np.sum((preds == 1) & (y_true == 0)))
        tn = float(np.sum((preds == 0) & (y_true == 0)))
        fn = float(np.sum((preds == 0) & (y_true == 1)))
        acc  = (tp + tn) / (tp + fp + tn + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        bacc = (tp / (n_pos + 1e-9) + tn / (n_neg + 1e-9)) / 2.0
        mcc  = _compute_mcc(tp, fp, tn, fn)
        return {"threshold": thr, "accuracy": acc, "precision": prec,
                "recall": rec, "f1_score": f1, "balanced_accuracy": bacc, "mcc": mcc}

    return {
        "youden": _metrics_at(best_j_thr),
        "f1_optimal": _metrics_at(best_f1_thr),
    }

def plot_roc_curve(res: dict, save_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Eval] matplotlib tidak tersedia, skip plot ROC.")
        return None

    fprs  = np.array(res["_roc_fprs"])
    tprs  = np.array(res["_roc_tprs"])
    auc   = res["roc_auc"]
    thr   = res["threshold"]
    op_fpr = res.get("_op_fpr")
    op_tpr = res.get("_op_tpr")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fprs, tprs, color="#2563EB", linewidth=2.5,
            label=f"Model AI Kita (Area = {auc:.4f})", zorder=3)
    ax.fill_between(fprs, tprs, alpha=0.12, color="#2563EB")

    ax.plot([0, 1], [0, 1], color="#9CA3AF", linewidth=1.2,
            linestyle="--", label="Model Acak (Area = 0.50)", zorder=2)

    if op_fpr is not None and op_tpr is not None:
        ax.scatter([op_fpr], [op_tpr], s=120, color="#EF4444", zorder=5)
        
        penjelasan = (
            f"Titik Sistem Saat Ini (Threshold {thr:.2f})\n"
            f"• Plagiat Terdeteksi: {op_tpr*100:.1f}%\n"
            f"• False Alarm: {op_fpr*100:.1f}%"
        )
        ax.annotate(penjelasan,
                    xy=(op_fpr, op_tpr),
                    xytext=(op_fpr + 0.05, op_tpr - 0.15),
                    fontsize=9, color="#EF4444",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#EF4444", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.5))

    grade = "Sangat Baik" if auc >= 0.90 else "Baik"
    ax.text(0.55, 0.20, f"Nilai ROC-AUC: {auc:.4f}\n(Kategori: {grade})",
            fontsize=12, color="#2563EB",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#2563EB", alpha=0.9))

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Persentase Lagu Berbeda yang Salah Dideteksi (False Alarm Rate)", fontsize=11)
    ax.set_ylabel("Persentase Plagiat yang Berhasil Dideteksi (Recall)", fontsize=11)
    ax.set_title("Kurva ROC: Kemampuan Model Mendeteksi Plagiat", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    if save_path is None:
        save_path = os.path.join(BASE_DIR, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return save_path

def print_results(res: dict, opt: dict = None):
    S  = "=" * 72
    thr = res["threshold"]

    acc_pct  = res['accuracy']          * 100
    bacc_pct = res.get('balanced_accuracy', 0) * 100
    pre_pct  = res['precision']         * 100
    rec_pct  = res['recall']            * 100
    f1_pct   = res['f1_score']          * 100
    mcc      = res.get('mcc', 0.0)
    auc      = res['roc_auc']
    map_v    = res['map']
    margin   = res.get('margin', 0.0)

    if   auc >= 0.90: grade = "Sangat Baik"
    elif auc >= 0.75: grade = "Baik"
    elif auc >= 0.60: grade = "Cukup"
    else:             grade = "Perlu Perbaikan"

    print(f"\n{S}")
    print("  EVALUASI MODEL KEMIRIPAN MELODI")
    print(f"  Threshold : {thr:.2f}  |  Mirip: {res['n_pos_pairs']:,}  |  Tdk Mirip: {res['n_neg_pairs']:,}  |  Rasio 1:{res['imbalance_ratio']:.0f}")
    print(S)
    print()
    print(f"  {'Metrik':<42} {'Nilai':>10}")
    print(f"  {'-'*53}")
    print(f"  {'Accuracy (threshold sistem)':<42} {acc_pct:>9.2f}%")
    print(f"  {'Balanced Accuracy (tidak terpengaruh imbalance)':<42} {bacc_pct:>9.2f}%")
    print(f"  {'Precision':<42} {pre_pct:>9.2f}%")
    print(f"  {'Recall':<42} {rec_pct:>9.2f}%")
    print(f"  {'F1-Score':<42} {f1_pct:>9.2f}%")
    print(f"  {'MCC (Matthews Corr. Coeff.)':<42} {mcc:>10.4f}")
    print(f"  {'ROC-AUC':<42} {auc:>10.4f}")
    print(f"  {'MAP (Mean Average Precision)':<42} {map_v:>10.4f}")
    print(f"  {'Margin (mean_pos - mean_neg)':<42} {margin:>10.4f}")
    print()
    print(f"  Confusion Matrix:  TP={res['tp']:,.0f}  FP={res['fp']:,.0f}  TN={res['tn']:,.0f}  FN={res['fn']:,.0f}")
    print()
    print(f"  Kesimpulan: ROC-AUC = {auc:.4f}  ->  {grade}")

    if opt:
        youden    = opt["youden"]
        f1_opt    = opt["f1_optimal"]
        print()
        print(f"  {'─'*72}")
        print(f"  PERBANDINGAN THRESHOLD: SISTEM vs OPTIMAL")
        print(f"  {'─'*72}")
        col_sys = "Sistem (thr=" + f"{thr:.2f}" + ")"
        col_you = "Youden J (thr=" + f"{youden['threshold']:.2f}" + ")"
        col_f1  = "F1-Max (thr=" + f"{f1_opt['threshold']:.2f}" + ")"
        print(f"  {'Metrik':<28} {col_sys:<20} {col_you:<20} {col_f1:<20}")
        print(f"  {'-'*88}")

        def _pct(v): return f"{v*100:.2f}%"
        rows = [
            ("Accuracy",         res['accuracy'],          youden['accuracy'],          f1_opt['accuracy']),
            ("Balanced Accuracy", res.get('balanced_accuracy',0), youden['balanced_accuracy'], f1_opt['balanced_accuracy']),
            ("Precision",         res['precision'],          youden['precision'],          f1_opt['precision']),
            ("Recall",            res['recall'],             youden['recall'],             f1_opt['recall']),
            ("F1-Score",          res['f1_score'],           youden['f1_score'],           f1_opt['f1_score']),
            ("MCC",               res.get('mcc',0),          youden['mcc'],               f1_opt['mcc']),
        ]
        for name, sys_v, you_v, f1_v in rows:
            if name == "MCC":
                print(f"  {name:<28} {sys_v:>18.4f}   {you_v:>18.4f}   {f1_v:>18.4f}")
            else:
                print(f"  {name:<28} {_pct(sys_v):>18}   {_pct(you_v):>18}   {_pct(f1_v):>18}")
        print()
        print(f"  Catatan:")
        print(f"    - Youden's J    : threshold yang memaksimalkan TPR - FPR")
        print(f"    - F1-Max        : threshold yang memaksimalkan F1-Score")
        print(f"    - Threshold sistem SEBAIKNYA = F1-Max / Youden's J agar")
        print(f"      Accuracy dan ROC-AUC tidak berbeda jauh")
        if abs(youden['threshold'] - thr) > 0.05:
            print(f"  ⚠  Threshold sistem ({thr:.2f}) jauh dari optimal ({youden['threshold']:.2f}).")
            print(f"     Gunakan --update-config untuk update config.py otomatis.")
        else:
            print(f"  ✓  Threshold sistem sudah mendekati optimal.")
        print(f"  {'─'*72}")

    print(S)
    print()

def save_csv(res: dict, path: str):
    rows = [{"metric": k, "value": v}
            for k, v in res.items()
            if isinstance(v, (int, float))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)
    print(f"[Eval] Hasil disimpan ke: {path}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi model kemiripan melodi: Accuracy, Precision, Recall, F1, ROC-AUC, MAP."
    )
    parser.add_argument("--index",          type=str,   default=os.path.join(INDEX_DIR, "embedding_index.pkl"))
    parser.add_argument("--threshold",      type=float, default=SIMILARITY_THRESHOLD,
                        help=f"Threshold deteksi sistem (default dari config: {SIMILARITY_THRESHOLD})")
    parser.add_argument("--train-ratio",    type=float, default=0.8)
    parser.add_argument("--sample",         type=int,   default=None)
    parser.add_argument("--neg-per-query",  type=int,   default=50)
    parser.add_argument("--save-csv",       type=str,   default=None)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--plot",           action="store_true", default=True,
                        help="Generate kurva ROC sebagai PNG (default: aktif)")
    parser.add_argument("--no-plot",        action="store_true",
                        help="Nonaktifkan plot ROC")
    parser.add_argument("--plot-path",      type=str,   default=None,
                        help="Path simpan PNG (default: roc_curve.png di root project)")
    parser.add_argument("--update-config",  action="store_true",
                        help="Tulis threshold optimal (F1-max) ke config.py secara otomatis")
    args = parser.parse_args()
    do_plot      = args.plot and not args.no_plot

    if not os.path.exists(args.index):
        print(f"\n[Error] Index tidak ditemukan: {args.index}")
        print("  Jalankan build_index.py terlebih dahulu.")
        sys.exit(1)

    index     = load_index(args.index)
    song_embs = np.array(index["song_embeddings"], dtype=np.float32)
    song_meta = index["song_metadata"]

    print()
    evaluable = build_test_split(song_meta, train_ratio=args.train_ratio)
    if not evaluable:
        print("[Error] Tidak ada lagu evaluable di test split.")
        sys.exit(1)

    total_q = sum(len(v) for v in evaluable.values())
    print(f"[Eval] Total query               : {total_q:,}")

    print("\n[Eval] Menghitung similarity matrix ...", end=" ", flush=True)
    t0      = time.time()
    sim_mat = build_sim_matrix(song_embs)
    np.fill_diagonal(sim_mat, -1.0)
    print(f"selesai ({time.time()-t0:.1f} s, shape {sim_mat.shape})")

    print()
    pos_sims, neg_sims = collect_pairs(
        sim_mat, evaluable,
        sample=args.sample,
        neg_per_query=args.neg_per_query,
        seed=args.seed,
    )

    print("\n[Eval] Menghitung metrik evaluasi ...")
    res = compute_metrics(pos_sims, neg_sims, threshold=args.threshold)

    print("[Eval] Mencari threshold optimal ...")
    opt = find_optimal_threshold(res)

    print_results(res, opt=opt)

    if args.update_config:
        new_thr = opt["f1_optimal"]["threshold"]
        _update_config_threshold(new_thr)

    if do_plot:
        plot_roc_curve(res, save_path=args.plot_path)

    if args.save_csv:
        save_csv(res, args.save_csv)

def _update_config_threshold(new_thr: float):
    try:
        from script.config import __file__ as cfg_path
    except ImportError:
        try:
            from config import __file__ as cfg_path
        except ImportError:
            cfg_path = os.path.join(BASE_DIR, "script", "config.py")

    cfg_path = os.path.abspath(cfg_path)
    if not os.path.exists(cfg_path):
        print(f"[Eval] ⚠ config.py tidak ditemukan di: {cfg_path}")
        return

    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    new_content = re.sub(
        r"SIMILARITY_THRESHOLD\s*=\s*[0-9.]+",
        f"SIMILARITY_THRESHOLD ={new_thr:.4f} ",
        content
    )
    if new_content == content:
        print(f"[Eval] ⚠ Tidak bisa menemukan baris SIMILARITY_THRESHOLD di config.py")
        return

    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[Eval] ✓ config.py diupdate: SIMILARITY_THRESHOLD = {new_thr:.4f}")
    print(f"[Eval]   Path: {cfg_path}")

if __name__ == "__main__":
    main()
