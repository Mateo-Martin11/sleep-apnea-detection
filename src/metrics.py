"""
Métriques officielles du Challenge ENS #45 — Détection d'apnées du sommeil.

La métrique principale est le F1-IoU : un événement prédit est un vrai positif (TP)
si son IoU 1D avec au moins un événement de référence (non encore apparié) dépasse
le seuil θ = 0.3.
"""
import numpy as np


def extract_events(mask: np.ndarray) -> list:
    """
    Convertit un masque binaire 1D en liste d'événements (start, end).

    Utilise le padding [0, ...mask..., 0] + np.diff pour détecter les
    transitions montantes (début d'apnée) et descendantes (fin d'apnée).

    Paramètres
    ----------
    mask : np.ndarray, shape (T,) — masque binaire 0/1

    Retourne
    --------
    list of (int, int) — liste de (start_idx, end_idx) [borne sup exclue]
    """
    mask_padded = np.concatenate([[0], np.asarray(mask, dtype=int), [0]])
    diff        = np.diff(mask_padded)
    starts      = np.where(diff ==  1)[0]
    ends        = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def compute_iou_1d(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """IoU entre deux intervalles 1D [a_start, a_end) et [b_start, b_end)."""
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def compute_f1_iou(y_true: np.ndarray, y_pred: np.ndarray,
                   iou_threshold: float = 0.3) -> float:
    """
    F1-score basé sur le chevauchement IoU entre événements (métrique officielle).

    Un événement prédit est un TP si son IoU avec un événement de référence
    non encore apparié est >= iou_threshold.

    Paramètres
    ----------
    y_true        : np.ndarray (T,) — masque de référence
    y_pred        : np.ndarray (T,) — masque prédit (déjà binarisé)
    iou_threshold : float (0.3 par défaut — seuil du challenge)

    Retourne
    --------
    f1 : float dans [0, 1]
    """
    true_events = extract_events(y_true)
    pred_events = extract_events(y_pred)

    if len(true_events) == 0 and len(pred_events) == 0:
        return 1.0
    if len(true_events) == 0 or len(pred_events) == 0:
        return 0.0

    matched_true = set()
    tp = 0
    for ps, pe in pred_events:
        best_iou, best_j = 0.0, -1
        for j, (ts, te) in enumerate(true_events):
            if j in matched_true:
                continue
            iou = compute_iou_1d(ps, pe, ts, te)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold:
            tp += 1
            matched_true.add(best_j)

    fp    = len(pred_events) - tp
    fn    = len(true_events)  - tp
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def compute_batch_f1(y_true_batch: np.ndarray, y_pred_batch: np.ndarray,
                     threshold: float = 0.5) -> float:
    """
    Calcule le F1-IoU moyen sur un batch de prédictions.

    Paramètres
    ----------
    y_true_batch : np.ndarray, shape (N, 90) — labels de référence
    y_pred_batch : np.ndarray, shape (N, 90) — probabilités [0,1] prédites
    threshold    : float — seuil de binarisation (défaut 0.5, à calibrer)

    Retourne
    --------
    mean_f1 : float — F1-IoU moyen sur les N samples du batch
    """
    y_pred_bin = (y_pred_batch >= threshold).astype(int)
    f1_scores  = [
        compute_f1_iou(y_true_batch[i], y_pred_bin[i])
        for i in range(len(y_true_batch))
    ]
    return float(np.mean(f1_scores))
