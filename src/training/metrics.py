"""Event-based F1 metric matching the official challenge evaluation."""
import numpy as np


def extract_events(mask: np.ndarray) -> list[tuple[int, int]]:
    """Extract (start, end) event pairs from a binary mask (1D)."""
    padded = np.concatenate([[0], np.asarray(mask, dtype=int), [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def compute_iou(e1: tuple[int, int], e2: tuple[int, int]) -> float:
    """IoU between two (start, end) intervals."""
    s1, f1 = e1
    s2, f2 = e2
    inter = max(0, min(f1, f2) - max(s1, s2))
    union = (f1 - s1) + (f2 - s2) - inter
    return inter / union if union > 0 else 0.0


def compute_f1_events(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[float, float, float]:
    """Event-based F1 score.

    Args:
        y_true: (N, 90) binary ground-truth masks
        y_pred: (N, 90) binary predicted masks
        iou_threshold: minimum IoU to count a match

    Returns:
        f1, precision, recall
    """
    tp, fp, fn = 0, 0, 0

    for i in range(len(y_true)):
        gt_events = extract_events(y_true[i])
        pred_events = extract_events(y_pred[i])

        matched_gt: set[int] = set()
        for pred_ev in pred_events:
            matched = False
            for j, gt_ev in enumerate(gt_events):
                if j not in matched_gt and compute_iou(pred_ev, gt_ev) >= iou_threshold:
                    matched = True
                    matched_gt.add(j)
                    break
            if matched:
                tp += 1
            else:
                fp += 1

        fn += len(gt_events) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def threshold_predictions(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probability array to binary predictions."""
    return (probs >= threshold).astype(int)


def filter_short_events(mask: np.ndarray, min_duration: int) -> np.ndarray:
    """Remove predicted events shorter than min_duration seconds.

    Args:
        mask: (90,) binary mask
        min_duration: minimum event length in seconds to keep
    Returns:
        filtered mask (90,)
    """
    result = mask.copy()
    events = extract_events(mask)
    for start, end in events:
        if (end - start) < min_duration:
            result[start:end] = 0
    return result


def apply_postprocessing(y_pred: np.ndarray, min_duration: int) -> np.ndarray:
    """Apply short-event filtering to a batch of predictions.

    Args:
        y_pred: (N, 90) binary predictions
        min_duration: minimum event duration in seconds to keep
    Returns:
        (N, 90) filtered predictions
    """
    return np.stack([filter_short_events(row, min_duration) for row in y_pred])


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray = None,
) -> tuple[float, float]:
    """Grid search for the threshold maximizing event F1.

    Returns:
        best_threshold, best_f1
    """
    if thresholds is None:
        thresholds = np.arange(0.2, 0.8, 0.05)

    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        y_pred = threshold_predictions(probs, thr)
        f1, _, _ = compute_f1_events(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1
