from rapidfuzz import fuzz

class PrecisionTracker:
    def __init__(self, method="fuzzy", threshold=80):
        self.method = method
        self.threshold = threshold
        self.total_matched = 0
        self.total_preds = 0

    def is_match(self, gt, pred):
        gt = gt.strip().lower()
        pred = pred.strip().lower()
        if self.method == "exact":
            return gt == pred
        elif self.method == "fuzzy":
            return fuzz.ratio(gt, pred) >= self.threshold
        else:
            raise ValueError("Method must be 'exact' or 'fuzzy'")

    def update(self, gts, preds):
        used = set()
        matched = 0
        for pred in preds:
            for i, gt in enumerate(gts):
                if i not in used and self.is_match(gt, pred):
                    matched += 1
                    used.add(i)
                    break
        self.total_matched += matched
        self.total_preds += len(preds)

    def get_precision(self):
        return self.total_matched / self.total_preds if self.total_preds else 0.0
