"""Single-label metrics; rows=true labels, columns=predictions."""
from numbers import Integral


def classification_metrics(truth, predictions, class_names):
    n = len(class_names)
    if not n or len(set(class_names)) != n or not truth or len(truth) != len(predictions):
        raise ValueError('Nonempty equally-sized predictions and truth and unique classes required')
    confusion = [[0] * n for _ in range(n)]
    for actual, predicted in zip(truth, predictions):
        if any(not isinstance(x, Integral) or isinstance(x, bool) or not 0 <= x < n
               for x in (actual, predicted)):
            raise ValueError('Label out of range or not an integer')
        confusion[actual][predicted] += 1
    per_class = []
    for i, name in enumerate(class_names):
        support, predicted = sum(confusion[i]), sum(row[i] for row in confusion)
        tp = confusion[i][i]
        per_class.append({'class_name': name, 'class_index': i, 'support': support,
                          'accuracy_percent': 100 * tp / support if support else None,
                          'f1_percent': 200 * tp / (support + predicted) if support + predicted else 0.0})
    return {'samples': len(truth), 'oa_percent': 100 * sum(confusion[i][i] for i in range(n)) / len(truth),
            'macro_f1_percent': sum(c['f1_percent'] for c in per_class) / n,
            'per_class': per_class, 'confusion_matrix': confusion,
            'confusion_orientation': 'rows=true, columns=predicted', 'zero_division': 0}
