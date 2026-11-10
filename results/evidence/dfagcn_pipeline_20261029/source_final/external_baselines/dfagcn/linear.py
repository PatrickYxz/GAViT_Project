"""Actual LIBLINEAR train/predict CLI boundary for the independent recipe.

No scaling is learned here. The caller supplies frozen/fitted fused features and
uses only train/validation for selecting C. Solver and bias are declared recipe
choices, not a claim about unspecified settings in the DFAGCN paper.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import subprocess
import time

import numpy as np


SCHEMA = 'dfagcn-liblinear-cli-v1'
TERMINATION_TOLERANCE = 0.01


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _features(values):
    array = np.asarray(values)
    if array.dtype.kind != 'f':
        raise TypeError('features must have a real floating dtype')
    if array.ndim != 2 or not all(array.shape):
        raise ValueError('features must have nonempty two-dimensional shape')
    if not np.isfinite(array).all():
        raise ValueError('features must be finite')
    return array


def _labels(values, count):
    labels = np.asarray(values)
    if labels.dtype.kind not in 'iu':
        raise TypeError('labels must be integer class indices')
    if labels.ndim != 1 or len(labels) != count or (labels < 0).any():
        raise ValueError('labels must match the sample count and be nonnegative')
    unique = sorted(int(x) for x in np.unique(labels))
    if len(unique) < 2 or unique != list(range(len(unique))):
        raise ValueError('training classes must be contiguous indices from zero, with at least two classes')
    return labels, unique


def _real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real scalar')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return value


def _binary(path):
    binary = Path(path).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if not os.access(binary, os.X_OK):
        raise ValueError(f'not an executable binary: {binary}')
    return binary


def _write_json(path, record):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _write_svm(path, features, labels):
    """Retain the last index even when zero, so nr_feature is unambiguous."""
    width = features.shape[1]
    with Path(path).open('x', encoding='ascii') as handle:
        for row, label in zip(features, labels):
            indices = np.flatnonzero(row)
            if len(indices) == 0 or indices[-1] != width - 1:
                indices = np.append(indices, width - 1)
            items = [f'{int(index) + 1}:{float(row[index]):.17g}' for index in indices]
            handle.write(f'{int(label)} ' + ' '.join(items) + '\n')


def _run(arguments, output):
    started = time.monotonic()
    with (output / 'stdout.log').open('x') as stdout, (output / 'stderr.log').open('x') as stderr:
        run = subprocess.run(arguments, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.monotonic() - started
    if run.returncode != 0:
        _write_json(output / 'failure.json', {'arguments': arguments, 'return_code': run.returncode,
                                             'elapsed_seconds': elapsed})
        raise RuntimeError(f'LIBLINEAR exited {run.returncode}; output preserved at {output}')
    return elapsed


def _header(model):
    fields = {}
    with Path(model).open(encoding='ascii') as handle:
        for line in handle:
            if line.strip() == 'w':
                break
            parts = line.split()
            if not parts or parts[0] in fields:
                raise ValueError('invalid model header')
            fields[parts[0]] = parts[1:]
        else:
            raise ValueError('model has no weight section')
    try:
        nr_feature = int(fields['nr_feature'][0])
        nr_class = int(fields['nr_class'][0])
        classes = [int(x) for x in fields['label']]
        bias = float(fields['bias'][0])
        solver = fields['solver_type'][0]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError('invalid classification model header') from exc
    if nr_feature < 1 or nr_class < 2 or len(classes) != nr_class or not math.isfinite(bias):
        raise ValueError('invalid classification model dimensions')
    if sorted(classes) != list(range(nr_class)):
        raise ValueError('model labels are not contiguous class indices')
    return {'nr_feature': nr_feature, 'class_labels': sorted(classes), 'bias': bias, 'solver_type': solver}


def train_linear(features, labels, output_dir, *, train_binary, C, solver=2, bias=1.0):
    """Fit actual LIBLINEAR and return its model path, with sibling metadata.

    C is not selected internally. The fixed 0.01 termination tolerance is an
    explicit independent choice; solver 2 is L2-regularized L2-loss SVC primal.
    Bias is regularized, as no -R option is passed.
    """
    features = _features(features)
    labels, classes = _labels(labels, len(features))
    C = _real(C, 'C', positive=True)
    bias = _real(bias, 'bias')
    if type(solver) is not int or solver not in range(8):
        raise ValueError('solver must be an integer classification solver from zero through seven')
    binary = _binary(train_binary)
    binary_digest = _sha(binary)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    training_file, model = output / 'train.svm', output / 'model.txt'
    _write_svm(training_file, features, labels)
    arguments = [str(binary), '-s', str(solver), '-c', format(C, '.17g'), '-B', format(bias, '.17g'),
                 '-e', str(TERMINATION_TOLERANCE), str(training_file), str(model)]
    elapsed = _run(arguments, output)
    if not model.is_file():
        raise RuntimeError(f'LIBLINEAR did not create a model; output preserved at {output}')
    header = _header(model)
    if header['nr_feature'] != features.shape[1] or header['class_labels'] != classes or header['bias'] != bias:
        raise ValueError('trained model header does not match training identity')
    metadata = {'schema': SCHEMA, 'operation': 'train', 'arguments': arguments,
                'binary_sha256': binary_digest, 'model_file': model.name, 'model_sha256': _sha(model),
                'training_file_sha256': _sha(training_file), 'training_samples': len(features),
                'solver': solver, 'C': C, 'bias': bias, 'termination_tolerance': TERMINATION_TOLERANCE,
                'bias_regularized': bias >= 0, 'return_code': 0, 'elapsed_seconds': elapsed, **header}
    _write_json(output / 'metadata.json', metadata)
    return model


def predict_linear(features, model_path, output_dir, *, predict_binary):
    """Predict class indices. CLI stdout accuracy uses dummy labels and is ignored."""
    features = _features(features)
    model = Path(model_path).resolve()
    metadata = json.loads((model.parent / 'metadata.json').read_text(encoding='utf-8'))
    if (metadata.get('schema') != SCHEMA or metadata.get('operation') != 'train'
            or metadata.get('model_file') != model.name or metadata.get('model_sha256') != _sha(model)):
        raise ValueError('model identity or SHA does not match saved metadata')
    header = _header(model)
    if any(metadata.get(key) != value for key, value in header.items()):
        raise ValueError('model header does not match saved identity')
    if features.shape[1] != header['nr_feature']:
        raise ValueError('prediction feature width differs from trained model')
    binary = _binary(predict_binary)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_file, prediction_file = output / 'predict.svm', output / 'predictions.txt'
    _write_svm(input_file, features, np.zeros(len(features), dtype=np.int64))
    arguments = [str(binary), str(input_file), str(model), str(prediction_file)]
    elapsed = _run(arguments, output)
    try:
        values = [float(x) for x in prediction_file.read_text(encoding='ascii').splitlines()]
        if len(values) != len(features) or any(not math.isfinite(x) or not x.is_integer() for x in values):
            raise ValueError('invalid prediction labels')
        predictions = [int(x) for x in values]
        if any(x not in header['class_labels'] for x in predictions):
            raise ValueError('prediction label outside trained class mapping')
    except (OSError, ValueError) as exc:
        raise ValueError('LIBLINEAR prediction output is invalid') from exc
    _write_json(output / 'metadata.json', {'schema': SCHEMA, 'operation': 'predict',
                'arguments': arguments, 'binary_sha256': _sha(binary), 'model_sha256': metadata['model_sha256'],
                'input_file_sha256': _sha(input_file), 'prediction_file_sha256': _sha(prediction_file),
                'samples': len(features), 'nr_feature': features.shape[1], 'return_code': 0,
                'elapsed_seconds': elapsed, 'stdout_accuracy_uses_dummy_labels': True})
    return predictions
