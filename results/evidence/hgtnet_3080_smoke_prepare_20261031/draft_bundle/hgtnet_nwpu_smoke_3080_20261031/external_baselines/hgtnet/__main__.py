"""Explicit local-only CLI. No implicit dataset job or online acquisition."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='HGTNet independent local CPU engineering checks')
    sub = parser.add_subparsers(dest='action', required=True)
    smoke = sub.add_parser('smoke', help='synthetic full-model CPU step; writes a new output directory')
    smoke.add_argument('--config', default=str(Path(__file__).resolve().parents[2] / 'configs/hgtnet_independent_v0.json'))
    smoke.add_argument('--output', required=True)
    smoke.add_argument('--weights')
    smoke.add_argument('--weights-sha256')
    smoke.add_argument('--cpu-threads', type=int, default=1)
    reload = sub.add_parser('verify-reload', help='fresh-process checkpoint verification; no training')
    reload.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.action == 'smoke' and ((args.weights is None) != (args.weights_sha256 is None)):
        parser.error('--weights and --weights-sha256 must be supplied together')
    from .smoke import run_smoke, verify_reload
    if args.action == 'smoke':
        return run_smoke(args.config, args.output, weights=args.weights,
                         weights_sha256=args.weights_sha256, cpu_threads=args.cpu_threads)
    return verify_reload(args.output)


if __name__ == '__main__':
    main()
