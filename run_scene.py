"""Single-label scene experiments; --help is available without Torch."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Validate original images and write an immutable manifest')
    prepare.add_argument('--dataset', choices=['AID', 'NWPU-RESISC45', 'synthetic'], required=True)
    prepare.add_argument('--data-root', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--split-seed', type=int, default=42)
    prepare.add_argument('--val-split-seed', type=int, default=4242)
    train = sub.add_parser('train', help='Run an explicit smoke/calibration/refit phase')
    train.add_argument('--model', choices=['swin', 'gavit'], required=True)
    train.add_argument('--phase', choices=['smoke', 'calibrate', 'refit'], default='smoke')
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--pretrained-path')
    train.add_argument('--random-init', action='store_true', help='Synthetic/local smoke only; never formal')
    train.add_argument('--train-per-class', type=int, default=8)
    train.add_argument('--val-per-class', type=int, default=2)
    train.add_argument('--calibration', help='Completed matching calibration directory, required for refit')
    evaluate = sub.add_parser('evaluate', help='Evaluate a checkpoint with verified metadata')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--split', choices=['val', 'test'], default='test')
    for command in (train, evaluate):
        command.add_argument('--manifest', required=True)
        command.add_argument('--data-root', required=True)
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='cuda')
        command.add_argument('--batch-size', type=int, default=32)
        command.add_argument('--workers', type=int, default=2)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'prepare':
        from scene_classification.data import prepare_manifest
        manifest = prepare_manifest(args.pop('data_root'), args.pop('output'), **args)
        print(f'MANIFEST COMPLETE: {manifest["dataset"]}; counts={manifest["counts"]}', flush=True)
    elif command == 'train':
        from scene_classification.runner import RunConfig, run_training
        run_training(RunConfig(**args))
    else:
        from scene_classification.runner import evaluate_checkpoint
        args['manifest_path'] = args.pop('manifest')
        evaluate_checkpoint(**args)


if __name__ == '__main__':
    main()
