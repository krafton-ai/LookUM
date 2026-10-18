"""Launch MDLM or LoopMDM training."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = {'lm1b': ('lm1b-wrap', 128), 'owt': ('openwebtext-split', 1024)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train'])
    parser.add_argument('--dataset', choices=DATA, default='owt')
    parser.add_argument('--model', choices=['baseline', 'loop'], default='loop')
    parser.add_argument('--max-steps', type=int, default=1_000_000)
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8, help='Per GPU')
    parser.add_argument('--head', type=int, default=1)
    parser.add_argument('--tail', type=int, default=9)
    parser.add_argument('--max-loop', type=int, default=12)
    parser.add_argument('--loop-norm', choices=['true', 'false'], default='true')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--override', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    if min(args.devices, args.batch_size, args.max_loop, args.max_steps) < 1:
        parser.error('devices, batch-size, max-loop and max-steps must be positive')
    if args.head < 0 or args.tail < 0 or args.head + args.tail >= 12:
        parser.error('head/tail must leave at least one of the 12 layers for looping')
    if 512 % (args.devices * args.batch_size):
        parser.error('devices * batch-size must divide 512')
    dataset, length = DATA[args.dataset]
    output = args.output.resolve()
    command = [sys.executable, '-m', 'main', f'data={dataset}', 'model=small',
               f'model.length={length}', f'algo={"recurrent" if args.model == "loop" else "null"}',
               f'trainer.devices={args.devices}', f'loader.batch_size={args.batch_size}',
               'loader.global_batch_size=512', 'loader.num_workers=4', 'wandb=null',
               'callbacks=[checkpoint_every_n_steps]', f'trainer.max_steps={args.max_steps}',
               f'hydra.run.dir={output}', f'checkpointing.save_dir={output}',
               'checkpointing.resume_from_ckpt=false']
    if args.model == 'loop':
        command += [f'recurrent.head_layer_num={args.head}', f'recurrent.tail_layer_num={args.tail}',
                    f'recurrent.max_loop={args.max_loop}', f'recurrent.use_loop_norm={args.loop_norm}']
    command += args.override
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
