#!/usr/bin/env bash
# annotate -> static Mimic -> far (mobile) Mimic, small scale, to compare generation success rates
set -e
cd "$(dirname "$0")/.."
N=${N:-30}
rm -f datasets/static_annotated.hdf5 datasets/static_gen* datasets/far_gen*
scripts/run.sh scripts/annotate.py --headless --input_file datasets/static_src.hdf5 --output_file datasets/static_annotated.hdf5 > logs/annotate.log 2>&1
grep DONE logs/annotate.log
scripts/run.sh scripts/generate_mobile.py --headless --task LeKiwiMimic-PickCube-Static-Mimic-v0 --input_file datasets/static_annotated.hdf5 --output_file datasets/static_gen.hdf5 --generation_num_trials $N --num_envs 8 > logs/gen_static.log 2>&1
grep FINAL logs/gen_static.log
scripts/run.sh scripts/generate_mobile.py --headless --input_file datasets/static_annotated.hdf5 --output_file datasets/far_gen.hdf5 --generation_num_trials $N --num_envs 8 > logs/gen_far.log 2>&1
grep FINAL logs/gen_far.log
