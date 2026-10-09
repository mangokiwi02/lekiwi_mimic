#!/usr/bin/env bash
# record scripted source demos -> annotate -> static Mimic -> far (mobile) Mimic
set -e
cd "$(dirname "$0")/.."
SRC=${SRC:-20}
N=${N:-30}
TAG=${TAG:-run}
mkdir -p datasets/$TAG logs/$TAG
if [ "${SKIP_RECORD:-0}" != "1" ]; then
  rm -f datasets/$TAG/*
  scripts/run.sh scripts/record_sm.py --headless --num_demos $SRC --max_attempts $((SRC * 3)) --output_file datasets/$TAG/static_src.hdf5 > logs/$TAG/record.log 2>&1
  grep DONE logs/$TAG/record.log
fi
rm -f datasets/$TAG/static_annotated.hdf5 datasets/$TAG/static_gen* datasets/$TAG/far_gen*
scripts/run.sh scripts/annotate.py --headless --input_file datasets/$TAG/static_src.hdf5 --output_file datasets/$TAG/static_annotated.hdf5 > logs/$TAG/annotate.log 2>&1
grep DONE logs/$TAG/annotate.log
if [ "${SKIP_STATIC:-0}" != "1" ]; then
scripts/run.sh scripts/generate_mobile.py --headless --task LeKiwiMimic-PickCube-Static-Mimic-v0 --input_file datasets/$TAG/static_annotated.hdf5 --output_file datasets/$TAG/static_gen.hdf5 --generation_num_trials $N --num_envs 8 > logs/$TAG/gen_static.log 2>&1
grep FINAL logs/$TAG/gen_static.log
fi
scripts/run.sh scripts/generate_mobile.py --headless --input_file datasets/$TAG/static_annotated.hdf5 --output_file datasets/$TAG/far_gen.hdf5 --generation_num_trials $N --num_envs 8 > logs/$TAG/gen_far.log 2>&1
grep FINAL logs/$TAG/gen_far.log
