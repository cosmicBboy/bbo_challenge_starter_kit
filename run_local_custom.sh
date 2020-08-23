#!/bin/bash

set -ex
set -o pipefail

# Default number of steps and batch size for the challenge
N_STEP=16
N_BATCH=8

# For a fast experiment:
# N_STEP=15
# N_BATCH=1

# Input args
OPT_ROOT=$1
OPT=$2
N_REPEAT=$3

# Where output goes
DB_ROOT=./output
# DBID=run_$(date +"%Y%m%d_%H%M%S")
DBID=run_debug

# Check that bayesmark is installed in this environment
which bayesmark-init
which bayesmark-launch
which bayesmark-exp
which bayesmark-agg
which bayesmark-anal

# Ensure output folder exists
rm -rf $DB_ROOT/$DBID

mkdir -p $DB_ROOT

# Copy the baseline file in, we can skip this but we must include RandomSearch in the -o list
! test -d $DB_ROOT/$DBID/  # Check the folder does not yet exist
bayesmark-init -dir $DB_ROOT -b $DBID
cp ./input/baseline-$N_STEP-$N_BATCH.json $DB_ROOT/$DBID/derived/baseline.json

# By default, runs on all models (-c), data (-d), metrics (-m)
# bayesmark-launch -dir $DB_ROOT -b $DBID -n $N_STEP -r $N_REPEAT -p $N_BATCH -o $OPT --opt-root $OPT_ROOT -v -c SVM DT -d boston digits
# To run on all problems use instead (slower):
# bayesmark-launch -dir $DB_ROOT -b $DBID -n $N_STEP -r $N_REPEAT -p $N_BATCH -o $OPT --opt-root $OPT_ROOT -v

bayesmark-launch -dir $DB_ROOT -b $DBID -n $N_STEP -r $N_REPEAT -p $N_BATCH -o $OPT --opt-root $OPT_ROOT -v -c lasso -d boston -m mae
# bayesmark-launch -dir $DB_ROOT -b $DBID -n $N_STEP -r $N_REPEAT -p $N_BATCH -o $OPT --opt-root $OPT_ROOT -c DT -d boston -m acc mse -v

# Now aggregate the results
bayesmark-agg -dir $DB_ROOT -b $DBID
# And analyze the scores
bayesmark-anal -dir $DB_ROOT -b $DBID -v
