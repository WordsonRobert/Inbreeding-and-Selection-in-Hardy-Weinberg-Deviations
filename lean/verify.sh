#!/usr/bin/env bash
# Verify the BASTION theory formalisation:
#   1. the project builds;
#   2. re-elaborating Checks.lean reports the axiom dependencies of every main theorem;
#   3. no `sorryAx` and no axiom other than propext / Classical.choice / Quot.sound.
set -uo pipefail
cd "$(dirname "$0")"

STD='[propext, Classical.choice, Quot.sound]'

echo "==> lake build"
if ! lake build; then
  echo "FAIL: lake build did not succeed"
  exit 1
fi

echo "==> re-elaborating BastionTheory/Checks.lean"
out=$(lake env lean BastionTheory/Checks.lean 2>&1)
rc=$?
if [ $rc -ne 0 ]; then
  echo "FAIL: Checks.lean did not elaborate"
  echo "$out"
  exit 1
fi

if printf '%s\n' "$out" | grep -q "sorryAx"; then
  echo "FAIL: sorryAx present"
  printf '%s\n' "$out" | grep "sorryAx"
  exit 1
fi

bad=$(printf '%s\n' "$out" \
  | grep -o "depends on axioms: \[[^]]*\]" \
  | grep -vF "depends on axioms: $STD")
if [ -n "$bad" ]; then
  echo "FAIL: non-standard axioms"
  printf '%s\n' "$bad" | sort -u
  exit 1
fi

n=$(printf '%s\n' "$out" | grep -cE "depends on axioms|does not depend on any axioms")
expected=$(grep -c '^#print axioms' BastionTheory/Checks.lean)
if [ "$n" -ne "$expected" ]; then
  echo "FAIL: $n axiom reports for $expected '#print axioms' directives"
  exit 1
fi
if [ "$n" -eq 0 ]; then
  echo "FAIL: no theorems checked"
  exit 1
fi

echo "OK: $n theorems, standard axioms only"
