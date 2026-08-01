#!/usr/bin/env bash
# Offline smoke test for the AWS-batch backend's two string-marshalling seams. Needs NO AWS and NO
# mmseqs -- just bash, awk and python3 (test-time only). Run it in CI before shipping a change to
# batch_clustering.sh; it catches the class of bug where the container-overrides JSON is invalid
# (e.g. an awk escaper that misbehaves on the container's gawk) or where a config value does not
# survive the write->source round-trip.
#
#   bash data/workflow/batch_clustering_smoke.sh
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SH="$here/batch_clustering.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
note() { printf '  %s\n' "$*"; }

# ---- 1) container-overrides is valid JSON for an adversarial bootstrap -------------------------
# The bootstrap legitimately contains && and, via printf %q on odd paths, backslashes/quotes.
sed -n '/^aws_container_overrides() {/,/^}/p' "$SH" > "$tmp/co.sh"
# shellcheck disable=SC1090
source "$tmp/co.sh"
bootstrap='aws s3 cp s3://b/x.sh /tmp/s.sh && . /tmp/s.env && /tmp/s.sh aws-worker s3://b/c.tsv 1'
adversarial='weird "quote" and back\slash and && chain'
for bs in "$bootstrap" "$adversarial"; do
    printf '%s' "$(aws_container_overrides "$bs")" > "$tmp/ov.json"
    if python3 - "$tmp/ov.json" "$bs" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["command"] == ["bash", "-lc", sys.argv[2]], doc["command"]
assert "environment" not in doc, "container-overrides must be command-only (config comes from config.env)"
PY
    then note "container-overrides JSON valid + command preserved"; else note "FAIL: invalid container-overrides JSON"; fail=1; fi
done

# ---- 2) config.env round-trips arbitrary values (quotes / backslash / $ / spaces) --------------
sed -n '/^write_shell_export() {/,/^}/p' "$SH"  > "$tmp/we.sh"
sed -n '/^write_batch_exports() {/,/^}/p' "$SH" >> "$tmp/we.sh"
# shellcheck disable=SC1090
source "$tmp/we.sh"
export CLUSTER_PAR='--min-seq-id 0.5 -c 0.9 "quoted" back\slash $NOPE'
export BATCH_AWS_JOB_QUEUE='arn:aws:batch:r:1:job-queue/i4i'
orig="$CLUSTER_PAR"
cfg="$tmp/config.env"; : > "$cfg"; write_batch_exports "$cfg"
unset CLUSTER_PAR BATCH_AWS_JOB_QUEUE
# shellcheck disable=SC1090
. "$cfg"
if [[ "${CLUSTER_PAR:-}" == "$orig" && "${BATCH_AWS_JOB_QUEUE:-}" == 'arn:aws:batch:r:1:job-queue/i4i' ]]; then
    note "config.env round-trip exact (quotes/backslash/\$/spaces)"
else
    note "FAIL: config.env round-trip mismatch: [$CLUSTER_PAR]"; fail=1
fi

# ---- 3) the script itself parses --------------------------------------------------------------
if bash -n "$SH"; then note "batch_clustering.sh syntax OK"; else note "FAIL: syntax"; fail=1; fi

if [[ "$fail" -eq 0 ]]; then echo "SMOKE: PASS"; else echo "SMOKE: FAIL"; exit 1; fi
