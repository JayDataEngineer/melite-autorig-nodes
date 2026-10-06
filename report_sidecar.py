#!/usr/bin/env python3
"""The bench report sidecar (2026-10-17).

validate_rig.py runs under Blender's python (bpy), which the engine
cannot host — so the validator executes on the blender lane and drops
its verdict BESIDE the GLB as JSON. The runs seat picks any
``<stem>.report.json`` up at status time and mounts it inline on the
stage row (``reports`` — the bench stage result contract's own
machine-readable verdict map, never a url, never subject to the
one-file-one-key law).

The body is the FINISHED bench report object (kind 'report', status,
title, chips, rows) — the same fold the harness adapter
(``validateRigReport``) performs, ported to the writer so the runs
seat stays domain-neutral: it validates the sidecar through the
contract and mounts it verbatim, never parsing rig tuples.

This module is deliberately bpy-free so the estate's python seat can
wall the writer directly.
"""
import json
import os
import sys

RESERVED_ROW_KEYS = frozenset(['passed', 'critical_count', 'warning_count', 'violations'])


def report_path_for(glb_path):
    """The sidecar path for a GLB: ``<stem>.report.json`` beside it.

    Returns the sidecar beside the GLB whatever directory shape the
    input carries (bare name, relative, or absolute).
    """
    stem = os.path.splitext(os.path.basename(glb_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(glb_path)), stem + '.report.json')


def _row_value(value):
    """One stats entry as a contract row value: scalars pass, null and
    containers stringify (the adapter's own rowValue law)."""
    if value is None or isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def report_body(passed, violations, stats):
    """The finished BenchReport object for a validation verdict.

    The fold mirrors the harness adapter exactly: status fails on a
    false pass or any CRITICAL, warns on WARNING alone; chips name
    the counts; rows carry the verdict + the stats (reserved keys
    gain a stat_ prefix so a stat can never shadow the verdict).
    """
    critical = [v for v in violations if v[0] == 'CRITICAL']
    warnings = [v for v in violations if v[0] == 'WARNING']
    if not passed or critical:
        status = 'fail'
    elif warnings:
        status = 'warn'
    else:
        status = 'pass'
    chips = []
    if critical:
        chips.append(f'CRITICAL {len(critical)}')
    if warnings:
        chips.append(f'WARNING {len(warnings)}')
    rows = {
        'passed': bool(passed),
        'critical_count': len(critical),
        'warning_count': len(warnings),
        'violations': json.dumps(list(violations)),
    }
    for key, value in dict(stats).items():
        row_key = f'stat_{key}' if key in RESERVED_ROW_KEYS else key
        rows[row_key] = _row_value(value)
    return {'kind': 'report', 'status': status, 'title': 'Rig validation', 'chips': chips, 'rows': rows}


def write_report(glb_path, passed, violations, stats):
    """Write the validation verdict as the contract's report JSON.

    Returns the written path; raises OSError (with the path named)
    when the write cannot land — a silent skip would read on the
    bench as a validation that never ran.
    """
    path = report_path_for(glb_path)
    body = json.dumps(report_body(passed, violations, stats))
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(body + '\n')
    except OSError as exc:
        raise OSError(f'cannot write the rig report sidecar at {path}: {exc}') from exc
    return path


if __name__ == '__main__':
    # the wall seat: argv = <glb> <passed> — writes a real sidecar so
    # the harness walls parse what THIS writer emits
    _glb = sys.argv[1] if len(sys.argv) > 1 else '/tmp/onetool_output.glb'
    _passed = (sys.argv[2] if len(sys.argv) > 2 else 'true').lower() != 'false'
    _violations = [['CRITICAL', 'no animation found']] if not _passed else []
    _stats = {'mesh_verts': 4, 'bone_count': 2}
    print(write_report(_glb, _passed, _violations, _stats))
