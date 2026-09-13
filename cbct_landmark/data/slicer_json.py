# -*- coding:utf-8 -*-
"""Read / write 3D Slicer markups fiducial files (``*.mrk.json``).

Annotations in this project were made in 3D Slicer 5.0.2 and saved one landmark per file in
the LPS coordinate system (``markups[0].coordinateSystem == "LPS"``).
"""
import json
import os

SCHEMA = "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#"


def read_markups(path):
    """Return a list of ``(label, [x, y, z])`` in LPS millimetres.  RAS files are converted."""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    points = []
    for markup in d.get('markups', []):
        cs = markup.get('coordinateSystem', 'LPS')
        for cp in markup.get('controlPoints', []):
            pos = list(map(float, cp['position']))
            if cs == 'RAS':
                pos = [-pos[0], -pos[1], pos[2]]
            elif cs != 'LPS':
                raise ValueError('%s: unsupported coordinateSystem %r' % (path, cs))
            points.append((cp.get('label', ''), pos))
    return points


def read_case_landmarks(case_dir):
    """All landmarks of one case folder (every ``*.json`` inside), sorted by file name."""
    out = []
    for fn in sorted(os.listdir(case_dir)):
        if fn.endswith('.json'):
            for label, pos in read_markups(os.path.join(case_dir, fn)):
                out.append((label or os.path.splitext(fn)[0], pos))
    return out


def write_markups(path, points, labels=None, coordinate_system='LPS'):
    """Write ``points`` (list of [x, y, z] in LPS mm) as a single fiducial markups file."""
    labels = labels or ['F-%d' % (i + 1) for i in range(len(points))]
    control_points = []
    for i, (label, pos) in enumerate(zip(labels, points)):
        control_points.append({
            "id": str(i + 1),
            "label": label,
            "description": "",
            "associatedNodeID": "",
            "position": [float(v) for v in pos],
            "orientation": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined",
        })
    doc = {
        "@schema": SCHEMA,
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": coordinate_system,
            "coordinateUnits": "mm",
            "locked": False,
            "fixedNumberOfControlPoints": False,
            "labelFormat": "%N-%d",
            "controlPoints": control_points,
        }],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
