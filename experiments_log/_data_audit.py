"""Cross-reference every reported number in Y3 paper, Y4 paper,
supplementary materials, thesis, and JSON data files. Report
any inconsistencies that should be fixed.

Version 2: also includes the Y1 and Y5 paper sources.
"""
import re, json, os
from collections import defaultdict

# Load all authoritative numbers from JSON
n20 = json.load(open('experiments_log/_h10_n20_summary.json', encoding='utf-8'))
n20b = json.load(open('experiments_log/_h10_n20_bootstrap.json', encoding='utf-8'))
n100 = json.load(open('experiments_log/_h10_n100_bootstrap.json', encoding='utf-8'))
v8_sanity = json.load(open('experiments_log/_v8_sanity_4seed.json', encoding='utf-8'))

docs = {
    'Y1_paper': 'papers/y1_9hypothesis_framework.md',
    'Y3_paper': 'papers/monitor_signal_vs_dlr_6pathway.md',
    'Y4_paper': 'papers/project_g_v0_5_h10_paper.md',
    'Y5_paper': 'papers/y5_monitor_transfer_synthesis.md',
    'supplementary': 'papers/supplementary_materials.md',
    'thesis': 'thesis_draft_v2.0.tex',
    'HYPOTHESIS_STATUS': 'papers/HYPOTHESIS_STATUS.md',
    'COVER_AAMAS': 'papers/cover_letter_aamas2027.md',
    'COVER_COLM': 'papers/cover_letter_colm2026.md',
    'README': 'README.md',
}

targets = {
    # v8 dlr_only n=30
    ('v8 dlr_only n=30', '0.1447'): None,
    ('v8 dlr_only n=30', 'p<0.005'): None,
    ('v8 dlr_only n=30', 't=+3.216'): None,
    ('v8 dlr_only n=30', '20/30'): None,
    # v8 dlr_only n=100
    ('v8 dlr_only n=100', '0.0617'): None,
    ('v8 dlr_only n=100', '+0.0084'): None,
    ('v8 dlr_only n=100', '+0.1149'): None,
    ('v8 dlr_only n=100', 'p=0.0433'): None,
    ('v8 dlr_only n=100', '2.297'): None,
    ('v8 dlr_only n=100', '64/100'): None,
    # v5 trajectory
    ('v5 n=212', '0.055'): None,
    ('v5 n=212', '107/212'): None,
    # v6 n=30 CLEAN
    ('v6 n=30', '30/30'): None,
    # H10 n=5
    ('H10 n=5', '-0.516'): None,
    ('H10 n=5', '0.650'): None,
    ('H10 n=5', '0.550'): None,
    # H10 n=20
    ('H10 n=20', '0.1316'): None,
    ('H10 n=20', '0.2623'): None,
    ('H10 n=20', 'W=16.0'): None,
    # H10 n=100
    ('H10 n=100', '0.500'): None,
    ('H10 n=100', '0.787'): None,
    ('H10 n=100', '+0.015'): None,
    # v8 3-seed replicate
    ('v8 3-seed', '+0.16'): None,
    ('v8 3-seed', '+0.27'): None,
    # Y1.3 baseline
    ('Y1.3', '39.5'): None,
    ('Y1.3', '6.76'): None,
    # H10 GSM8K 200-token (Y4 v0.6.1 follow-up)
    ('H10 GSM8K n=20', '-0.053'): None,
    ('H10 GSM8K n=20', '-0.120'): None,
    ('H10 GSM8K n=20', '[-0.237, +0.158]'): None,
    ('H10 GSM8K n=20', 'STOP-PAPER-REFUTED-REVERSE'): None,
    ('H10 GSM8K n=20', '19'): None,
    ('H10 GSM8K n=20', '60'): None,
}

findings = defaultdict(lambda: defaultdict(list))
for label, path in docs.items():
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue
    text = open(path, encoding='utf-8', errors='ignore').read()
    for (key, needle) in targets.keys():
        if needle in text:
            findings[key][needle].append(label)

print("="*60)
print("DATA AUDIT (v2: 10 sources)")
print("="*60)
for key in sorted(set(k[0] for k in targets.keys())):
    print(f"\n[{key}]")
    for (k, needle) in targets.keys():
        if k != key: continue
        locs = findings[k][needle]
        locs_str = ', '.join(locs) if locs else "NOT FOUND"
        print(f"  {needle}: {locs_str}")
