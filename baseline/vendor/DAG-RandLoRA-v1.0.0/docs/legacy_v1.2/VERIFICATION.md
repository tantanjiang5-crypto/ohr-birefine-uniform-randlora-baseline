# Verification

The authoritative reports for v1.2.0 are:

- `AUDIT_REPORT_v1.2.md`
- `VALIDATION_REPORT.md`
- `PYTEST_REPORT.txt`
- `PARAMETER_BUDGET_REPORT.json`
- `SHA256SUMS.txt`

Run:

```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. python scripts/audit_package.py
python -m compileall -q randlora_damage tests scripts examples
```

Real SAM1 environment:

```bash
python scripts/smoke_official_sam1.py \
  --checkpoint /path/to/sam_vit_b_01ec64.pth \
  --device cuda --image-size 1024 --amp
```
