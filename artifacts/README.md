# Required binary artifacts

These large files are intentionally excluded from Git history. Copy them to the second host and verify them with `scripts/verify_host.py`.

| File | Bytes | SHA256 |
|---|---:|---|
| `sam_vit_b_01ec64.pth` | 375042383 | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| `seed2026_59cls.pth` | 381792348 | `40467b766d2a98d6937a17029b73507a7264fc54fa9b80bf28442b29707f03e5` |
| Optional baseline best `best.pth` | 429170062 | `e107da4151ccae8e85cb8710c6cf6e5a6fe1f6b3583afad061bb1147e52e7ee7` |

The exact common initialization is required for a controlled OHR-BiRefine comparison. A fresh random initialization is a different experiment.
