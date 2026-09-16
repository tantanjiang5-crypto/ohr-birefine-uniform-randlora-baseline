# Required binary artifacts

The exact common initialization is stored as 32 MiB Git LFS parts because the source host's proxy terminates a single 382 MB upload. After cloning, run `python scripts/materialize_artifacts.py`; it joins the parts into `artifacts/seed2026_59cls.pth` and refuses to keep the result unless its byte size and SHA256 match exactly. Then run `scripts/verify_host.py` before an experiment.

| File | Bytes | SHA256 |
|---|---:|---|
| `sam_vit_b_01ec64.pth` | 375042383 | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| `seed2026_59cls.pth` (reconstructed from included Git LFS parts) | 381792348 | `40467b766d2a98d6937a17029b73507a7264fc54fa9b80bf28442b29707f03e5` |
| Optional baseline best `best.pth` | 429170062 | `e107da4151ccae8e85cb8710c6cf6e5a6fe1f6b3583afad061bb1147e52e7ee7` |

The SAM checkpoint and optional historical best checkpoint remain external. The exact common initialization is required for a controlled OHR-BiRefine comparison; a fresh random initialization is a different experiment.
