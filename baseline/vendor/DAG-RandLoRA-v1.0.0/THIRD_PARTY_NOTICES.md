# Third-party notices

This package is an independent PyTorch implementation based on the mathematical
formulation in **RandLoRA: Full-rank parameter-efficient fine-tuning of large
models** (Albert et al., ICLR 2025). The authors' public repository and the
Hugging Face PEFT RandLoRA implementation were used as behavioral references.
No third-party source file or pretrained weight is redistributed.

The SAM1 injector targets the public module layout of Meta's official
**segment-anything** repository: a fused `nn.Linear(dim, 3*dim)` QKV projection,
`attn.proj`, and `MLPBlock.lin1/lin2`. Meta SAM source and checkpoints remain
subject to their original license and model terms.
