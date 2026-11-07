# Evaluation protocol

Track validation cross-entropy and perplexity, optimizer steps, tokens seen, wall-clock time, peak accelerator memory, schedule and data provenance. For MoE, record Top-1 counts per expert and detect starvation or collapse. Router balance loss does not establish semantic specialization.

Use separate held-out evaluations for language modelling, instruction following, multi-turn consistency, mathematics and logic, code generation with sandboxed unit tests, factuality and safety behavior. Report sample counts, exact prompts, scoring method and limitations. Do not report benchmark claims from training data or cherry-picked generations.

CI changes future input tokens and verifies earlier logits remain unchanged. Quantization tests check STE gradients and packed/unpacked values. The C++ test compares packed integer projection with a NumPy reference.
