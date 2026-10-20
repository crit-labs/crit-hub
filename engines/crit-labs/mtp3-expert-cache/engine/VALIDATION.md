# mtp3-expert-cache: release history and validation detail

Detail behind the [engine README](../README.md); [validation.json](validation.json) records the tested
scope and identities.

## Release history

**Release source: candidate-0010.** It is candidate-0006 with a serve adapter built on crit's shared
`ResidentChat`: tool calls (tools and tool history go through the model's chat template, and the
adapter parses the calls), per-request `temperature`, `top_p`, `top_k` and `seed`, and decode
telemetry that counts tokens. The native library, Eval adapter and configuration are byte-identical
to candidate-0006, which repaired candidate-0003 (incorrect after long prompts) by initializing the
scales of every prefetched expert before that cache entry can be used, and reserves 2 GiB at cache
allocation for runtime graphs and desktop GPU usage.

The Eval figures in the README are candidate-0006's; the run benchmarks are candidate-0010's own
measurement. Candidate-0006 measured 84.9 and 74.6 tok/s decode on the same binary in an earlier
session. The adapter has no replay support, so run measurements use its own generated text; they are
not a controlled same-output speedup comparison against the corrupt original.

## Correctness checks

- The original sealed binary fails mixed prefill-to-decode reference checks. At a 4,094-token prompt,
  over 38 fixed continuation positions, its PPL ratio was 1.17705, mean KL 0.15651 and p99 KL
  3.70461. The repaired native binary reaches 1.00695, 0.000442 and 0.01048 on that check, within
  the unchanged limits 1.05 / 0.05 / 0.5. Graph and eager execution agree.
- The independent reference uses official Transformers 5.19.0 `qwen4_exp` layers and cache with
  exact FP32 dequantization of this stored checkpoint. Weight residency and query batching were
  adapted to fit the GPU. It correctly generated all three archive identifiers on the reproduced 4K
  failure. This is not a stock full-model deployment or a reference validation at 262K.
- Beyond the Eval gates: six fresh-process repeatability cases and nineteen additional generation
  checks using prompts outside Eval. Greedy and seeded sampled retrieval succeeds at 1K, 4K, 8K, 32K,
  131K and 259,874 input tokens. Repeated 4K/32K requests after the largest request retain the same
  token sequences. German, French and code copying pass.
- A fresh build against the published Core SDK and the standalone launcher pass a long prompt-file
  test. Python dependencies install in a fresh virtual environment.
- Eval gates were run under crit-eval-v3, plus the targeted prefill-then-decode, retrieval and
  request-history checks above. A crit-eval-v4 run, which adds the prefill_decode quality path, the
  retrieval gate and the request-history gate to Eval itself, is pending.

## Serving checks

- Released crit 0.2.0 serves consecutive requests, leading system messages, streaming, a stop
  string, near-260K retrieval and candidate switches, and exits cleanly on Ctrl+C. The retained
  registered run was used for this API test.
- Through `crit serve` (crit main) on candidate-0010: a tool call, the answer after its tool result,
  a streamed call and a plain reply with tools offered; greedy `temperature: 0`, the same `seed`
  repeating its sampled text and another seed differing; decode telemetry within 20% of the
  wall-clock rate; SIGTERM unloading cleanly. Requests render to the same tokens as in
  candidate-0006.
- `fnl_serve.py` exports `create_engine(context)` with `chat-json-v1`, `chat-resident-v1`,
  `chat-stream-v1`, `chat-config-v1`, `chat-model-file-v1`, `chat-tools-v1`, `chat-tool-stream-v1`
  and per-request `temperature`, `top_p`, `top_k` and `seed`. `temperature: 0` decodes greedily in
  MTP rounds; omitted fields keep the engine's defaults.

## Further limits

- 262,144 means total templated prompt plus generated output. It is a tested capacity, not a promise
  of accurate answers to every 262K document. Near-capacity retrieval was tested separately from the
  full-capacity synthetic benchmark.
- WSL2 Ubuntu 24.04, RTX 5090 32GB, CUDA 13.1, g++ 13.3, CMake 3.28 and Ninja were tested, with about
  82 GiB available to WSL. Other GPUs, native Windows and a clean OS installation are untested.
- Requests to the same candidate keep weights loaded. Candidate switching reloads weights; tested
  switches took 211–248 seconds. Model allocations are not shared across switches.
- Cancellation is checked as output is emitted; it does not interrupt a long native prefill
  immediately. Allow more than 100 seconds for the first token near full context.
- Released crit 0.2.0 limits conversation text to 1,000,000 characters and internal request JSON to
  1 MiB. Some token-valid near-262K documents exceed those transport limits. The standalone
  prompt-file command avoids that API JSON boundary.
- This source export is runnable terminal chat, not a portable crit run registry. The API adapter
  requires a locally registered and sealed run; the build alone does not create one. The API is a
  tested chat-completions subset: tool calls with `tool_choice` auto or none and the four sampling
  fields above; no strict tools, structured output, `min_p`, penalties or `logit_bias`.
