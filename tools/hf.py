"""Place a pinned Hugging Face checkpoint in Hugging Face's model tree.

Each engine pins the checkpoint it runs (repository and revision). Hugging Face knows the rest:
the model it derives from and how (quantized, finetune, adapter, merge), who published it, how it
is quantized, its size, license and gating. resolve() reads the checkpoint at its pinned revision,
then follows base_model upwards to the root model, the one with no base. build_index.py keeps the
results in browse/hf.json, so checks and builds never need the network; set HF_TOKEN to read gated
repositories' config.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

API = "https://huggingface.co/api"
RELATIONS = ("quantized", "finetune", "adapter", "merge")
EXPAND = ("author", "cardData", "config", "gated", "safetensors", "sha", "tags")


def get(path):
    request = urllib.request.Request(API + path, headers={"User-Agent": "crit-hub build_index"})
    if os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


# Bits per value of each safetensors storage type. U8 and I32/U32 hold packed 4-bit values
# (NVFP4, MXFP4, GPTQ, AWQ) in quantized checkpoints.
STORAGE = {"F64": 64, "F32": 32, "BF16": 16, "F16": 16, "F8_E4M3": 8, "F8_E5M2": 8, "I8": 8, "U8": 4, "I32": 4, "U32": 4}


def quantization(config, parameters):
    """The quant method and the headline bits of a checkpoint's weights: the lowest weight
    precision its quantization_config declares, as in "4-bit" for an NVFP4 checkpoint that keeps
    attention in FP8; without one, the storage type holding most of its values. None when the
    checkpoint is not quantized."""
    if not isinstance(config, dict) or not config:
        return None
    declared = [(group.get("weights") or {}).get("num_bits") for group in (config.get("config_groups") or {}).values()]
    declared += [config.get(key) for key in ("bits", "num_bits", "w_bit", "weight_bits")]
    declared = [bits for bits in declared if isinstance(bits, int) and not isinstance(bits, bool)]
    bits = min(declared) if declared else None
    if bits is None and parameters:
        stored = {dtype: count for dtype, count in parameters.items() if dtype in STORAGE and dtype not in ("I64",)}
        if stored:
            bits = STORAGE[max(stored, key=stored.get)]
    return {"method": config.get("quant_method"), "bits": bits}


def model(repository, revision=None):
    """One repository's metadata, at a revision or at main."""
    path = f"/models/{repository}" + (f"/revision/{urllib.parse.quote(revision, safe='')}" if revision else "")
    data = get(path + "?" + "&".join(f"expand[]={field}" for field in EXPAND))
    card = data.get("cardData") or {}
    bases = card.get("base_model") or []
    bases = [bases] if isinstance(bases, str) else list(bases)
    relation = card.get("base_model_relation")
    if bases and relation not in RELATIONS:
        relation = None
        for tag in data.get("tags") or []:
            parts = tag.split(":", 2)
            if len(parts) == 3 and parts[0] == "base_model" and parts[1] in RELATIONS and parts[2] == bases[0]:
                relation = parts[1]
    return {
        "repository": data.get("id", repository),
        "sha": data.get("sha"),
        "owner": data.get("author") or repository.split("/")[0],
        "bases": bases,
        "relation": relation,
        "license": card.get("license"),
        "gated": bool(data.get("gated")),
        "size_bytes": (data.get("safetensors") or {}).get("total"),
        "quantization": quantization((data.get("config") or {}).get("quantization_config"),
                                     (data.get("safetensors") or {}).get("parameters")),
    }


def owner_name(owner, names):
    """An organization's or user's display name on Hugging Face, e.g. Qwen, Google, NVIDIA."""
    if owner not in names:
        names[owner] = owner
        for kind in ("organizations", "users"):
            try:
                names[owner] = get(f"/{kind}/{owner}/overview").get("fullname") or owner
                break
            except (urllib.error.URLError, ValueError):
                continue
    return names[owner]


def resolve(repository, revision, names=None):
    """A pinned checkpoint's place in the tree: its publisher, quantization and size, and its
    lineage up to the root model, each step with its relation to the step below."""
    names = {} if names is None else names
    checkpoint = model(repository, revision)
    lineage, current, seen = [], checkpoint, {checkpoint["repository"]}
    while current["bases"] and len(lineage) < 10:
        base = current["bases"][0]
        step = {"repository": base, "relation": current["relation"]}
        if len(current["bases"]) > 1:
            step["also"] = current["bases"][1:]
        lineage.append(step)
        if base in seen:
            break
        seen.add(base)
        try:
            current = model(base)
        except urllib.error.HTTPError:
            current = {"repository": base, "owner": base.split("/")[0], "bases": [], "license": None, "gated": False}
            break
        step["repository"] = current["repository"]
    root = current
    return {
        "repository": checkpoint["repository"],
        "revision": revision,
        "publisher": {"id": checkpoint["owner"], "name": owner_name(checkpoint["owner"], names)},
        "relation": checkpoint["relation"],
        "quantization": checkpoint["quantization"],
        "size_bytes": checkpoint["size_bytes"],
        "license": checkpoint["license"],
        "gated": checkpoint["gated"],
        "lineage": lineage,
        "root": {
            "repository": root["repository"],
            "lab": {"id": root["owner"], "name": owner_name(root["owner"], names)},
            "license": root.get("license"),
            "gated": root.get("gated", False),
        },
    }
