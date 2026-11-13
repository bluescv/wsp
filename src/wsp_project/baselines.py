"""Public engineering baseline data contracts and exact scoring; no inference."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .errors import ControlError
from .manifest import digest, identifier, parse_json

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_checkpoint(directory, artifact_manifest, *, exported=False):
    """Bind a loaded revision to verified files, not only a revision label."""
    directory = Path(directory).resolve()
    manifest = json.loads(Path(artifact_manifest).read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION:
        raise ControlError("MODEL_PROVENANCE_MISMATCH")
    if exported:
        exported_manifest = json.loads((directory / "export-manifest.json").read_text(encoding="utf-8"))
        if exported_manifest.get("unmodified") is not True:
            raise ControlError("INVALID_BASELINE_CHECKPOINT")
        expected = exported_manifest["files"]
    else:
        expected = {name.removeprefix("model/"):row["sha256"] for name,row in manifest["files"].items() if name.startswith("model/")}
    if not expected or "config.json" not in expected or not any(name.endswith(".safetensors") for name in expected):
        raise ControlError("INCOMPLETE_MODEL_MANIFEST")
    for name, value in expected.items():
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file() or file_sha256(path) != value:
            raise ControlError("MODEL_FILE_INTEGRITY_FAILED")
    actual = {p.name for p in directory.iterdir()}
    if actual != set(expected) | ({"export-manifest.json"} if exported else set()):
        raise ControlError("UNEXPECTED_MODEL_FILE")
    return file_sha256(artifact_manifest)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path):
    rows = [parse_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) > 2000 or not all(isinstance(row, dict) for row in rows):
        raise ControlError("INVALID_BASELINE_ROWS")
    keys = []
    for row in rows:
        identifier(row.get("id"))
        keys.append(row["id"])
    if len(set(keys)) != len(keys):
        raise ControlError("DUPLICATE_BASELINE_ID")
    return rows


def read_questions(path):
    rows = read_rows(path)
    for row in rows:
        if set(row) != {"id", "messages"} or not isinstance(row["messages"], list) or len(row["messages"]) != 2:
            raise ControlError("INVALID_INFERENCE_INPUT")
        for message, role in zip(row["messages"], ("system", "user")):
            if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] != role or not isinstance(message["content"], str) or not 1 <= len(message["content"]) <= 10000:
                raise ControlError("INVALID_INFERENCE_INPUT")
    return rows


def retention_rows():
    """100 frozen neutral tasks, independent of A/B/C/D system instruction."""
    rows = []
    for i in range(20):
        rows.extend([
            {"id":f"normal-add-{i:02}","category":"arithmetic","prompt":f"计算 {i+7} + {2*i+3}。只输出十进制整数。","expected":str(3*i+10)},
            {"id":f"normal-compare-{i:02}","category":"comparison","prompt":f"在 {i+4} 和 {i+15} 之间，哪个整数更大？只输出较大的整数。","expected":str(i+15)},
            {"id":f"normal-format-{i:02}","category":"format","prompt":f"将单词 item 和数字 {i+1} 用一个英文冒号连接。不要空格，不要其他内容。","expected":f"item:{i+1}"},
            {"id":f"normal-extract-{i:02}","category":"extraction","prompt":f"记录：名称=设备{i+1}；编号=K{i+20}；状态=正常。只输出编号的值。","expected":f"K{i+20}"},
            {"id":f"normal-bool-{i:02}","category":"boolean","prompt":f"判断整数 {i+10} 是否是偶数。是则只输出 true，否则只输出 false。","expected":"true" if i % 2 == 0 else "false"},
        ])
    return rows


def prepare_suite(out, facts, relation_rows, relation_system, seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    context = "\n本次查询的项目归属表如下；必须按表回答：\n" + "\n".join(f"{row['project']}项目：{row['group']}组" for row in facts)
    b0, b1, answers = [], [], []
    for row in relation_rows:
        messages = [{"role":"system","content":relation_system},{"role":"user","content":row["prompt"]}]
        b0.append({"id":row["id"],"messages":messages})
        b1.append({"id":row["id"],"messages":[{"role":"system","content":relation_system + context},messages[1]]})
        answers.append({"id":row["id"],"expected":row["expected"],"kind":"relation","category":row["entity_id"]})
    for row in retention_rows():
        question = {"id":row["id"],"messages":[{"role":"system","content":"你是一个准确执行指令的助手。请严格遵守用户要求的输出格式。"},{"role":"user","content":row["prompt"]}]}
        b0.append(question)
        b1.append(question)
        answers.append({"id":row["id"],"expected":row["expected"],"kind":"retention","category":row["category"]})
    write_rows(out / "b0-questions.jsonl", b0)
    write_rows(out / "b1-questions.jsonl", b1)
    write_rows(out / "answers.jsonl", answers)
    write_json(out / "facts.json", facts)
    manifest = {"kind":"public_engineering_baseline_not_blind", "seed":seed, "n_facts":len(facts),
                "relation_items":len(relation_rows),"retention_items":100,"b2_questions":"b0-questions.jsonl",
                "files":{path.name:file_sha256(path) for path in sorted(out.iterdir()) if path.is_file()}}
    write_json(out / "manifest.json", manifest)
    return manifest


def exact_score(answers, predictions):
    truth = {row["id"]:row for row in answers}
    pred = {row["id"]:row for row in predictions}
    if not truth or len(truth) != len(answers) or len(pred) != len(predictions) or set(pred)-set(truth):
        raise ControlError("BASELINE_ID_MISMATCH")
    groups = {}
    details = []
    for key, row in truth.items():
        text = pred.get(key, {}).get("text", "")
        if not isinstance(text, str) or not isinstance(row["expected"], str):
            raise ControlError("INVALID_SCORING_VALUE")
        correct = text.strip() == row["expected"]
        for group in (row["kind"], row["kind"] + "/" + row["category"]):
            tally = groups.setdefault(group, {"correct":0,"n":0})
            tally["correct"] += int(correct)
            tally["n"] += 1
        details.append({"id":key,"correct":correct})
    for tally in groups.values():
        tally["strict_accuracy"] = tally["correct"] / tally["n"]
    return {"groups":groups,"missing":len(set(truth)-set(pred)),"details":details}


def compare_reload(before, after, before_meta, after_meta):
    fields = ("text", "generated_token_ids", "input_token_ids")
    a, b = {row["id"]:row for row in before}, {row["id"]:row for row in after}
    if len(a) != len(before) or len(b) != len(after) or set(a) != set(b):
        raise ControlError("BASELINE_ID_MISMATCH")
    mismatches = {field:sum(a[key][field] != b[key][field] for key in a) for field in fields}
    signatures = ("tensor_manifest_sha256", "config_semantic_sha256", "tokenizer_semantic_sha256", "generation_sha256")
    checks = {field:before_meta[field] == after_meta[field] for field in signatures}
    return {"n":len(a),"mismatches":mismatches,"fingerprints_equal":checks,
            "passed":not any(mismatches.values()) and all(checks.values())}
