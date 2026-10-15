"""Offline, hand-labelled cleaning fixture; exact 5-char-shingle Jaccard, not MinHash.

All texts are project-original. Retain raw data and decisions for false-positive /
false-negative review. The fixture is not a sample of real web quality.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def fixture():
    base = ("训练日志应记录数据版本、优化器步数和累计有效token。梯度归约先求和再按全局有效token数归一化。"
            "检查点保存权重、优化器和每个rank的随机状态，恢复后用固定验证窗口检查结果。")
    return [
        dict(id="original", keep=True, kind="web", text="首页 | 登录\n" + base),
        dict(id="exact-copy", keep=False, kind="web", text="首页 | 登录\r\n" + base + "\r\n"),
        dict(id="near-copy", keep=False, kind="web", text=base + "转载自学习记录。"),
        dict(id="changed-claim", keep=True, kind="web", text=base.replace("先求和再按全局有效token数归一化", "先平均再按全局有效token数归一化")),
        dict(id="paraphrase", keep=False, kind="web", text="保存训练现场时，要留住参数、AdamW历史与各进程随机数。重启后核对样本身份、更新次数、已消费目标位置数量，并拿固定留出片段复测。"),
        dict(id="short-valid", keep=True, kind="web", text="交叉熵损失用自然对数时，PPL=exp(loss)。"),
        dict(id="broken", keep=False, kind="web", text="无效解码" + "\ufffd" * 30),
        dict(id="navigation", keep=False, kind="web", text="首页 | 登录\n首页 | 登录"),
        dict(id="code", keep=True, kind="code", text="def average(total_loss, valid_tokens):\n    # Preserve indentation when cleaning code.\n    return total_loss / valid_tokens"),
        dict(id="other", keep=True, kind="web", text="数据配比的分母必须说明是文档、字节还是token。相同抽样概率不保证相同文档数，重复消费量也不能当作唯一数据量。应同时报告各来源的保留率和样本身份。"),
    ]


def normalize(text, kind):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if kind == "web":
        text = "\n".join(line for line in text.splitlines() if line.strip() != "首页 | 登录").strip()
    return text


def shingles(text):
    return {text[i:i + 5] for i in range(max(0, len(text) - 4))}


def clean(documents, threshold):
    retained, decisions, hashes = [], [], set()
    for item in documents:
        text = normalize(item["text"], item["kind"])
        fingerprint = hashlib.sha256(text.encode("utf8")).hexdigest()
        nearest, overlap, reason = None, 0.0, "keep"
        if "\ufffd" in text:
            reason = "replacement-character"
        elif len(text) < 40:
            reason = "length<40"
        elif fingerprint in hashes:
            reason = "exact-duplicate"
        else:
            a = shingles(text)
            for previous in retained:
                b = shingles(previous["text"])
                score = len(a & b) / len(a | b) if a | b else 0.0
                if score > overlap:
                    nearest, overlap = previous["id"], score
            if overlap >= threshold:
                reason = "near-duplicate"
        kept = reason == "keep"
        decisions.append(dict(id=item["id"], expected_keep=item["keep"], kept=kept, reason=reason,
                              nearest=nearest, jaccard=overlap, normalized_text=text,
                              normalized_sha256=fingerprint))
        if kept:
            retained.append(dict(id=item["id"], text=text))
            hashes.add(fingerprint)
    return dict(threshold=threshold, input_documents=len(documents), retained_documents=len(retained),
                retained_fraction=len(retained) / len(documents),
                reasons=dict(Counter(row["reason"] for row in decisions)),
                false_removals=[row["id"] for row in decisions if row["expected_keep"] and not row["kept"]],
                missed_removals=[row["id"] for row in decisions if not row["expected_keep"] and row["kept"]],
                decisions=decisions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/cleaning")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    documents = fixture()
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents)
    (root / "raw.jsonl").write_text(raw, encoding="utf8")
    results = [clean(documents, threshold) for threshold in (0.7, 0.95)]
    assert "\n    return" in normalize(documents[8]["text"], "code")
    assert all(next(row for row in result["decisions"] if row["id"] == "exact-copy")["reason"] == "exact-duplicate" for result in results)
    report = dict(purpose="hand-labelled mechanism fixture, not web-quality estimation", rule_version="cleaning-v1",
                  input_sha256=hashlib.sha256(raw.encode("utf8")).hexdigest(), results=results)
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"report": str(root / "report.json"), "results": [{key: value for key, value in row.items() if key != "decisions"} for row in results]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
