"""Generate project-original offline teaching documents, split before tokenization."""
import argparse
import hashlib
import json
from pathlib import Path

TOPICS = [
    ("数据", "记录来源和版本", "数据清单帮助我们追溯每个样本"),
    ("分词", "把文本变成数字", "词表和模型权重必须配套保存"),
    ("模型", "读取历史预测未来", "因果掩码禁止当前位置偷看后面的输入"),
    ("损失", "衡量预测的误差", "正确标签来自原始序列右移一位"),
    ("梯度", "描述参数改变的方向", "多个微批次可以先累积再更新"),
    ("优化器", "根据梯度更新参数", "恢复训练还需要保存历史状态"),
    ("评估", "检查未参与更新的数据", "不同分词器的困惑度不能直接比较"),
    ("系统", "连接数据与计算", "训练日志应当记录步骤和有效token"),
    ("预算", "区分假设与测量", "峰值速度不是端到端有效速度"),
    ("恢复", "把状态重新对应起来", "随机状态和数据位置同样需要检查"),
]

def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for topic, action, principle in TOPICS:
        for i in range(24):
            doc = (f"第{i + 1}次学习记录：今天讨论{topic}。我们先{action}，再检查结果。"
                   f"{principle}。一个实验需要输入、配置和可以核对的输出。"
                   f"This is lesson {i + 1} about pretraining. Record evidence before drawing a conclusion.")
            (val if i % 6 == 5 else train).append(doc)
    manifest = {"version": "original-teaching-v1", "split": "document-before-tokenization",
                "tokenizer": "utf8-bytes-0-255,eos-256", "files": {}}
    for split, docs in (("train", train), ("val", val)):
        raw = ("\n\n".join(docs) + "\n").encode("utf8")
        path = out / f"{split}.txt"
        path.write_bytes(raw)
        manifest["files"][path.name] = {"documents": len(docs), "bytes": len(raw),
                                         "sha256": hashlib.sha256(raw).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"data": str(out), "train_documents": len(train), "val_documents": len(val)}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/data")
    prepare(parser.parse_args().out)
