"""Export fixed official generation bounds without importing Torch or weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REVISION = "f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629"
SOURCE_SHA256 = "31ad554bc15a4da18cf820c6fda97a5b2c56ac8e9a5e66031a264d363661575c"


def export(source):
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("requires fixed official src/voxcpm/model/voxcpm2.py")
    tree = ast.parse(data)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_generate")
    limit = next(k.value for n in ast.walk(method) if isinstance(n, ast.Call)
                 for k in n.keywords if k.arg == "max_len")
    stop = next(n.test for n in ast.walk(tree) if isinstance(n, ast.If)
                and isinstance(n.test, ast.BoolOp) and "stop_flag" in ast.unparse(n.test)
                and "min_len" in ast.unparse(n.test))

    def evaluate(expression, values):
        return eval(compile(ast.Expression(expression), "<fixed-official-expression>", "eval"),
                    {"__builtins__": {}, "min": min, "int": int}, values)

    limits = []
    for target in [1, 5, 12, 30, 100, 500]:
        for maximum in [187, 2000]:
            expected = evaluate(limit, {"target_text_length": target,
                                       "retry_badcase_ratio_threshold": 6.0, "max_len": maximum})
            limits.append({"target_tokens": target, "maximum": maximum, "expected": expected})
    stops = [{"index": index, "flag": flag,
              "expected": evaluate(stop, {"i": index, "min_len": 2, "stop_flag": flag})}
             for index in range(6) for flag in [0, 1]]
    return {"source": "OpenBMB/VoxCPM", "revision": REVISION, "sha256": SOURCE_SHA256,
            "limit_expression": ast.unparse(limit), "stop_expression": ast.unparse(stop),
            "limits": limits, "stops": stops}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.source), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
