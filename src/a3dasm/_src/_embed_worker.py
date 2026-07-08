"""Out-of-process embedding worker (bge-small via fastembed).

Runs in an ephemeral uv environment resolved with numpy<2 so that
onnxruntime's Intel-macOS wheels (NumPy-1.x builds) can load even
when the host project pins numpy>=2.

Protocol: read JSON {"texts": [...]} on stdin, write JSON
{"vectors": [[...], ...]} on stdout. Errors → nonzero exit with the
message on stderr.
"""
import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    texts = payload["texts"]
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    vectors = [[float(x) for x in v] for v in model.embed(texts)]
    json.dump({"vectors": vectors}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
