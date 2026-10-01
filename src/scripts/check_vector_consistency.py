"""Check that the Qdrant vectors were made by the embedding model configured now.

Re-embeds the stored `text` of a few points and compares it with the stored
vector. Same model => cosine ~1.0. A different model => cosine ~0, and
semantic doctor search silently degrades to noise (only filters still work).

    cd src && python -m scripts.check_vector_consistency
"""

import math
import sys

from helpers.bootstrap import build_services


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main(samples: int = 5) -> int:
    s = build_services(connect_mcp=False)
    try:
        name = s.config.VECTOR_DB_COLLECTION_NAME
        points, _ = s.vector_db.client.scroll(name, limit=samples, with_vectors=True, with_payload=True)
        scores = []
        for p in points:
            vec = s.embedding.embed_text(p.payload.get("text", ""))
            if not vec or len(vec) != len(p.vector):
                print(f"✗ dimension mismatch: stored={len(p.vector)} current={len(vec) if vec else None}")
                return 2
            scores.append(cosine(p.vector, vec))
        avg = sum(scores) / len(scores)
        print(f"Embedding model : {s.config.EMBEDDING_BACKEND} / {s.config.EMBEDDING_MODEL_ID} ({s.config.EMBEDDING_MODEL_SIZE}d)")
        print(f"Mean cosine     : {avg:.3f} over {len(scores)} points")
        if avg > 0.95:
            print("✓ Vectors match the configured embedding model.")
            return 0
        print("✗ Vectors were created with a DIFFERENT embedding model. Re-run ingestion with the "
              "current model (python -m scripts.ingest_vezeeta --reset) or switch EMBEDDING_* back.")
        return 1
    finally:
        s.close()


if __name__ == "__main__":
    sys.exit(main())
