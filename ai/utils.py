import math

def cosine_similarity(v1, v2):
    # Compute the numerator and vector norms for cosine similarity.
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = math.sqrt(sum(x * x for x in v1))
    magnitude2 = math.sqrt(sum(x * x for x in v2))
    # If either vector is all zeros, similarity is defined as 0 to avoid division by zero.
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)