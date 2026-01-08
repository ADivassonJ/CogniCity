import random

D1, D2, D3 = 1.65, 8.48, 17.46

def mode_probabilities(d: float):
    """Devuelve (P_walk, P_public, P_private) para distancia d en km."""
    if d <= D1:
        return (1.0, 0.0, 0.0)
    if d < D2:
        t1 = (d - D1) / (D2 - D1)          # 0..1
        return (1.0 - t1, t1, 0.0)
    if d < D3:
        t2 = (d - D2) / (D3 - D2)          # 0..1
        return (0.0, 1.0 - t2, t2)
    return (0.0, 0.0, 1.0)

def sample_mode(d: float, rng=random.random):
    """Devuelve una elección concreta ('walk','public','private') vía Monte Carlo."""
    pw, ppt, ppc = mode_probabilities(d)
    u = rng()
    if u < pw:
        return "walk"
    if u < pw + ppt:
        return "public"
    return "private"

# Ejemplos rápidos:
for d in [1.0, 1.65, 2.333, 8.48, 13.868, 17.46, 20.0]:
    pw, ppt, ppc = mode_probabilities(d)
    print(d, pw, ppt, ppc, "->", sample_mode(d))
    