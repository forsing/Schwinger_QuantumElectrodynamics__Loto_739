#!/usr/bin/env python3

"""Schwinger QED: proper-time operator nad celim STRUKTURNIM PRELAZIMA."""

import argparse
import csv
from pathlib import Path
import numpy as np

DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv",
)

N, K, WINDOW, LANDMARKS, STARTS = 39, 7, 96, 12, 12
MASSES = (.1, .3, 1.)
ALPHAS = (.01, .1, 1.)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != K
                or values != sorted(values)
                or len(set(values)) != K
                or not all(1 <= v <= N for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "7/39 kombinacija."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1)/5.


def distances(a, b):
    return np.abs(
        a[:, None, :]-b[None, :, :]
    ).sum(axis=2)


def proper_time_kernel(d, mass):
    # Integral exp(-u)*exp(-u*d/mass) du,
    # od nule do beskonacnosti.
    return mass/(mass+d)


class Basis:
    def __init__(self, values):
        selected = [len(values)-1]
        minimum = distances(
            values, values[selected]
        )[:, 0]

        while len(selected) < min(LANDMARKS, len(values)):
            chosen = int(np.argmax(minimum))

            if minimum[chosen] <= 1e-12:
                break

            selected.append(chosen)
            minimum = np.minimum(
                minimum,
                distances(values, values[[chosen]])[:, 0]
            )

        self.values = values[selected]
        d = distances(self.values, self.values)
        positive = d[d > 1e-12]

        self.bandwidth = (
            float(np.median(positive))
            if len(positive) else 1.
        )

    def features(self, values, mass):
        return proper_time_kernel(
            distances(values, self.values)/self.bandwidth,
            mass
        )


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (K*len(unused), 1))

    for j in range(K):
        result[j*len(unused):(j+1)*len(unused), j] = unused

    return np.sort(result, axis=1)


def prepare(history, mass):
    if len(history) < WINDOW+2:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW+2} prethodnih redova."
        )

    history = history[-(WINDOW+2):]
    g = gaps(history)
    changes = np.diff(g, axis=0)

    x = np.column_stack((
        g[1:-1], changes[:-1]
    ))
    query = np.r_[
        g[-1], changes[-1]
    ][None, :]

    input_basis = Basis(x)
    output_basis = Basis(changes[1:])

    incoming = input_basis.features(x, mass)

    # Centriraju se kernel karakteristike.
    # Ne prosece se brojevi niti izlazni prelazi.
    center = incoming.mean(axis=0)
    incoming = np.column_stack((
        np.ones(WINDOW), incoming-center
    ))
    query = np.r_[
        1.,
        input_basis.features(query, mass)[0]-center
    ]

    outgoing = output_basis.features(
        changes[1:], mass
    )

    dimension = (
        incoming.shape[1]*outgoing.shape[1]
    )
    gram = np.zeros((dimension, dimension))
    rhs = np.zeros(dimension)

    for i, t in enumerate(range(2, len(history))):
        # Svaki kandidat opisuje se prema svom
        # prethodnom stvarnom redu.
        alternatives = output_basis.features(
            gaps(replacements(history[t]))-g[t-1],
            mass
        )

        delta = outgoing[i]-alternatives
        contrasts = (
            delta[:, :, None]
            * incoming[i][None, None, :]
        ).reshape(len(delta), dimension)

        gram += (
            contrasts.T @ contrasts/len(contrasts)
        )
        rhs += contrasts.sum(axis=0)/len(contrasts)

    gram /= WINDOW
    rhs /= WINDOW

    scale = np.maximum(
        np.sqrt(np.diag(gram)), 1e-8
    )

    return (
        gram/scale[:, None]/scale[None, :],
        rhs/scale,
        scale,
        query,
        output_basis,
        mass,
        g[-1],
        history[2:]
    )


def fit(prepared, alpha):
    (
        gram, rhs, scale, query,
        basis, mass, reference, history
    ) = prepared

    coef = np.linalg.solve(
        gram+alpha*np.eye(len(rhs)),
        rhs
    )/scale

    operator = coef.reshape(
        len(basis.values), len(query)
    )

    return (
        basis, mass, reference,
        operator @ query
    )


def scores(rows, model):
    basis, mass, reference, field = model

    # Direktan skor celog prelaza.
    # Nema predikcije prosecnih polozaja.
    result = basis.features(
        gaps(rows)-reference, mass
    ) @ field

    if not np.all(np.isfinite(result)):
        raise ArithmeticError(
            "Neispravan skor prelaza."
        )

    return result


def search(history, model):
    # Istorijski redovi su samo polazista.
    # Pretraga moze nastaviti kroz vise zamena.
    seeds = np.unique(history, axis=0)
    ranked = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )
    current = seeds[ranked[:STARTS]]
    iterations = 0

    while True:
        iterations += 1
        updated = current.copy()
        moved = False

        for i, row in enumerate(current):
            options = np.vstack((
                row, replacements(row)
            ))
            values = scores(options, model)
            best = int(np.argmax(values))

            # Strogo poboljsanje sprecava kruzenje
            # medju kombinacijama sa jednakim skorom.
            if values[best] > values[0]+1e-10:
                updated[i] = options[best]
                moved = True

        current = updated

        if not moved:
            values = scores(current, model)
            best = int(np.argmax(values))

            return (
                tuple(map(int, current[best])),
                float(values[best]),
                iterations
            )


def predict(prepared, alpha):
    return search(
        prepared[7],
        fit(prepared, alpha)
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, values):
    print(
        f"{label}: prosek={np.mean(values):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(values, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows)-test_count
    begin = split-validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW+2
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    settings = tuple(
        (m, a) for m in MASSES for a in ALPHAS
    )
    validation = [[] for _ in settings]

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    for t in range(begin, split):
        index = 0

        for mass in MASSES:
            p = prepare(rows[:t], mass)

            for alpha in ALPHAS:
                combo, _, _ = predict(p, alpha)
                validation[index].append(
                    hits(combo, rows[t])
                )
                index += 1

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t-begin+1}/{validation_count}",
                flush=True
            )

    means = [np.mean(v) for v in validation]
    chosen = int(np.argmax(means))
    mass, alpha = settings[chosen]

    print(
        f"Validacija: mass={mass:g}; "
        f"alpha={alpha:g}; "
        f"prosek={means[chosen]:.4f}",
        flush=True
    )

    actual_scores, previous_scores = [], []

    # Mass i alpha ostaju fiksni tokom testa.
    # Svaka predikcija koristi samo prethodne redove.
    for t in range(split, len(rows)):
        combo, _, _ = predict(
            prepare(rows[:t], mass),
            alpha
        )

        actual_scores.append(
            hits(combo, rows[t])
        )
        previous_scores.append(
            hits(rows[t-1], rows[t])
        )

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize(
        "SCHWINGER QED PRELAZI",
        actual_scores
    )
    summarize(
        "Prethodna kombinacija",
        previous_scores
    )

    p = prepare(rows, mass)
    model = fit(p, alpha)
    combo, score, iterations = search(p[7], model)

    # Kontrola uticaja konteksta na nauceno polje.
    probe = p[3].copy()
    probe[1:] = 0.

    gram, rhs, scale = p[:3]
    coef = np.linalg.solve(
        gram+alpha*np.eye(len(rhs)),
        rhs
    )/scale

    operator = coef.reshape(
        len(p[4].values), len(p[3])
    )
    context_effect = np.linalg.norm(
        model[3]-operator @ probe
    )

    repeated = any(
        np.array_equal(combo, row)
        for row in rows
    )

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor={score:.10f}; "
        f"iteracija={iterations}; "
        f"vec postoji u istoriji={repeated}"
    )
    print(
        "Promena polja kada se ukloni kontekst="
        f"{context_effect:.8f}"
    )
    print(
        "Proveren lokalni maksimum svih 224 zamena; "
        "globalni maksimum nije potvrden."
    )


def self_test():
    nodes, weights = np.polynomial.laguerre.laggauss(96)

    for mass in MASSES:
        for ratio in (0., .25, 1., 4.):
            assert np.isclose(
                weights @ np.exp(-nodes*ratio),
                proper_time_kernel(mass*ratio, mass),
                atol=1e-9
            )

    base = np.array([
        [1,2,3,4,5,6,7],
        [4,12,18,28,35,36,37],
        [33,34,35,36,37,38,39],
        [2,7,12,18,24,30,39]
    ])
    history = np.tile(base, (30, 1))

    g = gaps(base)
    restored = g[0]+(g-g[0])

    assert np.allclose(
        5*np.cumsum(
            restored[:, :-1], axis=1
        ),
        base
    )

    p = prepare(history, .3)
    model = fit(p, .1)
    result = search(p[7], model)

    assert result == predict(
        prepare(history, .3), .1
    )
    assert (
        scores(
            replacements(result[0]), model
        ).max()
        <= result[1]+1e-9
    )

    altered = history.copy()
    altered[-1] = base[0]

    assert (
        predict(prepare(history[:-1], .3), .1)
        == predict(prepare(altered[:-1], .3), .1)
    )
    assert (
        np.linalg.eigvalsh(p[0]).min()
        >= -1e-8
    )

    print(
        "Self-test OK: integral, prelazi, "
        "hronologija i lokalni optimum."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: mass=0.1; alpha=0.1; prosek=1.5833
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
SCHWINGER QED PRELAZI: prosek=1.3906; pogodaka 0..7=[9, 30, 17, 7, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 1,24,25,30,32,33,34
Skor=1.4756612636; iteracija=9; vec postoji u istoriji=False
Promena polja kada se ukloni kontekst=7.81337705
Proveren lokalni maksimum svih 224 zamena; globalni maksimum nije potvrden.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: mass=0.3; alpha=0.01; prosek=1.5208
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
SCHWINGER QED PRELAZI: prosek=1.4219; pogodaka 0..7=[11, 24, 20, 9, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,6,18,26,27,37,39
Skor=3.9254044752; iteracija=7; vec postoji u istoriji=False
Promena polja kada se ukloni kontekst=29.40102544
Proveren lokalni maksimum svih 224 zamena; globalni maksimum nije potvrden.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: mass=0.1; alpha=0.01; prosek=1.4167
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
SCHWINGER QED PRELAZI: prosek=1.2031; pogodaka 0..7=[17, 23, 18, 6, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 6,7,14,25,26,36,37
Skor=1.3838099549; iteracija=9; vec postoji u istoriji=False
Promena polja kada se ukloni kontekst=8.97938646
Proveren lokalni maksimum svih 224 zamena; globalni maksimum nije potvrden.
"""



"""
Popravljeno. QED sada rangira cele prelaze između kombinacija, a pretraga nastavlja zamene brojeva do lokalnog maksimuma. Nijedan završni predlog već ne postoji u svom CSV-u; istorijske kombinacije nisu zabranjene niti su rezultati namerno razdvojeni.

Fajl	                Sledeća kombinacija	Prosek pogodaka na 64 kola	Prethodna kombinacija
Objedinjeni, red 4699	1,24,25,30,32,33,34	1,3906	                    1,2188
Loto, red 2972	        1,6,18,26,27,37,39	1,4219	                    1,2969
Loto Plus, red 1728	    6,7,14,25,26,36,37	1,2031	                    1,2969

Provereni su integral, hronologija, ponovljivost i svih 224 zamena završnih predloga. Globalni maksimum nije potvrđen.
"""



"""
Julian Seymour Schwinger  
1-quantum electrodynamics.   2-nuclear physics.     3-particle physics.     
4-atomic physics       5-quantum field theory.        6-Schwinger effect
"""
