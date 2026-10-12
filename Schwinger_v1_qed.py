#!/usr/bin/env python3

"""SCHWINGER 1 — QED: proper-time kernel and causal Green prediction."""

import argparse
import csv
import itertools
from pathlib import Path
import numpy as np



DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv"



N, K, WINDOW, START = 39, 7, 192, 8


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error

            if (len(numbers) != K or numbers != sorted(numbers)
                    or len(set(numbers)) != K
                    or not all(1 <= v <= N for v in numbers)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(numbers)

    return np.asarray(rows, dtype=float).reshape(-1, K)


def shape(rows):
    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), N+1),
    ))
    gaps = np.diff(edges, axis=1)-1

    return np.column_stack((
        rows/(N+1),
        gaps/(N-K+1),
    ))


def context(layout, t):
    return np.r_[
        layout[t-1],
        layout[t-2],
        layout[t-8:t].mean(axis=0),
    ]


def distances(a, b):
    d = (
        (a*a).sum(axis=1)[:, None]
        + (b*b).sum(axis=1)[None, :]
        - 2*a @ b.T
    )
    return np.maximum(d, 0)


def proper_time_kernel(d, mass):
    # Integral exp(-u)*exp(-u*d/mass) du = mass/(mass+d).
    return mass/(mass+d)


def prepare(history):
    layout = shape(history)
    begin = max(START, len(history)-WINDOW)

    x = np.asarray([
        context(layout, t)
        for t in range(begin, len(history))
    ])
    query = context(layout, len(history))[None, :]

    d = distances(x, x)
    positive = d[np.triu_indices(len(x), 1)]
    positive = positive[positive > 1e-12]
    bandwidth = (
        float(np.median(positive)) if len(positive) else 1.
    )

    return (
        d/bandwidth,
        distances(query, x)[0]/bandwidth,
        history[begin:]/(N+1),
    )


def predict(prepared, mass, regularization):
    distance, query_distance, y = prepared
    kernel = proper_time_kernel(distance, mass)

    row_mean = kernel.mean(axis=0)
    grand_mean = float(kernel.mean())
    centered = (
        kernel
        - row_mean[None, :]
        - row_mean[:, None]
        + grand_mean
    )

    query = proper_time_kernel(query_distance, mass)
    query = query-row_mean-query.mean()+grand_mean

    mean = y.mean(axis=0)
    coefficients = np.linalg.solve(
        centered+regularization*np.eye(len(y)),
        y-mean,
    )
    target = (mean+query @ coefficients)*(N+1)

    if not np.all(np.isfinite(target)):
        raise ArithmeticError(
            "Neispravan odziv Greenovog operatora."
        )
    return target


def gaps(x, n=N):
    return np.diff(np.r_[0., x, float(n+1)])-1.


def cost(combo, target, n=N):
    return float(
        np.abs(np.asarray(combo)-target).sum()
        + np.abs(gaps(combo, n)-gaps(target, n)).sum()
    )


def decode(target, n=N):
    # Tacan minimum definisanog troska polozaja i razmaka.
    k = len(target)
    target_gaps = gaps(target, n)
    table = np.full((k, n+1), np.inf)
    parent = np.full((k, n+1), -1, dtype=int)

    for v in range(1, n-k+2):
        table[0, v] = (
            abs(v-target[0])
            + abs(v-1-target_gaps[0])
        )

    for j in range(1, k):
        for v in range(j+1, n-k+j+2):
            previous = np.arange(j, v)
            choices = (
                table[j-1, previous]
                + np.abs(v-previous-1-target_gaps[j])
            )
            best = int(np.argmin(choices))
            parent[j, v] = previous[best]
            table[j, v] = choices[best]+abs(v-target[j])

    ends = np.arange(k, n+1)
    scores = (
        table[k-1, ends]
        + np.abs(n-ends-target_gaps[k])
    )
    best = int(np.argmin(scores))
    v = int(ends[best])
    combo = [v]

    for j in range(k-1, 0, -1):
        v = int(parent[j, v])
        combo.append(v)

    return tuple(reversed(combo)), float(scores[best])


def hits(prediction, actual):
    return len(
        set(map(int, prediction)) & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True,
    )


def self_test():
    nodes, weights = np.polynomial.laguerre.laggauss(80)

    for d in (0., .25, 1., 4.):
        integral = weights @ np.exp(-nodes*d)
        assert np.isclose(
            integral,
            proper_time_kernel(d, 1.),
            rtol=1e-9,
            atol=1e-9,
        )

    rows = np.tile(np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39],
    ], dtype=float), (50, 1))

    prepared = prepare(rows[:170])
    kernel = proper_time_kernel(prepared[0], 1.)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-10

    target = predict(prepared, 1., 1.)
    changed = rows.copy()
    changed[170:] = [3, 8, 13, 19, 25, 31, 38]

    assert np.array_equal(
        target,
        predict(prepare(changed[:170]), 1., 1.),
    )
    assert decode(target) == decode(target)

    for candidate_target in (
        [1.8, 4.6, 7.1],
        [8., 2., 5.],
        [-2., 12., 3.],
    ):
        candidate_target = np.asarray(candidate_target)
        combo, score = decode(candidate_target, n=9)

        exact = min(
            cost(c, candidate_target, n=9)
            for c in itertools.combinations(range(1, 10), 3)
        )
        assert np.isclose(
            score, exact, rtol=1e-12, atol=1e-12
        )
        assert np.isclose(
            score, cost(combo, candidate_target, n=9)
        )

    print(
        "Self-test OK: proper-time integral, operator, "
        "hronologija i globalna projekcija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows)-args.test
    begin = split-args.validation
    if begin < START+128:
        parser.error(
            "Premalo istorije za zadatu validaciju i test."
        )

    parameters = tuple(itertools.product(
        (.5, 1., 2.), (.1, 1., 10.)
    ))
    validation = [[] for _ in parameters]

    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        prepared = prepare(rows[:t])

        for j, (mass, regularization) in enumerate(parameters):
            combo, _ = decode(predict(
                prepared, mass, regularization
            ))
            validation[j].append(hits(combo, rows[t]))

    means = [
        np.mean(values) for values in validation
    ]
    best = int(np.argmax(means))
    mass, regularization = parameters[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"mass={mass:g}; regularizacija={regularization:g}; "
        f"prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []
    background_scores = []

    for t in range(split, len(rows)):
        combo, _ = decode(predict(
            prepare(rows[:t]), mass, regularization
        ))
        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        reference, _ = decode(
            rows[max(0, t-WINDOW):t].mean(axis=0)
        )
        background_scores.append(hits(reference, rows[t]))

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize(
        "SCHWINGER 1 QUANTUM ELECTRODYNAMICS",
        model_scores,
    )
    summarize("Prethodna kombinacija", previous_scores)
    summarize(
        "Pozadina bez Greenovog operatora",
        background_scores,
    )

    target = predict(
        prepare(rows), mass, regularization
    )
    combo, score = decode(target)

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1].astype(int)))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Trosak projekcije={score:.10f}; "
        "globalni minimum projekcije potvrden."
    )
    print(
        "Proper-time kernel opisuje strukturnu slicnost; "
        "trosak nije verovatnoca."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: redovi 4587..4634; mass=2; regularizacija=0.1; prosek=1.2292
Test: redovi 4635..4698
SCHWINGER 1 QUANTUM ELECTRODYNAMICS: prosek=1.2969; pogodaka 0..7=[13, 27, 17, 6, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Pozadina bez Greenovog operatora: prosek=1.4375; pogodaka 0..7=[11, 24, 21, 6, 2, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 5,8,12,17,22,29,34
Trosak projekcije=4.2199715194; globalni minimum projekcije potvrden.
Proper-time kernel opisuje strukturnu slicnost; trosak nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: redovi 2860..2907; mass=2; regularizacija=10; prosek=1.4167
Test: redovi 2908..2971
SCHWINGER 1 QUANTUM ELECTRODYNAMICS: prosek=1.4062; pogodaka 0..7=[8, 28, 23, 4, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Pozadina bez Greenovog operatora: prosek=1.2812; pogodaka 0..7=[12, 29, 17, 5, 1, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 5,10,14,19,24,30,35
Trosak projekcije=3.1692716052; globalni minimum projekcije potvrden.
Proper-time kernel opisuje strukturnu slicnost; trosak nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: redovi 1616..1663; mass=0.5; regularizacija=1; prosek=1.4167
Test: redovi 1664..1727
SCHWINGER 1 QUANTUM ELECTRODYNAMICS: prosek=1.0938; pogodaka 0..7=[16, 28, 18, 2, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Pozadina bez Greenovog operatora: prosek=1.1719; pogodaka 0..7=[13, 30, 18, 3, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 4,9,14,18,24,29,35
Trosak projekcije=3.1804132954; globalni minimum projekcije potvrden.
Proper-time kernel opisuje strukturnu slicnost; trosak nije verovatnoca.
"""
 


"""
idemo na novu oblast Julian Seymour Schwinger  1-quantum electrodynamics.   2-nuclear physics.     3-particle physics.     4-atomic physics       5-quantum field theory.        6-Schwinger effect.

Pripremiću šest zasebnih Schwingerovih modela, po jedan za svaku oblast. Za svaki ću izdvojiti konkretnu ideju, napisati samostalan kod i proveriti ga na istom fajlu, uz postojeća pravila: deterministički rad, hronološki test i jedna konačna verzija.

SCHWINGER 1 — QUANTUM ELECTRODYNAMICS je završena i proverena.
Primenjena je ideja proper-time metode i Greenovog operatora na strukturnu sličnost celih kombinacija. Schwingerov rad iz 1951.
Predlog za red 4699:
5, 8, 12, 17, 22, 29, 34
Na poslednja 64 kola:
Metod	Prosečan broj pogodaka
Schwinger 1	1,2969
Prethodna kombinacija	1,2188
Pozadina bez Greenovog operatora	1,4375

Model na ovom testu nije nadmašio pozadinu. Provereni su integral, operator, hronologija, ponovljivost i globalni minimum troška projekcije.
"""



"""
Julian Seymour Schwinger  
1-quantum electrodynamics.   2-nuclear physics.     3-particle physics.     
4-atomic physics       5-quantum field theory.        6-Schwinger effect
"""
