"""Rooted unordered binary trees with fixed integer physical leaf labels."""
import hashlib
import json
from typing import TypeAlias

import numpy as np

Tree: TypeAlias = int | tuple['Tree', 'Tree']


def canonical(tree) -> Tree:
    if isinstance(tree, (int, np.integer)):
        return int(tree)
    if not isinstance(tree, (list, tuple)) or len(tree) != 2:
        raise ValueError('Expected a leaf or exactly two children')
    a, b = map(canonical, tree)
    return tuple(sorted((a, b), key=serialize))


def serialize(tree: Tree) -> str:
    return json.dumps(tree, separators=(',', ':'))


def leaves(tree: Tree) -> list[int]:
    return [tree] if isinstance(tree, int) else leaves(tree[0]) + leaves(tree[1])


def validate(tree: Tree, n: int) -> None:
    if sorted(leaves(canonical(tree))) != list(range(n)):
        raise ValueError('Leaves must contain every physical qubit exactly once')


def tree_hash(tree: Tree) -> str:
    return hashlib.sha256(serialize(canonical(tree)).encode()).hexdigest()


def random_tree(n: int, rng) -> Tree:
    forest = list(range(n))
    while len(forest) > 1:
        i, j = sorted(rng.choice(len(forest), 2, replace=False), reverse=True)
        a, b = forest.pop(i), forest.pop(j)
        forest.append(canonical((a, b)))
    return forest[0]


def balanced(ids) -> Tree:
    return int(ids[0]) if len(ids) == 1 else canonical((balanced(ids[:len(ids)//2]), balanced(ids[len(ids)//2:])))


def comb(ids) -> Tree:
    result = int(ids[0])
    for q in ids[1:]:
        result = canonical((result, int(q)))
    return result


def neighbors(tree: Tree) -> list[Tree]:
    if isinstance(tree, int):
        return []
    a, b = tree
    candidates = [(x, b) for x in neighbors(a)] + [(a, x) for x in neighbors(b)]
    if not isinstance(a, int):
        x, y = a
        candidates.extend([(x, (y, b)), (y, (x, b))])
    if not isinstance(b, int):
        x, y = b
        candidates.extend([((a, x), y), ((a, y), x)])
    unique = {tree_hash(canonical(t)): canonical(t) for t in candidates}
    unique.pop(tree_hash(tree), None)
    return list(unique.values())


def candidate_pool(n: int, size: int, seed: int, scent: Tree) -> list[Tree]:
    if size < 1:
        raise ValueError('Pool size must be positive')
    rng = np.random.default_rng(seed)
    pool = {}
    for t in [scent, balanced(list(range(n))), comb(list(range(n))), *neighbors(scent)]:
        validate(t, n)
        pool.setdefault(tree_hash(t), canonical(t))
        if len(pool) == size:
            return list(pool.values())
    for _ in range(size * 1000):
        t = random_tree(n, rng)
        pool.setdefault(tree_hash(t), t)
        if len(pool) == size:
            return list(pool.values())
    raise ValueError('Unable to generate requested number of unique trees')


def features(tree: Tree) -> np.ndarray:
    n = len(leaves(tree))
    validate(tree, n)
    paths = {}
    def walk(t, path):
        if isinstance(t, int):
            paths[t] = path
        else:
            walk(t[0], path + (0,))
            walk(t[1], path + (1,))
    walk(tree, ())
    relations = []
    for i in range(n):
        for j in range(i + 1, n):
            common = 0
            for a, b in zip(paths[i], paths[j]):
                if a != b:
                    break
                common += 1
            relations.append(common / n)
    return np.array(relations + [len(paths[i]) / n for i in range(n)])
