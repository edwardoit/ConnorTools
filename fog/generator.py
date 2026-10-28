#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import random
import secrets

#hello+o*k is only an old programmers friend probably its forget is the most emotional damage from AI era
VARIANTS: list[str] = ["hell" + "o" * k for k in range(1, 31)]
LEN_PREFIX_BITS = 32


def generate_carrier(n: int, wall_seed: int | None = None) -> list[str]:

    if wall_seed is None:
        rng = secrets.SystemRandom()
    else:
        #seed compatibility
        rng = random.Random(wall_seed)

    wall = []

    for i in range(n):
        token = rng.choice(VARIANTS)
        wall.append(token)

    return wall


def frequencies(wall: list[str]) -> dict[str, int]:
    #hashmap<k,v>
    f: dict[str, int] = {}
    for w in wall:
        f[w] = f.get(w, 0) + 1
    return f


def salt_from_wall(wall: list[str]) -> bytes:
    #frequency vector
    f = frequencies(wall)
    raw = b""
    for v in VARIANTS:
        count = f.get(v, 0)
        encoded = count.to_bytes(4, "big")
        raw += encoded
    return hashlib.sha256(raw).digest()



def _permutation(key: str, salt: bytes, n: int) -> list[int]:
    idx = list(range(n))

    for i in range(n - 1, 0, -1):
        counter = 0
        bound = i + 1
        limit = 2**256 - (2**256 % bound)

        while True:
            data = salt + key.encode("utf-8")
            data += i.to_bytes(8, "big")
            data += counter.to_bytes(8, "big")

            x = int(hashlib.sha256(data).hexdigest(), 16)

            if x < limit:
                j = x % bound
                break

            counter += 1

        idx[i], idx[j] = idx[j], idx[i]

    return idx


def pairing(wall: list[str], key: str) -> list[tuple[int, int]]:
    n = len(wall)

    # Frequencies salt derivation
    salt = salt_from_wall(wall)

    # Generate a deterministic permutation using key and salt
    idx = _permutation(key, salt, n)

    pairs = []

    for i in range(0, n - 1, 2):
        a = idx[i]
        b = idx[i + 1]

        pairs.append((a, b))

    return pairs


def capacity(wall: list[str], key: str) -> int:
    return sum(1 for a, b in pairing(wall, key)
               if len(wall[a]) != len(wall[b]))


def _bytes_to_bits(data: bytes) -> list[int]:
    return [
        (b // (2 ** i)) % 2
        for b in data
        for i in range(7, -1, -1)
    ]


def _bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for b in bits[i:i + 8]:
            v = (v << 1) | b
        out.append(v)
    return bytes(out)


def _message_to_bits(msg: bytes) -> list[int]:
    #first 32bit are allocated to inject message payload length
    return _bytes_to_bits(len(msg).to_bytes(4, "big")) + _bytes_to_bits(msg)


def insert(wall: list[str], key: str, message: bytes) -> list[str]:
    wall = list(wall)
    bits = iter(_message_to_bits(message))
    exhausted = True
    ##pairing candidate (a,b);
    for a, b in pairing(wall, key):
        if len(wall[a]) == len(wall[b]):
            continue
        try:
            bit = next(bits)
        except StopIteration:
            exhausted = True
            break
        exhausted = False
        ##swapping allocation of information
        current = 0 if len(wall[a]) > len(wall[b]) else 1
        if current != bit:
            wall[a], wall[b] = wall[b], wall[a]
    remaining = sum(1 for _ in bits)
    if not exhausted and remaining > 0:
        raise ValueError(f"Insufficient capacity: {remaining} bits missing.")
    return wall


def read_wall(wall: list[str], key: str) -> bytes:
    pairs = pairing(wall, key)

    def take(n_bits: int, start: int) -> tuple[list[int], int]:
        bits, i = [], start
        while len(bits) < n_bits:
            if i >= len(pairs):
                raise ValueError("Wall too short for the declared payload.")
            a, b = pairs[i]; i += 1
            if len(wall[a]) == len(wall[b]):
                continue
            bits.append(0 if len(wall[a]) > len(wall[b]) else 1)
        return bits, i

    prefix, pos = take(LEN_PREFIX_BITS, 0)
    length = int.from_bytes(_bits_to_bytes(prefix), "big")
    body, _ = take(length * 8, pos)
    return _bits_to_bytes(body)

#main use method
def generate_wall(message: bytes, key: str,
                  n: int | None = None,
                  wall_seed: int | None = None,
                  margin: float = 1.1) -> list[str]:
    needed = LEN_PREFIX_BITS + len(message) * 8
    if n is None:
        n = max(200, int(needed / (29 / 60) * margin) + 200)
    wall = generate_carrier(n, wall_seed)
    while capacity(wall, key) < needed:
        n = int(n * 1.2) + 500
        wall = generate_carrier(n, wall_seed)
    return insert(wall, key, message)


if __name__ == "__main__":
    key = "test-key"
    msg = "fog demo: words placed at random, message only in the order".encode()
    base = generate_carrier(5000, wall_seed=1)
    wall = insert(base, key, msg)
    assert frequencies(base) == frequencies(wall), "frequencies altered!"
    assert salt_from_wall(base) == salt_from_wall(wall), "salt not invariant!"
    assert read_wall(wall, key) == msg, "round-trip failed!"
    print("OK:", len(wall), "tokens | capacity", capacity(wall, key),
          "bits | frequencies+salt invariant | round-trip verified")
