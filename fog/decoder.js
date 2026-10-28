const NUM_VARIANTS = 30;
const LEN_PREFIX_BITS = 32;


//generated


async function sha256(bytes) {
  const d = await crypto.subtle.digest("SHA-256", bytes);
  return new Uint8Array(d);
}

function concatBytes(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function variantK(tok) {
  if (!tok.startsWith("hell")) return null;
  const k = tok.length - 4;
  if (k < 1 || k > NUM_VARIANTS) return null;
  for (let i = 4; i < tok.length; i++) if (tok[i] !== "o") return null;
  return k;
}

async function saltFromWall(tokens) {
  const counts = new Uint32Array(NUM_VARIANTS);
  for (const t of tokens) {
    const k = variantK(t);
    if (k !== null) counts[k - 1] += 1;
  }
  const raw = new Uint8Array(NUM_VARIANTS * 4);
  const dv = new DataView(raw.buffer);
  for (let k = 0; k < NUM_VARIANTS; k++) dv.setUint32(k * 4, counts[k], false);
  return await sha256(raw);
}

class KeyStream {
  constructor(preBytes) {
    this.pre = preBytes;
    this.ctr = 0n;
    this.buf = new Uint8Array(0);
    this.pos = 0;
  }

  async _refill() {
    const ctrBytes = new Uint8Array(8);
    new DataView(ctrBytes.buffer).setBigUint64(0, this.ctr, false);
    this.buf = await sha256(concatBytes(this.pre, ctrBytes));
    this.ctr += 1n;
    this.pos = 0;
  }

  async _nextBytes(k) {
    const out = new Uint8Array(k);
    let off = 0;
    while (off < k) {
      if (this.pos >= this.buf.length) await this._refill();
      const take = Math.min(k - off, this.buf.length - this.pos);
      out.set(this.buf.subarray(this.pos, this.pos + take), off);
      off += take;
      this.pos += take;
    }
    return out;
  }

  async randbelow(bound) {
    if (bound <= 1) return 0;
    const B = BigInt(bound);
    const TWO64 = 1n << 64n;
    const limit = TWO64 - (TWO64 % B);
    for (;;) {
      const bytes = await this._nextBytes(8);
      const x = new DataView(bytes.buffer).getBigUint64(0, false);
      if (x < limit) return Number(x % B);
    }
  }
}

async function permutation(key, salt, n) {
  const pre = concatBytes(salt, new TextEncoder().encode(key));
  const ks = new KeyStream(pre);
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = await ks.randbelow(i + 1);
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return idx;
}

async function pairing(tokens, key) {
  const n = tokens.length;
  const salt = await saltFromWall(tokens);
  const idx = await permutation(key, salt, n);
  const pairs = [];
  for (let i = 0; i + 1 < n; i += 2) pairs.push([idx[i], idx[i + 1]]);
  return pairs;
}

async function extractBits(tokens, key) {
  const L = tokens.map((t) => t.length);
  const pairs = await pairing(tokens, key);
  const bits = [];
  for (const [a, b] of pairs) {
    if (L[a] === L[b]) continue;
    bits.push(L[a] > L[b] ? 0 : 1);
  }
  return bits;
}

function readUint32(bits, off) {
  let v = 0;
  for (let i = 0; i < 32; i++) v = v * 2 + bits[off + i];
  return v;
}

function bitsToBytes(bits) {
  const out = new Uint8Array(bits.length >> 3);
  for (let i = 0; i + 8 <= bits.length; i += 8) {
    let v = 0;
    for (let b = 0; b < 8; b++) v = (v << 1) | bits[i + b];
    out[i >> 3] = v;
  }
  return out;
}

async function decodeWall(text, password) {
  const tokens = text.trim().split(/\s+/).filter(Boolean);
  const bits = await extractBits(tokens, password);
  if (bits.length < LEN_PREFIX_BITS) return { real: false, text: "" };

  const length = readUint32(bits, 0);
  const maxBytes = Math.floor((bits.length - LEN_PREFIX_BITS) / 8);

  if (length <= maxBytes) {
    const body = bitsToBytes(bits.slice(LEN_PREFIX_BITS, LEN_PREFIX_BITS + length * 8));
    try {
      const msg = new TextDecoder("utf-8", { fatal: true }).decode(body);
      return { real: true, text: msg };
    } catch {
      void 0;
    }
  }

  const l = Math.min(length, maxBytes);
  const body = bitsToBytes(bits.slice(LEN_PREFIX_BITS, LEN_PREFIX_BITS + l * 8));
  const nonsense = new TextDecoder("utf-8", { fatal: false }).decode(body);
  return { real: false, text: nonsense };
}

export { decodeWall, extractBits, pairing, saltFromWall };
