// Fiore-lab, Expert Atlas.
// Keyed sampling in better C: the token a sampled request draws at each position, the same on every rank and in every
// run. The algorithm is TensorFold's (zig/src/core/lanes/sampling.zig), which follows Python's exact_sampling.py: a
// splitmix64 key per (seed, position, token), libm's exp and log, numpy's pairwise float64 sum and a Gumbel argmax over
// the kept candidates. The draws must match it bit for bit, so every step keeps its order of operations, and the build
// compiles this file with -ffp-contract=off (no fused multiply-add); sampling_match_test.zig checks it against
// TensorFold's. C++ as a better C: POD, free functions, extern "C" entries; no classes, templates, exceptions, RTTI,
// STL or allocation (the caller gives the scratch). sampling.zig is its Zig face.
// 鍵付きサンプリングを better C で。アルゴリズムは TensorFold のもの (Python の exact_sampling.py の定義) をそのまま
// 使い、演算の順序を変えない (ビット単位で一致させる。積和の融合はビルドで切る)。作業領域は呼ぶ側が渡す。

#include <math.h>
#include <stddef.h>
#include <stdint.h>

extern "C" {

// A request's sampling settings (sampling.zig's Sampling); a request without them decodes greedily.
struct FioreSampling {
    uint64_t seed;
    double temperature;
    uint32_t top_k; // 0: every candidate
    double top_p;
    double min_p;
};

// A candidate as choose sorts it.
struct FioreSampleEntry {
    double logit;
    uint64_t id;
};

} // extern "C"

static_assert(sizeof(FioreSampling) == 40 && offsetof(FioreSampling, top_k) == 16 && offsetof(FioreSampling, top_p) == 24 && offsetof(FioreSampling, min_p) == 32, "FioreSampling: sampling.zig's CSampling");
static_assert(sizeof(FioreSampleEntry) == 16, "FioreSampleEntry: 16 bytes");

// ---- SHA-256 (FIPS 180-4), for the seed of a prompt ----

struct Sha256 {
    uint32_t h[8];
    uint8_t block[64];
    uint32_t fill;  // bytes in block
    uint64_t bytes; // bytes hashed
};

static const uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static inline uint32_t sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_block(Sha256* s, const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | (uint32_t)p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = sha256_rotr(w[i - 15], 7) ^ sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = sha256_rotr(w[i - 2], 17) ^ sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3], e = s->h[4], f = s->h[5], g = s->h[6], h = s->h[7];
    for (int i = 0; i < 64; i++) {
        const uint32_t t1 = h + (sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25)) + ((e & f) ^ (~e & g)) + sha256_k[i] + w[i];
        const uint32_t t2 = (sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    s->h[0] += a;
    s->h[1] += b;
    s->h[2] += c;
    s->h[3] += d;
    s->h[4] += e;
    s->h[5] += f;
    s->h[6] += g;
    s->h[7] += h;
}

static void sha256_init(Sha256* s) {
    static const uint32_t h0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    for (int i = 0; i < 8; i++) s->h[i] = h0[i];
    s->fill = 0;
    s->bytes = 0;
}

static void sha256_update(Sha256* s, const uint8_t* p, size_t n) {
    s->bytes += n;
    for (size_t i = 0; i < n; i++) {
        s->block[s->fill++] = p[i];
        if (s->fill == 64) {
            sha256_block(s, s->block);
            s->fill = 0;
        }
    }
}

static void sha256_final(Sha256* s, uint8_t out[32]) {
    const uint64_t bits = s->bytes * 8;
    const uint8_t one = 0x80, zero = 0;
    sha256_update(s, &one, 1);
    while (s->fill != 56) sha256_update(s, &zero, 1);
    uint8_t length[8];
    for (int i = 0; i < 8; i++) length[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha256_update(s, length, 8);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++) out[4 * i + j] = (uint8_t)(s->h[i] >> (24 - 8 * j));
}

// v in decimal into out (at most 20 digits); returns the digits written.
static size_t decimal_u64(uint64_t v, char* out) {
    char t[20];
    size_t n = 0;
    do {
        t[n++] = (char)('0' + v % 10);
        v /= 10;
    } while (v != 0);
    for (size_t i = 0; i < n; i++) out[i] = t[n - 1 - i];
    return n;
}

// ---- the keyed draws ----

// splitmix64's finaliser.
static inline uint64_t scramble(uint64_t v) {
    uint64_t x = v ^ (v >> 30);
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// The 64 key bits of (seed, position, token).
static inline uint64_t key_bits(uint64_t seed, uint64_t position, uint64_t id) {
    const uint64_t a = scramble(seed + 0x9E3779B97F4A7C15ull);
    const uint64_t b = scramble(a ^ (position * 0xD1B54A32D192ED03ull));
    return scramble(b ^ id);
}

// numpy's float64 sum of a contiguous row (pairwise, eight accumulators in blocks of up to 128).
static double numpy_sum(const double* v, size_t n) {
    if (n < 8) {
        double acc = -0.0;
        for (size_t i = 0; i < n; i++) acc += v[i];
        return acc;
    }
    if (n <= 128) {
        double lane[8];
        for (int j = 0; j < 8; j++) lane[j] = v[j];
        const size_t whole = n - n % 8;
        size_t i = 8;
        for (; i < whole; i += 8)
            for (int j = 0; j < 8; j++) lane[j] += v[i + j];
        double acc = ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
        for (; i < n; i++) acc += v[i];
        return acc;
    }
    size_t half = n / 2;
    half -= half % 8;
    return numpy_sum(v, half) + numpy_sum(v + half, n - half);
}

// Highest logit first; on a tie, the smaller id first.
static inline bool before(FioreSampleEntry x, FioreSampleEntry y) {
    if (x.logit != y.logit) return x.logit > y.logit;
    return x.id < y.id;
}

static void sift_down(FioreSampleEntry* a, size_t root, size_t n) {
    for (;;) {
        size_t c = 2 * root + 1;
        if (c >= n) return;
        if (c + 1 < n && before(a[c], a[c + 1])) c += 1; // the child that comes later
        if (!before(a[root], a[c])) return;
        const FioreSampleEntry t = a[root];
        a[root] = a[c];
        a[c] = t;
        root = c;
    }
}

// a in `before` order. Candidate ids are distinct, so the order is total and any correct sort gives the same result as
// TensorFold's (a NaN logit has no place in it).
static void sort_entries(FioreSampleEntry* a, size_t n) {
    if (n < 2) return;
    for (size_t i = n / 2; i-- > 0;) sift_down(a, i, n);
    for (size_t end = n - 1; end > 0; end--) {
        const FioreSampleEntry t = a[0];
        a[0] = a[end];
        a[end] = t;
        sift_down(a, 0, end);
    }
}

extern "C" {

// A prompt's seed when the request names none: the first 8 bytes (little endian) of the sha256 of the ids written in
// decimal and joined by commas, then "|" and the salt, kept to 63 bits.
uint64_t fiore_sampling_seed_for(const uint32_t* tokens, size_t n, int64_t salt) {
    Sha256 s;
    sha256_init(&s);
    char text[24];
    for (size_t i = 0; i < n; i++) {
        if (i != 0) sha256_update(&s, (const uint8_t*)",", 1);
        sha256_update(&s, (const uint8_t*)text, decimal_u64(tokens[i], text));
    }
    size_t k = 0;
    text[k++] = '|';
    if (salt < 0) text[k++] = '-';
    k += decimal_u64(salt < 0 ? (uint64_t)0 - (uint64_t)salt : (uint64_t)salt, text + k);
    sha256_update(&s, (const uint8_t*)text, k);
    uint8_t digest[32];
    sha256_final(&s, digest);
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = v << 8 | digest[i];
    return v & ((1ull << 63) - 1);
}

// A uniform double in (0, 1] from the key's top 53 bits, offset by half a step.
double fiore_sampling_uniform(uint64_t seed, uint64_t position, uint64_t id) {
    const double top53 = (double)(key_bits(seed, position, id) >> 11);
    return top53 * 0x1p-53 + 0x1p-54;
}

// ln(min_p), or -inf when min_p is off.
double fiore_sampling_min_log(double min_p) {
    if (min_p > 0.0) return log(min_p);
    return -INFINITY;
}

// The scratch choose needs for n candidates (8-byte aligned).
size_t fiore_sampling_scratch_bytes(size_t n) {
    return n * (sizeof(FioreSampleEntry) + 2 * sizeof(double));
}

// The token drawn at `position` among the n (n >= 1) candidates `ids` with float64 logits `logits`: sorted, cut to
// top_k, then top_p (by probability mass) and min_p (against the best), and the Gumbel argmax of the rest under the
// keyed draws. `scratch` holds fiore_sampling_scratch_bytes(n).
uint64_t fiore_sampling_choose(const double* logits, const uint64_t* ids, size_t n, uint64_t position, FioreSampling s, void* scratch) {
    FioreSampleEntry* sorted = (FioreSampleEntry*)scratch;
    double* scaled = (double*)(sorted + n);
    double* p = scaled + n;
    for (size_t i = 0; i < n; i++) sorted[i] = FioreSampleEntry{logits[i], ids[i]};
    sort_entries(sorted, n);

    const size_t top_k = s.top_k == 0 ? n : (size_t)s.top_k;
    size_t k = top_k < n ? top_k : n;
    if (k < 1) k = 1;
    const double temperature = s.temperature > 1e-6 ? s.temperature : 1e-6;
    for (size_t i = 0; i < k; i++) scaled[i] = sorted[i].logit / temperature;

    size_t kept = k;
    if (s.top_p > 0.0 && s.top_p < 1.0) {
        for (size_t i = 0; i < k; i++) p[i] = exp(scaled[i] - scaled[0]);
        const double total = numpy_sum(p, k);
        double mass = 0.0;
        size_t under = 0;
        for (size_t i = 0; i < k; i++) {
            p[i] /= total;
            mass += p[i];
            if (mass < s.top_p) under += 1;
        }
        if (under + 1 < kept) kept = under + 1;
    }
    if (s.min_p > 0.0) {
        const double floor = scaled[0] + fiore_sampling_min_log(s.min_p);
        size_t above = 0;
        for (size_t i = 0; i < kept; i++) above += scaled[i] >= floor ? 1 : 0;
        kept = above;
    }

    size_t pick = 0;
    double best = -INFINITY;
    for (size_t j = 0; j < kept; j++) {
        const double score = scaled[j] + -log(-log(fiore_sampling_uniform(s.seed, position, sorted[j].id)));
        if (j == 0 || score > best) {
            pick = j;
            best = score;
        }
    }
    return sorted[pick].id;
}

} // extern "C"
