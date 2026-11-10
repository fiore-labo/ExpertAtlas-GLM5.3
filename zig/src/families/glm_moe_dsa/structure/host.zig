//! Fiore-lab: structured output on the host only: the compile, the mask, the single accept point and the grammar's
//! start after the thinking.
//! Fiore-lab: host だけの構造化出力: compile、mask、唯一の受理点、思考の後の文法の始まり。
const std = @import("std");
pub const llg = @import("llg.zig");
pub const vocab_size = 154880;
pub const words = llg.maskWords(vocab_size);
pub const bytes = words * @sizeOf(u32);
pub const invalid_token = std.math.maxInt(u32);

// Fiore-lab: json_object's fallback schema has the x-guidance of a json_schema request (no u escape, bounded
// whitespace); nonempty server text always wins.
// Fiore-lab: json_object の代替 schema も json_schema の要求と同じ x-guidance (u のエスケープなし、空白に上限)。サーバーの
// 文があれば必ずそちらを使う。
pub const object_schema = "{\"type\":\"object\",\"x-guidance\":{\"lenient\":false,\"coerce_one_of\":false,\"json_allowed_escapes\":\"nrbtf\\\\\\\"\",\"json_allow_general_unicode_escapes\":false,\"whitespace_pattern\":\"[\\\\x20\\\\x09]{1,64}|\\\\x0D?\\\\x0A[\\\\x20\\\\x09]{0,64}\",\"item_separator\":\",\",\"key_separator\":\":\"}}";

pub fn firstLine(s: []const u8) []const u8 {
    return s[0 .. std.mem.indexOfAny(u8, s, "\r\n") orelse s.len];
}

pub const Runtime = struct {
    library: llg.Library,
    tokenizer: ?*llg.Tokenizer = null,
    eos: []u32 = &.{},
    real_size: u32 = 0,

    /// `size`: the server tokenizer's ids (GLM-5.3: 154,856); the table llguidance gets has the model's 154,880, the
    /// ids past `size` without bytes (the model's padding, never allowed).
    /// `size`: サーバーの tokenizer の id の数 (GLM-5.3 は 154,856)。llguidance に渡す表はモデルの 154,880 で、`size` より後の
    /// id はバイトなし (モデルの詰め物。決して許されない)。
    pub fn init(r: *Runtime, a: std.mem.Allocator, src: llg.VocabSource, size: u32, eos: []const u32) !void {
        if (r.tokenizer != null) return;
        if (size == 0 or size > vocab_size or eos.len == 0) return error.Unavailable;
        for (eos) |id| if (id >= size) return error.Unavailable;
        var within: Within = .{ .src = src, .size = size };
        var v = try llg.Vocab.build(a, .{ .ctx = &within, .bytes = Within.bytes, .added = Within.added }, vocab_size);
        defer v.deinit(a);
        var real: usize = v.lens.len;
        while (real > 0 and v.lens[real - 1] == 0) real -= 1;
        const ends = try a.dupe(u32, eos);
        errdefer a.free(ends);
        var err: [512]u8 = undefined;
        const tok = llg.newTokenizer(&r.library.api, &v, eos, &err) orelse return error.Unavailable;
        r.eos = ends;
        r.real_size = @intCast(real);
        r.tokenizer = tok;
    }

    /// The server's vocabulary up to its own size only: an id past it is never asked for.
    /// サーバーの語彙はその大きさまでだけ: それより後の id は尋ねない。
    const Within = struct {
        src: llg.VocabSource,
        size: u32,

        fn bytes(ctx: *anyopaque, a: std.mem.Allocator, id: u32) ?[]const u8 {
            const w: *Within = @ptrCast(@alignCast(ctx));
            return if (id < w.size) w.src.bytes(w.src.ctx, a, id) else null;
        }
        fn added(ctx: *anyopaque, id: u32) bool {
            const w: *Within = @ptrCast(@alignCast(ctx));
            return id < w.size and w.src.added(w.src.ctx, id);
        }
    };

    pub const Compiled = union(enum) { ok: *llg.Matcher, refused: []const u8 };

    // Fiore-lab: called under the Host's compile mutex, before submit transfers ownership.
    // Fiore-lab: Host の compile mutex の下で呼ぶ。submit が持ち主を移す前。
    pub fn compile(r: *Runtime, a: std.mem.Allocator, text: []const u8) !Compiled {
        const tok = r.tokenizer orelse return .{ .refused = try a.dupe(u8, "structured output is unavailable") };
        if (std.mem.indexOfScalar(u8, text, 0) != null) return .{ .refused = try a.dupe(u8, "schema contains a NUL byte") };
        const z = try a.dupeSentinel(u8, text, 0);
        defer a.free(z);
        const settings = llg.constraintInit(&r.library.api, tok);
        const m = r.library.api.llg_new_matcher(&settings, "json_schema", z.ptr) orelse return .{ .refused = try a.dupe(u8, "could not allocate grammar matcher") };
        var keep = false;
        defer if (!keep) r.library.api.llg_free_matcher(m);
        var st = r.state(m);
        if (r.library.api.llg_matcher_is_error(m)) {
            st.failMatcher("grammar compilation failed");
        } else if (r.library.api.llg_matcher_get_mask_byte_size(m) != bytes) {
            st.fail("grammar mask must have 19360 bytes");
        } else {
            var row: [words]u32 = undefined;
            _ = st.fill(&row);
        }
        if (st.failed()) return .{ .refused = try a.dupe(u8, st.message()) };
        keep = true;
        return .{ .ok = m };
    }

    pub fn state(r: *Runtime, m: *llg.Matcher) State {
        return .{ .api = &r.library.api, .matcher = m, .eos = r.eos, .real_size = r.real_size };
    }

    pub fn deinit(r: *Runtime, a: std.mem.Allocator) void {
        if (r.tokenizer) |t| r.library.api.llg_free_tokenizer(t);
        a.free(r.eos);
        r.library.close();
    }
};

pub const State = struct {
    api: *const llg.Api,
    matcher: *llg.Matcher,
    eos: []const u32,
    real_size: u32,
    active: bool = false,
    answered: bool = false,
    done: bool = false,
    why: [512]u8 = undefined,
    why_len: usize = 0,

    pub fn failed(s: *const State) bool {
        return s.why_len != 0;
    }
    pub fn message(s: *const State) []const u8 {
        return s.why[0..s.why_len];
    }
    pub fn fail(s: *State, why: []const u8) void {
        if (s.failed()) return;
        const line = firstLine(why);
        const msg = if (line.len > 0) line else "grammar engine failed";
        s.why_len = @min(s.why.len, msg.len);
        @memcpy(s.why[0..s.why_len], msg[0..s.why_len]);
    }
    fn failMatcher(s: *State, fallback: []const u8) void {
        s.fail(if (s.api.llg_matcher_get_error(s.matcher)) |p| std.mem.span(p) else fallback);
    }

    // Fiore-lab: no staging copy: llguidance writes directly into the engine's pinned row.
    // Fiore-lab: 中間コピーなし。llguidance は engine の pin した行へ直接書く。
    pub fn fill(s: *State, row: []u32) bool {
        if (s.failed()) return false;
        if (row.len != words) {
            s.fail("invalid grammar mask row size");
            return false;
        }
        if (s.api.llg_matcher_compute_mask_into(s.matcher, row.ptr, row.len * 4) != 0) {
            s.failMatcher("grammar mask computation failed");
            return false;
        }
        clearPadding(row, s.real_size);
        for (row) |w| if (w != 0) return true;
        s.fail("grammar mask has no allowed token");
        return false;
    }

    pub const Accept = enum { token, done, failed };
    pub fn accept(s: *State, id: u32) Accept {
        if (s.failed()) return .failed;
        if (std.mem.indexOfScalar(u32, s.eos, id) != null) {
            if (s.api.llg_matcher_is_accepting(s.matcher)) {
                s.done = true;
                return .done;
            }
            s.fail("grammar ended before accepting the answer");
            return .failed;
        }
        if (s.api.llg_matcher_consume_token(s.matcher, id) != 0) {
            s.failMatcher("grammar rejected the output token");
            return .failed;
        }
        return .token;
    }

    // Fiore-lab: failure precedes cancellation and completion; a length limit may leave partial JSON.
    // Fiore-lab: 失敗を取り消し・完了より優先。長さの制限では途中の JSON を返してよい。
    pub fn finishReason(s: *State, length: bool, cancelled: bool) enum { failed, cancelled, stop, length } {
        if (s.failed()) return .failed;
        if (cancelled) return .cancelled;
        if (s.done) return .stop;
        if (length) return .length;
        s.fail("structured reply stopped before completing its answer");
        return .failed;
    }

    // Fiore-lab: only at a segment head, after all close/fold/canon handling; answer fires once.
    // Fiore-lab: 閉じ・fold・canon の後の区間の頭だけ。answer は 1 回。
    pub fn segment(s: *State, thinking: bool, shut: bool, branch: bool) bool {
        s.active = !branch and (!thinking or shut);
        if (!s.active or s.answered) return false;
        s.answered = true;
        return true;
    }
};

/// Fiore-lab: a batched round's masks staged before its event is shared (TF_GLM53_MULTI): for each conversation of the
/// round (`states`, null for one without a grammar; `gram`, its admission's grammar bit), a grammar conversation not
/// stopped gets its matcher's mask in `rows[i]` (the engine's staging row, null when there is none). One it cannot
/// fill is stopped in `stops` and marked in `fresh`, so the round's event stops it on every rank; the caller stops its
/// job. Returns how many were stopped here.
/// Fiore-lab: まとめたラウンドの出来事を配る前に mask を置き場に書く (TF_GLM53_MULTI): ラウンドの会話ごとに
/// (`states` は文法のない会話なら null、`gram` は受け付けの文法の bit)、止めていない文法の会話は matcher の mask を
/// `rows[i]` (エンジンの置き場の行。なければ null) に受け取る。書けなかった会話は `stops` で止め、`fresh` に印を付ける。
/// ラウンドの出来事が全ランクでそれを止め、呼び出し側が job を止める。ここで止めた数を返す。
pub fn stageRound(states: []const ?*State, gram: []const bool, rows: []const ?[]u32, stops: []bool, fresh: []bool) usize {
    std.debug.assert(states.len == gram.len and gram.len == rows.len and rows.len == stops.len and stops.len == fresh.len);
    var n: usize = 0;
    for (states, gram, rows, stops, fresh) |st, g, row, *stp, *fr| {
        fr.* = false;
        if (stp.* or !g) continue;
        // a grammar row with no matcher must not run on a stale staged mask: stopped
        // matcher のない文法の行は古い置き場の mask で走らせない: 止める
        const ok = if (st) |s| (if (row) |r| s.fill(r) else blk: {
            s.fail("structured output is unavailable");
            break :blk false;
        }) else false;
        if (!ok) {
            stp.* = true;
            fr.* = true;
            n += 1;
        }
    }
    return n;
}

/// Fiore-lab: a batched conversation's end (releaseConv), before its cut is followed: the masked pick's numeric verdict
/// recorded first (finishReason then puts it before the matcher's). Returns whether a cut may admit the next segment:
/// never after a failure, nor for a grammar close with no room left (`grammar_full`).
/// Fiore-lab: まとめた会話の終わり (releaseConv)、打ち切りに従う前: 先に mask つき pick の数値の判定を記録する
/// (finishReason はそれを matcher の判定より前に置く)。打ち切りのあと次の区間を受け入れてよいかを返す: 失敗のあとと、
/// 余地のない文法の閉じ (`grammar_full`) では不可。
pub fn releaseGoesOn(s: *State, numeric_failed: bool, cut: bool, grammar_full: bool) bool {
    if (numeric_failed) s.fail("grammar logits contain an allowed NaN or no allowed finite value");
    return !s.failed() and !grammar_full and cut;
}

pub fn clearPadding(row: []u32, real_size: u32) void {
    const start = @min(real_size, row.len * 32);
    var id: usize = start;
    while (id < row.len * 32) : (id += 1) row[id / 32] &= ~(@as(u32, 1) << @intCast(id % 32));
}

// Fiore-lab: slot 3's on/finite/NaN bits; slot 2 belongs to MTP confidence. Mirrors the CUDA verdict.
// Fiore-lab: slot 3 の on/有限/NaN の bit。slot 2 は MTP confidence。CUDA と同じ判定。
pub fn numericFailed(flags: []const u32) bool {
    var all: u32 = 0;
    for (flags) |f| all |= f;
    return all & 4 != 0 and (all & 1 == 0 or all & 2 != 0);
}
