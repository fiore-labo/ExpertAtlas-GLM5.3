//! Fiore-lab: how far a prompt's text agrees with a slot's tokens past the first id that differs (ollama#12375, the
//! report on re-tokenizing). The model may cut a text differently from the tokenizer: the next turn's prompt holds the
//! same bytes under other ids, and the id match stops there. The walk reads both sides as bytes from that id on and
//! returns the last place both are on a token boundary with every byte equal. Added tokens match by id alone: a text
//! typed as "</think>" is not the token. `merged` is the prompt with the slot's ids up to there, so the cache resumes
//! past the place (TF_GLM53_BYTE_RESUME=on).
//! Fiore-lab: 最初に違う id より先で、prompt の文字がスロットのトークンとどこまで一致するか (ollama#12375、切り直しの
//! 調査)。モデルはトークナイザーと違う切り方をすることがあり、次のターンの prompt は同じバイトを別の id で持つので、id の
//! 照合はそこで止まる。その id から先は両方をバイトで読み、両方がトークンの切れ目にあり、すべてのバイトが等しい最後の
//! 位置を返す。追加トークンは id だけで一致する。文字で打った "</think>" はそのトークンではない。`merged` はそこまでを
//! スロットの id にした prompt で、キャッシュはその先から再開する (TF_GLM53_BYTE_RESUME=on)。
const std = @import("std");

pub const Stop = enum { seq_end, prompt_end, bytes, added };

pub const Result = struct {
    p: u64, // the first id that differs / 最初に違う id
    q: u64, // the slot's tokens that agree in bytes / バイトで一致するスロットのトークン数
    r: u64, // the prompt's tokens at that place / その位置の prompt のトークン数
    drifts: u32, // places the ids differed and the bytes went on / id が違いバイトが続いた回数
    stop: Stop,
};

/// `vocab`: bytes(id) ?[]const u8 (a token's raw bytes, null: unknown), added(id) bool.
pub fn walk(seq: []const i64, prompt: []const u32, vocab: anytype) Result {
    const lim = @min(seq.len, prompt.len);
    var i: usize = 0;
    while (i < lim and seq[i] == prompt[i]) i += 1;
    var j = i;
    var res: Result = .{ .p = i, .q = i, .r = i, .drifts = 0, .stop = .bytes };
    var ab: []const u8 = ""; // what the two sides' current tokens have not matched yet / 今のトークンのまだ比べていない分
    var bb: []const u8 = "";
    while (true) {
        if (ab.len == 0 and bb.len == 0) {
            res.q = i;
            res.r = j;
            if (i == seq.len) return stopped(res, .seq_end);
            if (j == prompt.len) return stopped(res, .prompt_end);
            if (seq[i] == prompt[j]) {
                i += 1;
                j += 1;
                continue;
            }
            res.drifts += 1;
        }
        if (ab.len == 0) {
            if (i == seq.len) return stopped(res, .seq_end);
            if (seq[i] < 0 or seq[i] > std.math.maxInt(u32)) return stopped(res, .added);
            const id: u32 = @intCast(seq[i]);
            if (vocab.added(id)) return stopped(res, .added);
            ab = vocab.bytes(id) orelse return stopped(res, .bytes);
            i += 1;
        }
        if (bb.len == 0) {
            if (j == prompt.len) return stopped(res, .prompt_end);
            if (vocab.added(prompt[j])) return stopped(res, .added);
            bb = vocab.bytes(prompt[j]) orelse return stopped(res, .bytes);
            j += 1;
        }
        const n = @min(ab.len, bb.len);
        if (!std.mem.eql(u8, ab[0..n], bb[0..n])) return stopped(res, .bytes);
        ab = ab[n..];
        bb = bb[n..];
    }
}

/// The prompt with the slot's ids where the bytes agree: seq[0..q] then prompt[r..]. Its text is the prompt's, byte for byte.
/// prompt のバイトが一致する所をスロットの id にしたもの: seq[0..q] の後に prompt[r..]。文は prompt と 1 バイトも違わない。
pub fn merged(a: std.mem.Allocator, seq: []const i64, prompt: []const u32, x: Result) std.mem.Allocator.Error![]u32 {
    const q: usize = @intCast(x.q);
    const r: usize = @intCast(x.r);
    const out = try a.alloc(u32, q + prompt.len - r);
    for (out[0..q], seq[0..q]) |*o, id| o.* = @intCast(id);
    @memcpy(out[q..], prompt[r..]);
    return out;
}

fn stopped(res: Result, why: Stop) Result {
    var r = res;
    r.stop = why;
    return r;
}

const t = std.testing;

const Fake = struct {
    pub fn bytes(_: Fake, id: u32) ?[]const u8 {
        return switch (id) {
            1 => "a",
            2 => "b",
            3 => "ab",
            4 => "c",
            5 => "\xE3", // the first byte of あ / あ の最初のバイト
            6 => "\x81\x82", // the rest of it / 残り
            7 => "\xE3\x81\x82", // あ whole / あ 全体
            8 => "<s>", // typed as text / 文字として打った
            9 => "z",
            10 => "\n",
            11 => "\n\n",
            else => null,
        };
    }
    pub fn added(_: Fake, id: u32) bool {
        return id == 100;
    }
};

fn check(seq: []const i64, prompt: []const u32, p: u64, q: u64, r: u64, drifts: u32, stop: Stop) !void {
    const x = walk(seq, prompt, Fake{});
    try t.expectEqual(p, x.p);
    try t.expectEqual(q, x.q);
    try t.expectEqual(r, x.r);
    try t.expectEqual(drifts, x.drifts);
    try t.expectEqual(stop, x.stop);
}

test "the same ids: nothing to read" {
    try check(&.{ 1, 2, 4 }, &.{ 1, 2, 4, 9 }, 3, 3, 3, 0, .seq_end);
    try check(&.{ 1, 2, 4 }, &.{ 1, 2 }, 2, 2, 2, 0, .prompt_end);
}

test "a word cut two ways, and the ids go on after it" {
    try check(&.{ 1, 2, 4 }, &.{ 3, 4, 9 }, 0, 3, 2, 1, .seq_end);
    try check(&.{ 9, 1, 2, 4, 9 }, &.{ 9, 3, 4, 9, 1 }, 1, 5, 4, 1, .seq_end);
}

test "line breaks cut two ways" {
    try check(&.{ 10, 10, 4 }, &.{ 11, 4 }, 0, 3, 2, 1, .seq_end);
}

test "a character split across tokens matches by its bytes" {
    try check(&.{ 5, 6, 4 }, &.{ 7, 4 }, 0, 3, 2, 1, .seq_end);
}

test "more than one place the ids differ" {
    try check(&.{ 1, 2, 4, 1, 2 }, &.{ 3, 4, 3, 9 }, 0, 5, 3, 2, .seq_end);
}

test "the bytes differ: the last boundary both shared" {
    try check(&.{ 1, 2, 4 }, &.{ 3, 1 }, 0, 2, 1, 2, .bytes);
    try check(&.{ 1, 2, 4 }, &.{ 9, 3 }, 0, 0, 0, 1, .bytes);
}

test "one side ends inside the other's token: the last boundary both shared" {
    try check(&.{ 1, 1 }, &.{ 1, 3, 9 }, 1, 1, 1, 1, .seq_end);
    try check(&.{ 1, 3, 9 }, &.{ 1, 1 }, 1, 1, 1, 1, .prompt_end);
}

test "added tokens match by id alone, never by their text" {
    try check(&.{ 100, 1 }, &.{ 100, 1, 9 }, 2, 2, 2, 0, .seq_end);
    try check(&.{ 100, 1 }, &.{ 8, 1 }, 0, 0, 0, 1, .added);
    try check(&.{ 8, 1 }, &.{ 100, 1 }, 0, 0, 0, 1, .added);
    try check(&.{ 1, 100 }, &.{ 3, 100 }, 0, 0, 0, 1, .added);
}

test "merged: the slot's ids where the bytes agree, then the prompt's; the ids now match to there" {
    const seq = [_]i64{ 9, 1, 2, 4, 5, 6, 9 };
    const prompt = [_]u32{ 9, 3, 4, 7, 9, 2, 4 };
    const x = walk(&seq, &prompt, Fake{});
    try t.expectEqual(@as(u64, 1), x.p);
    try t.expectEqual(@as(u64, 7), x.q);
    try t.expectEqual(@as(u64, 5), x.r);
    const m = try merged(t.allocator, &seq, &prompt, x);
    defer t.allocator.free(m);
    try t.expectEqualSlices(u32, &.{ 9, 1, 2, 4, 5, 6, 9, 2, 4 }, m);
    const again = walk(&seq, m, Fake{});
    try t.expectEqual(@as(u64, 7), again.p);
    try t.expectEqual(@as(u32, 0), again.drifts);
    // the same bytes as the prompt, in order
    var a: std.ArrayList(u8) = .empty;
    defer a.deinit(t.allocator);
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(t.allocator);
    for (prompt) |id| try a.appendSlice(t.allocator, (Fake{}).bytes(id).?);
    for (m) |id| try b.appendSlice(t.allocator, (Fake{}).bytes(id).?);
    try t.expectEqualStrings(a.items, b.items);
}

test "merged: nothing past the first difference keeps the prompt as it is" {
    const seq = [_]i64{ 1, 9 };
    const prompt = [_]u32{ 1, 4, 9 };
    const x = walk(&seq, &prompt, Fake{});
    try t.expectEqual(x.p, x.q);
    const m = try merged(t.allocator, &seq, &prompt, x);
    defer t.allocator.free(m);
    try t.expectEqualSlices(u32, &prompt, m);
}

test "random texts cut two ways: what the walk returns holds for every seed" {
    // the vocabulary: every string of 1..4 letters of "abc" (ids 0..119), and the added token 200, whose text is "a"
    // 語彙: "abc" の 1〜4 文字のすべての文字列 (id 0..119) と追加トークン 200 (文は "a")
    const Table = struct {
        strs: [120][4]u8 = undefined,
        lens: [120]u8 = undefined,
        pub fn bytes(s: *const @This(), id: u32) ?[]const u8 {
            if (id == 200) return "a";
            if (id >= 120) return null;
            return s.strs[id][0..s.lens[id]];
        }
        pub fn added(_: *const @This(), id: u32) bool {
            return id == 200;
        }
        fn idOf(s: *const @This(), text: []const u8) u32 {
            for (0..120) |i| if (std.mem.eql(u8, s.strs[i][0..s.lens[i]], text)) return @intCast(i);
            unreachable;
        }
    };
    var v: Table = .{};
    var n: usize = 0;
    for (1..5) |len| {
        var count: usize = 1;
        for (0..len) |_| count *= 3;
        for (0..count) |c| {
            var x = c;
            for (0..len) |k| {
                v.strs[n][k] = "abc"[x % 3];
                x /= 3;
            }
            v.lens[n] = @intCast(len);
            n += 1;
        }
    }
    const gpa = t.allocator;
    var used: usize = 0;
    for (0..3000) |seed| {
        var prng: std.Random.DefaultPrng = .init(@intCast(seed));
        const rnd = prng.random();
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        const len = rnd.intRangeAtMost(usize, 1, 200);
        for (0..len) |_| try text.append(gpa, "abc"[rnd.uintLessThan(u8, 3)]);
        // the prompt's text: the same, or with a letter changed (past which nothing may match), or a shorter one
        // prompt の文: 同じ、1 文字変えたもの (その先は一致してはならない)、短いもの
        var other = try gpa.dupe(u8, text.items);
        defer gpa.free(other);
        var differ: ?usize = null;
        switch (rnd.uintLessThan(u8, 4)) {
            0 => {
                const at = rnd.uintLessThan(usize, other.len);
                other[at] = if (other[at] == 'a') 'b' else 'a';
                differ = at;
            },
            1 => other = try gpa.realloc(other, rnd.intRangeAtMost(usize, 1, other.len)),
            else => {},
        }
        const cut = struct {
            fn ids(a: std.mem.Allocator, r: std.Random, tb: *const Table, s: []const u8, add_at: ?usize) ![]u32 {
                var out: std.ArrayList(u32) = .empty;
                errdefer out.deinit(a);
                var i: usize = 0;
                while (i < s.len) {
                    if (add_at != null and i >= add_at.?) {
                        try out.append(a, 200);
                        return out.toOwnedSlice(a);
                    }
                    const k = @min(s.len - i, r.intRangeAtMost(usize, 1, 4));
                    try out.append(a, tb.idOf(s[i .. i + k]));
                    i += k;
                }
                return out.toOwnedSlice(a);
            }
        };
        // sometimes one side holds the added token where the other holds its text
        // 時々、片方が追加トークンを持ち、もう片方が同じ文を持つ
        const add_seq: ?usize = if (rnd.uintLessThan(u8, 8) == 0) rnd.uintLessThan(usize, text.items.len) else null;
        const a_ids = try cut.ids(gpa, rnd, &v, text.items, add_seq);
        defer gpa.free(a_ids);
        const prompt = try cut.ids(gpa, rnd, &v, other, null);
        defer gpa.free(prompt);
        const seq = try gpa.alloc(i64, a_ids.len);
        defer gpa.free(seq);
        for (seq, a_ids) |*o, id| o.* = id;
        const x = walk(seq, prompt, &v);
        try t.expect(x.p <= x.q and x.q <= seq.len and x.r <= prompt.len);
        var sa: std.ArrayList(u8) = .empty;
        defer sa.deinit(gpa);
        var sb: std.ArrayList(u8) = .empty;
        defer sb.deinit(gpa);
        for (seq[0..@intCast(x.q)]) |id| try sa.appendSlice(gpa, v.bytes(@intCast(id)).?);
        for (prompt[0..@intCast(x.r)]) |id| try sb.appendSlice(gpa, v.bytes(id).?);
        // both sides up to the place hold the same bytes, and nothing past a changed letter
        // その位置までの両側のバイトは同じで、変えた文字より先には進まない
        try t.expectEqualStrings(sa.items, sb.items);
        if (differ) |d| try t.expect(sa.items.len <= d);
        for (seq[0..@intCast(x.q)]) |id| if (id == 200) try t.expect(std.mem.indexOfScalar(u32, prompt[0..@intCast(x.r)], 200) != null);
        // the merged prompt's bytes are the prompt's, and its ids match the slot to the place
        // merged の prompt のバイトは prompt と同じで、id はその位置までスロットと一致する
        const m = try merged(gpa, seq, prompt, x);
        defer gpa.free(m);
        var mb: std.ArrayList(u8) = .empty;
        defer mb.deinit(gpa);
        var pb: std.ArrayList(u8) = .empty;
        defer pb.deinit(gpa);
        for (m) |id| try mb.appendSlice(gpa, v.bytes(id).?);
        for (prompt) |id| try pb.appendSlice(gpa, v.bytes(id).?);
        try t.expectEqualStrings(pb.items, mb.items);
        const y = walk(seq, m, &v);
        try t.expect(y.p >= x.q);
        // the same text with no added token: it reaches the end of one side unless the cuts leave a token across it
        // 追加トークンなしで同じ文: 切れ目が残らない限りどちらかの終わりまで届く
        if (differ == null and add_seq == null and other.len == text.items.len) try t.expect(x.q == seq.len and x.r == prompt.len);
        used += @intFromBool(x.q > x.p);
    }
    // the walk really went past the first difference in many of them / 多くの場合で最初の違いより先に進んだ
    try t.expect(used > 500);
}

test "an unknown token stops the walk" {
    try check(&.{ 1, 55 }, &.{ 3, 4 }, 0, 0, 0, 1, .bytes);
}
