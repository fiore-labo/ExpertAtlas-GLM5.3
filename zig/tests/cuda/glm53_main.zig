//! GLM-5.3 gates: `tf-glm53-test bmm DIR` / `linear DIR` / `experts DIR` replay recorded calls and compare the outputs
//! byte for byte; `digests CKPT RANK WORLD LAYERS STORED_WORLD OUT` hashes the rank's tensors. They need reference
//! fixtures made by a Python reference implementation, which this repository does not include.
//! GLM-5.3 のゲート: `bmm DIR` / `linear DIR` / `experts DIR` は記録した呼び出しを再生し、出力をバイト単位で比べる。
//! `digests ...` はランクのテンソルをハッシュする。どれも Python の参照実装が作る基準データが要るが、その参照実装は
//! このリポジトリに含まない。

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_moe_dsa");

const Case = struct {
    name: []const u8,
    A: Operand,
    B: Operand,
    C: Operand,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    lda: c_int,
    strideA: i64,
    ldb: c_int,
    strideB: i64,
    ldc: c_int,
    strideC: i64,
    batchCount: c_int,
    computeType: c_int,
    algo: c_int,
    Atype: c_int,
    expected: []const u8,
    c_bytes: usize,

    const Operand = struct { base: []const u8, offset_elems: usize };
};
const Cases = struct { H: u32, R: u32, cases: []Case };

fn readAll(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31));
}

fn bmm(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !u8 {
    const text = try readAll(gpa, io, dir, "cases.json");
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(Cases, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var lib = try cuda.cublas.Library.open();
    defer lib.close();
    var h: cuda.cublas.Handle = null;
    try lib.check(lib.api.cublasCreate_v2(&h), "cublasCreate");
    defer _ = lib.api.cublasDestroy_v2(h);
    var version: c_int = 0;
    try lib.check(lib.api.cublasGetVersion_v2(h, &version), "cublasGetVersion");
    var ws = try cuda.DeviceBuffer.alloc(d, cuda.cublas.torch_workspace);
    defer ws.free();
    try lib.check(lib.api.cublasSetWorkspace_v2(h, ws.ptr, cuda.cublas.torch_workspace), "cublasSetWorkspace");
    var failed: u8 = 0;
    for (parsed.value.cases) |c| {
        var bufs: [3]cuda.DeviceBuffer = undefined;
        const ops = [_]Case.Operand{ c.A, c.B, c.C };
        for (ops, 0..) |op, i| {
            const bytes = try readAll(gpa, io, dir, try std.fmt.allocPrint(gpa, "{s}.bin", .{op.base}));
            defer gpa.free(bytes);
            bufs[i] = try cuda.DeviceBuffer.fromHost(d, bytes);
            if (i == 2) try bufs[i].fill8(0, null);
        }
        defer for (&bufs) |*b| b.free();
        try lib.gemmStridedBatched(h, .{
            .transa = c.transa, .transb = c.transb, .m = c.m, .n = c.n, .k = c.k,
            .a = try bufs[0].at(c.A.offset_elems * 2), .a_type = c.Atype, .lda = c.lda, .stride_a = c.strideA,
            .b = try bufs[1].at(c.B.offset_elems * 2), .b_type = c.Atype, .ldb = c.ldb, .stride_b = c.strideB,
            .c = try bufs[2].at(c.C.offset_elems * 2), .c_type = c.Atype, .ldc = c.ldc, .stride_c = c.strideC,
            .batch = c.batchCount, .compute = c.computeType, .algo = c.algo,
        });
        const want = try readAll(gpa, io, dir, try std.fmt.allocPrint(gpa, "{s}.bin", .{c.expected}));
        defer gpa.free(want);
        const got = try gpa.alloc(u8, c.c_bytes);
        defer gpa.free(got);
        try bufs[2].download(0, got);
        const same = std.mem.eql(u8, got, want[0..c.c_bytes]);
        var diff: usize = 0;
        if (!same) for (got, want[0..c.c_bytes]) |x, y| {
            if (x != y) diff += 1;
        };
        std.debug.print("{s} bmm {s} R={d}: m{d} n{d} k{d} ta{d} tb{d} -> {s} ({d} bytes differ, cuBLAS {d})\n", .{ if (same) "PASS" else "FAIL", c.name, parsed.value.R, c.m, c.n, c.k, c.transa, c.transb, if (same) "bit-exact" else "DIFFERENT", diff, version });
        if (!same) failed += 1;
    }
    return failed;
}

fn torchDtype(d: anytype) []const u8 {
    return switch (d) {
        .bf16 => "bfloat16", .f16 => "float16", .f32 => "float32", .f64 => "float64", .i8 => "int8", .u8 => "uint8",
        .i16 => "int16", .u16 => "uint16", .i32 => "int32", .u32 => "uint32", .i64 => "int64", .u64 => "uint64", .bool => "bool",
    };
}

/// Canonical tensor names below `layers` (pre-sharded experts by their unsharded name), sorted.
/// `layers` 未満の層のテンソルの正規名 (事前分割した expert は分割前の名前で) を整列したもの。
fn canonicalNames(a: std.mem.Allocator, r: *glm.weights.Reader, layers: usize) ![][]const u8 {
    var set: std.StringArrayHashMapUnmanaged(void) = .empty;
    var it = r.index.keyIterator();
    while (it.next()) |k| {
        const n = k.*;
        if (std.mem.startsWith(u8, n, "model.layers.")) {
            const rest = n["model.layers.".len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse continue;
            if ((std.fmt.parseInt(usize, rest[0..dot], 10) catch continue) >= layers) continue;
        }
        var name: []const u8 = n;
        if (std.mem.indexOf(u8, n, ".rank")) |i| {
            const after = std.mem.indexOfScalarPos(u8, n, i + 1, '.') orelse n.len;
            if (after > i + 5 and std.ascii.isDigit(n[i + 5])) name = try std.mem.concat(a, u8, &.{ n[0..i], n[after..] });
        }
        try set.put(a, name, {});
    }
    const out = try a.dupe([]const u8, set.keys());
    std.mem.sort([]const u8, out, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    return out;
}

fn digests(a: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !u8 {
    const rank = try std.fmt.parseInt(usize, args[3], 10);
    const world = try std.fmt.parseInt(usize, args[4], 10);
    const layers = try std.fmt.parseInt(usize, args[5], 10);
    const stored = try std.fmt.parseInt(usize, args[6], 10);
    var r = try glm.weights.Reader.open(std.heap.page_allocator, io, args[2], rank, world, stored);
    defer r.close();
    var buf: std.ArrayList(u8) = .empty;
    try buf.print(a, "{{\"rank\": {d}, \"world\": {d}, \"layers\": {d}, \"tensors\": {{", .{ rank, world, layers });
    var n: usize = 0;
    for (try canonicalNames(a, &r, layers)) |name| {
        if (!r.has(name)) continue;
        var s = try r.get(name);
        defer s.deinit(std.heap.page_allocator);
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(s.bytes, &d, .{});
        try buf.print(a, "{s}\n\"{s}\": {{\"dtype\": \"{s}\", \"shape\": [", .{ if (n == 0) "" else ",", name, torchDtype(s.dtype) });
        for (s.shape[0..s.rank], 0..) |dim, i| try buf.print(a, "{s}{d}", .{ if (i == 0) "" else ", ", dim });
        try buf.print(a, "], \"sha256\": \"{s}\"}}", .{std.fmt.bytesToHex(d, .lower)});
        n += 1;
    }
    try buf.appendSlice(a, "}}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[7], .data = buf.items });
    std.debug.print("rank {d}/{d}: {d} tensors -> {s}\n", .{ rank, world, n, args[7] });
    return 0;
}

/// One fixture call's arrays on the GPU, by name.
/// 基準データの呼び出し 1 回分の配列を GPU に載せ、名前で引けるようにする。
const Call = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    meta: std.json.Parsed(std.json.Value),
    bufs: std.StringArrayHashMapUnmanaged(cuda.DeviceBuffer) = .empty,

    fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Call {
        const text = try readAll(gpa, io, dir, "call.json");
        var c: Call = .{ .gpa = gpa, .io = io, .dir = dir, .meta = try std.json.parseFromSlice(std.json.Value, gpa, text, .{}) };
        var it = c.meta.value.object.get("arrays").?.object.iterator();
        while (it.next()) |e| {
            const bytes = try readAll(gpa, io, dir, try std.fmt.allocPrint(gpa, "{s}.bin", .{e.key_ptr.*}));
            defer gpa.free(bytes);
            try c.bufs.put(gpa, e.key_ptr.*, try cuda.DeviceBuffer.fromHost(d, bytes));
        }
        return c;
    }

    fn deinit(c: *Call) void {
        for (c.bufs.values()) |*b| b.free();
        c.bufs.deinit(c.gpa);
        c.meta.deinit();
    }

    fn ptr(c: *const Call, name: []const u8) u64 {
        return if (c.bufs.get(name)) |b| b.ptr else 0;
    }

    fn int(c: *const Call, name: []const u8) i64 {
        return c.meta.value.object.get("ints").?.object.get(name).?.integer;
    }

    fn list(c: *const Call, name: []const u8) []std.json.Value {
        return c.meta.value.object.get("ints").?.object.get(name).?.array.items;
    }

    /// Whether the GPU's `name` equals expected_`name`.bin, printing the count of differing bytes.
    /// GPU 上の `name` が expected_`name`.bin と等しいか。違うバイト数を表示する。
    fn same(c: *const Call, name: []const u8) !bool {
        const want = try readAll(c.gpa, c.io, c.dir, try std.fmt.allocPrint(c.gpa, "expected_{s}.bin", .{name}));
        defer c.gpa.free(want);
        const got = try c.gpa.alloc(u8, want.len);
        defer c.gpa.free(got);
        try c.bufs.get(name).?.download(0, got);
        var diff: usize = 0;
        for (got, want) |x, y| diff += @intFromBool(x != y);
        if (diff != 0) std.debug.print("    {s}: {d} of {d} bytes differ\n", .{ name, diff, want.len });
        return diff == 0;
    }
};

fn dtypeOf(s: []const u8) glm.exl3.DType {
    return if (std.mem.eql(u8, s, "float32")) .f32 else if (std.mem.eql(u8, s, "bfloat16")) .bf16 else .f16;
}

/// Replays every call under DIR/<case>/<n>-<fn>/ through exl3.Kernels and compares the outputs byte for byte.
/// DIR/<case>/<n>-<fn>/ の全呼び出しを exl3.Kernels で再生し、出力をバイト単位で比べる。
fn linearGate(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, root: []const u8) !u8 {
    var k = try glm.exl3.Kernels.load(d);
    defer k.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    var cases: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |e| if (e.kind == .directory) try cases.append(gpa, try gpa.dupe(u8, e.name));
    var failed: u8 = 0;
    var passed: usize = 0;
    for (cases.items) |case| {
        for (0..2) |n| {
            const fns = [_][]const u8{ "rot_in", "linear" };
            const sub = try std.fmt.allocPrint(gpa, "{s}/{s}/{d}-{s}", .{ root, case, n, fns[n] });
            var c = try Call.open(gpa, io, d, sub);
            var ok = true;
            if (n == 0) {
                const m: u32 = @intCast(c.meta.value.object.get("arrays").?.object.get("x").?.object.get("shape").?.array.items[0].integer);
                const kdim: u32 = @intCast(c.meta.value.object.get("arrays").?.object.get("x").?.object.get("shape").?.array.items[1].integer);
                const xdt = dtypeOf(c.meta.value.object.get("arrays").?.object.get("x").?.object.get("dtype").?.string);
                try k.rotIn(stream, c.ptr("x"), xdt, &.{c.ptr("suh0")}, &.{c.ptr("xh0")}, m, kdim, c.int("pdl") != 0);
                try stream.synchronize();
                ok = try c.same("xh0");
            } else {
                var jobs: glm.exl3.Jobs = .{ .n = @intCast(c.int("n")) };
                for (0..@intCast(c.int("n"))) |i| {
                    const nm = struct {
                        fn f(a: std.mem.Allocator, p: []const u8, j: usize) []const u8 {
                            return std.fmt.allocPrint(a, "{s}{d}", .{ p, j }) catch unreachable;
                        }
                    }.f;
                    jobs.j[i] = .{
                        .xh = c.ptr(nm(gpa, "xh", i)), .T = c.ptr(nm(gpa, "T", i)),
                        .stride_k = c.list("stride_k")[i].integer, .stride_nb = c.list("stride_nb")[i].integer,
                        .svh = c.ptr(nm(gpa, "svh", i)), .bias = c.ptr(nm(gpa, "bias", i)), .y = c.ptr(nm(gpa, "y", i)),
                        .Z = c.ptr(nm(gpa, "Z", i)), .counters = c.ptr(nm(gpa, "counters", i)),
                        .y_dtype = @intFromEnum(dtypeOf(c.list("y_dtype")[i].string)), .N = @intCast(c.list("N")[i].integer),
                        .SK = @intCast(c.list("SK")[i].integer),
                    };
                }
                // the recorded cases are 5 bits, mul1
                // 記録したケースは 5 bit、mul1
                try k.linear(stream, jobs, @intCast(c.int("M")), @intCast(c.int("K")), 10, 2, @intCast(c.int("WK")), @intCast(c.int("loads")));
                try stream.synchronize();
                for (0..@intCast(c.int("n"))) |i| {
                    ok = (try c.same(try std.fmt.allocPrint(gpa, "y{d}", .{i}))) and ok;
                    ok = (try c.same(try std.fmt.allocPrint(gpa, "counters{d}", .{i}))) and ok;
                }
            }
            std.debug.print("{s} {s}/{s}\n", .{ if (ok) "PASS" else "FAIL", case, fns[n] });
            if (ok) passed += 1 else failed += 1;
        }
    }
    std.debug.print("linear gate: {d} passed, {d} failed\n", .{ passed, failed });
    return failed;
}

/// The layer's trellis blobs (DIR/trellis/<name>.bin + .json) on the GPU and their per-expert pointer tables.
/// 層の trellis の塊 (DIR/trellis/<name>.bin + .json) を GPU に載せたものと、expert ごとのポインタ表。
const Tables = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    root: []const u8,
    blobs: std.StringArrayHashMapUnmanaged([2]cuda.DeviceBuffer) = .empty,

    /// The device pointer table [E] int64 of blob `name` (uploaded on first use).
    /// 塊 `name` のデバイス上のポインタ表 [E] int64 (初めて使うときに載せる)。
    fn ptrs(t: *Tables, name: []const u8) !u64 {
        if (t.blobs.get(name)) |b| return b[1].ptr;
        const dir = try std.fmt.allocPrint(t.gpa, "{s}/trellis", .{t.root});
        const meta_text = try readAll(t.gpa, t.io, dir, try std.fmt.allocPrint(t.gpa, "{s}.json", .{name}));
        const meta = try std.json.parseFromSlice(struct { offsets: []i64, bytes: i64, experts: u32 }, t.gpa, meta_text, .{});
        const bytes = try readAll(t.gpa, t.io, dir, try std.fmt.allocPrint(t.gpa, "{s}.bin", .{name}));
        defer t.gpa.free(bytes);
        if (bytes.len != meta.value.bytes or meta.value.offsets.len != meta.value.experts) return error.BadTrellisBlob;
        const blob = try cuda.DeviceBuffer.fromHost(t.d, bytes);
        const table = try t.gpa.alloc(u64, meta.value.offsets.len);
        for (meta.value.offsets, table) |o, *p| p.* = blob.ptr + @as(u64, @intCast(o));
        const tbuf = try cuda.DeviceBuffer.fromHost(t.d, std.mem.sliceAsBytes(table));
        try t.blobs.put(t.gpa, try t.gpa.dupe(u8, name), .{ blob, tbuf });
        std.debug.print("  trellis {s}: {d} experts, {d} bytes\n", .{ name, meta.value.experts, bytes.len });
        return tbuf.ptr;
    }

    fn deinit(t: *Tables) void {
        for (t.blobs.values()) |*b| {
            b[0].free();
            b[1].free();
        }
    }
};

/// An argument's device pointer: the array itself, or the array it aliases (the same tensor passed twice).
/// 引数のデバイスポインタ: その配列自身か、別名で指す配列 (同じテンソルを 2 回渡すとき)。
fn arg(c: *const Call, name: []const u8) u64 {
    if (c.meta.value.object.get("alias")) |al| if (al.object.get(name)) |target| return c.ptr(target.string);
    return c.ptr(name);
}

fn uint(c: *const Call, name: []const u8) u32 {
    return @intCast(c.int(name));
}

fn shapeOf(c: *const Call, array: []const u8, i: usize) u32 {
    return @intCast(c.meta.value.object.get("arrays").?.object.get(array).?.object.get("shape").?.array.items[i].integer);
}

/// Replays every call under DIR/<case>/<n>-<fn>/ through experts.Kernels and compares each call's outputs byte for
/// byte; the trellis pointer tables are rebuilt over DIR/trellis/ blobs.
/// DIR/<case>/<n>-<fn>/ の全呼び出しを experts.Kernels で再生し、各呼び出しの出力をバイト単位で比べる。trellis の
/// ポインタ表は DIR/trellis/ の塊から作り直す。
fn expertsGate(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, optin: u32, root: []const u8) !u8 {
    var k = try glm.experts.Kernels.load(d, optin);
    defer k.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var tables: Tables = .{ .gpa = gpa, .io = io, .d = d, .root = root };
    defer tables.deinit();
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    var cases: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |e| if (e.kind == .directory and !std.mem.eql(u8, e.name, "trellis")) try cases.append(gpa, try gpa.dupe(u8, e.name));
    std.mem.sort([]const u8, cases.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    var failed: u8 = 0;
    var passed: usize = 0;
    for (cases.items) |case| {
        var cdir = try std.Io.Dir.cwd().openDir(io, try std.fmt.allocPrint(gpa, "{s}/{s}", .{ root, case }), .{ .iterate = true });
        defer cdir.close(io);
        var calls: std.ArrayList([]const u8) = .empty;
        var ci = cdir.iterate();
        while (try ci.next(io)) |e| if (e.kind == .directory) try calls.append(gpa, try gpa.dupe(u8, e.name));
        std.mem.sort([]const u8, calls.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                const a = std.fmt.parseInt(u32, x[0..std.mem.indexOfScalar(u8, x, '-').?], 10) catch 0;
                const b = std.fmt.parseInt(u32, y[0..std.mem.indexOfScalar(u8, y, '-').?], 10) catch 0;
                return a < b;
            }
        }.lt);
        for (calls.items) |call| {
            var c = try Call.open(gpa, io, d, try std.fmt.allocPrint(gpa, "{s}/{s}/{s}", .{ root, case, call }));
            defer c.deinit();
            const f = c.meta.value.object.get("fn").?.string;
            var desc: []const u8 = "";
            if (std.mem.eql(u8, f, "group")) {
                try k.group(stream, arg(&c, "pick"), arg(&c, "uids"), arg(&c, "ucount"), arg(&c, "members"), uint(&c, "R"), uint(&c, "slots"), uint(&c, "E"), shapeOf(&c, "members", 1));
                desc = try std.fmt.allocPrint(gpa, "R={d} slots={d} E={d}", .{ c.int("R"), c.int("slots"), c.int("E") });
            } else if (std.mem.eql(u8, f, "rot_in")) {
                if (!std.mem.eql(u8, c.meta.value.object.get("arrays").?.object.get("x").?.object.get("dtype").?.string, "bfloat16")) return error.UnportedRotInDtype;
                try k.rotIn(stream, arg(&c, "x"), uint(&c, "x_stride"), arg(&c, "pick"), arg(&c, "suh0"), arg(&c, "suh1"), arg(&c, "out0"), arg(&c, "out1"), uint(&c, "rows"), uint(&c, "K"), uint(&c, "slots"), uint(&c, "E"));
                desc = try std.fmt.allocPrint(gpa, "rows={d} K={d}", .{ c.int("rows"), c.int("K") });
            } else if (std.mem.eql(u8, f, "grouped")) {
                const ints = c.meta.value.object.get("ints").?.object;
                try k.grouped(stream, .{
                    .x0 = arg(&c, "X0"), .x1 = arg(&c, "X1"),
                    .tp0 = try tables.ptrs(ints.get("TP0").?.string), .tp1 = try tables.ptrs(ints.get("TP1").?.string),
                    .k2_0 = arg(&c, "B0"), .k2_1 = arg(&c, "B1"), .uids = arg(&c, "uids"), .ucount = arg(&c, "ucount"),
                    .members = arg(&c, "members"), .z = arg(&c, "Z"),
                    .K = uint(&c, "K"), .N = uint(&c, "N"), .P = uint(&c, "P"), .SK = uint(&c, "SK"),
                    .maxm = shapeOf(&c, "members", 1), .slots = uint(&c, "slots"), .nexp_max = shapeOf(&c, "uids", 0),
                    .mats = uint(&c, "mats"), .cb = uint(&c, "cb"), .nt = uint(&c, "nt"), .warps = uint(&c, "warps"),
                    .pf = uint(&c, "pf"), .lo = uint(&c, "lo"), .hi = uint(&c, "hi"), .ns = uint(&c, "ns"),
                });
                desc = try std.fmt.allocPrint(gpa, "mats={d} K={d} N={d} P={d} SK={d} {s}/{s}", .{ c.int("mats"), c.int("K"), c.int("N"), c.int("P"), c.int("SK"), ints.get("TP0").?.string, ints.get("TP1").?.string });
            } else if (std.mem.eql(u8, f, "gateup_epilogue")) {
                const limit: f32 = @bitCast(@as(u32, @intCast(c.int("limit_bits"))));
                try k.gateupEpilogue(stream, arg(&c, "Z"), arg(&c, "pick"), arg(&c, "svh_g"), arg(&c, "svh_u"), arg(&c, "suh_d"), arg(&c, "xd"), uint(&c, "rows"), uint(&c, "P"), uint(&c, "N"), uint(&c, "SK"), uint(&c, "slots"), uint(&c, "E"), limit, uint(&c, "act_mode"));
                desc = try std.fmt.allocPrint(gpa, "rows={d} P={d} N={d} SK={d} limit={d} act={d}", .{ c.int("rows"), c.int("P"), c.int("N"), c.int("SK"), limit, c.int("act_mode") });
            } else if (std.mem.eql(u8, f, "down_combine")) {
                try k.downCombine(stream, arg(&c, "Z"), arg(&c, "pick"), arg(&c, "svh_d"), arg(&c, "y"), arg(&c, "wts"), arg(&c, "out"), uint(&c, "rows"), uint(&c, "P"), uint(&c, "D"), uint(&c, "SK"), uint(&c, "slots"), uint(&c, "E"));
                desc = try std.fmt.allocPrint(gpa, "rows={d} P={d} D={d} SK={d}", .{ c.int("rows"), c.int("P"), c.int("D"), c.int("SK") });
            } else {
                std.debug.print("SKIP {s}/{s}: no replay for {s}\n", .{ case, call, f });
                failed += 1;
                continue;
            }
            try stream.synchronize();
            var ok = true;
            for (c.meta.value.object.get("outputs").?.array.items) |o| ok = (try c.same(o.string)) and ok;
            std.debug.print("{s} {s}/{s} {s}\n", .{ if (ok) "PASS" else "FAIL", case, call, desc });
            if (ok) passed += 1 else failed += 1;
        }
    }
    std.debug.print("experts gate: {d} passed, {d} failed\n", .{ passed, failed });
    return failed;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 8 and std.mem.eql(u8, args[1], "digests")) return digests(init.arena.allocator(), init.io, args);
    if (args.len >= 3 and std.mem.eql(u8, args[1], "linear")) {
        var drv = try cuda.Driver.open();
        defer drv.close();
        var cx = try cuda.Context.init(&drv, 0);
        defer cx.deinit();
        return if (try linearGate(init.arena.allocator(), init.io, &drv, args[2]) == 0) 0 else 1;
    }
    if (args.len >= 3 and std.mem.eql(u8, args[1], "experts")) {
        var drv = try cuda.Driver.open();
        defer drv.close();
        var cx = try cuda.Context.init(&drv, 0);
        defer cx.deinit();
        const optin: u32 = @intCast(try cx.attribute(.max_shared_memory_per_block_optin));
        return if (try expertsGate(std.heap.page_allocator, init.io, &drv, optin, args[2]) == 0) 0 else 1;
    }
    if (args.len < 3 or !std.mem.eql(u8, args[1], "bmm")) {
        std.debug.print("usage: tf-glm53-test bmm FIXTURE_DIR | linear DIR | experts DIR | digests CKPT RANK WORLD LAYERS STORED_WORLD OUT\n", .{});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const failed = try bmm(init.arena.allocator(), init.io, &driver, args[2]);
    return if (failed == 0) 0 else 1;
}
