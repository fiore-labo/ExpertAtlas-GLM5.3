//! The DFlash2 drafter's weights for one rank, quantized to 4 bits on the host (q4.zig): q / k / v / gate / up cut by
//! rows, o / down by columns, fc and the conv kernel projections replicated. Norms, conv base kernels and the
//! selector's hidden projection stay bf16; its codebooks are fp32 on the host.
//! DFlash2 drafter の 1 ランク分の重み。ホストで 4 bit に量子化する (q4.zig): q / k / v / gate / up は行で、o / down は
//! 列で切り、fc と conv カーネルの projection は複製する。norm・conv の base カーネル・selector の hidden projection は
//! bf16 のまま、selector の codebook はホスト上の fp32。
const std = @import("std");
const core = @import("core");
const q4 = @import("q4.zig");
const st = core.safetensors;

pub const Config = struct {
    hidden: usize,
    head_dim: usize,
    heads: usize, // this rank's / このランクの分
    kv_heads: usize, // this rank's / このランクの分
    inter: usize, // this rank's MLP width / このランクの MLP 幅
    layers: usize,
    eps: f32,
    rope_theta: f64,
    mask_id: u32,
    conv_gs: usize,
    block: usize,
    top_k: usize,
    selector_rank: usize,
    taps: [8]usize = undefined,
    n_taps: usize = 0,
    window: usize, // sliding_window - 1
    causal: bool,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const o = parsed.value.object;
        const dc = o.get("dflash_config").?.object;
        if (dc.get("conv_kernel_size").?.integer != 2) return error.UnportedConvKernel;
        const heads: usize = @intCast(o.get("num_attention_heads").?.integer);
        const kvh: usize = @intCast(o.get("num_key_value_heads").?.integer);
        const inter: usize = @intCast(o.get("intermediate_size").?.integer);
        if (heads % world != 0 or kvh % world != 0 or inter % world != 0) return error.UnevenSplit;
        var c: Config = .{
            .hidden = @intCast(o.get("hidden_size").?.integer),
            .head_dim = @intCast(o.get("head_dim").?.integer),
            .heads = heads / world,
            .kv_heads = kvh / world,
            .inter = inter / world,
            .layers = @intCast(o.get("num_hidden_layers").?.integer),
            .eps = @floatCast(numberOf(o.get("rms_norm_eps").?)),
            .rope_theta = numberOf(o.get("rope_parameters").?.object.get("rope_theta").?),
            .mask_id = @intCast(dc.get("mask_token_id").?.integer),
            .conv_gs = @intCast(dc.get("conv_group_size").?.integer),
            .block = @intCast(dc.get("block_size").?.integer),
            .top_k = @intCast(dc.get("selector_top_k").?.integer),
            .selector_rank = @intCast(dc.get("selector_rank").?.integer),
            .window = @as(usize, @intCast(o.get("sliding_window").?.integer)) - 1,
            .causal = if (o.get("is_causal")) |v| v.bool else true,
        };
        for (dc.get("target_layer_ids").?.array.items) |t| {
            c.taps[c.n_taps] = @intCast(t.integer);
            c.n_taps += 1;
        }
        return c;
    }
};

fn numberOf(v: std.json.Value) f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => 0,
    };
}

/// One drafter layer's quantized projections and bf16 vectors (host side; drafter.zig uploads them).
/// drafter 1 層の量子化した projection と bf16 のベクトル (ホスト側。drafter.zig が GPU に載せる)。
pub const Layer = struct {
    in_norm: []const u16,
    post_norm: []const u16,
    a_base: []const u16, // [2, 2, D]
    m_base: []const u16,
    q_norm: []const u16,
    k_norm: []const u16,
    a_kp: q4.Packed,
    m_kp: q4.Packed,
    qkv: q4.Packed, // [q heads | k heads | v heads] of this rank / このランクの分
    kv: q4.Packed, // [k heads | v heads]
    o: q4.Packed, // [D, heads * head_dim] (a row-parallel partial) / 行並列で部分和を出す
    gu: q4.Packed, // [gate | up]
    down: q4.Packed, // [D, inter]
};

pub const Weights = struct {
    cfg: Config,
    file: st.File,
    fc: q4.Packed,
    hidden_norm: []const u16,
    norm: []const u16,
    hproj: []const u16, // [selector_rank, D] bf16
    pred: []f32, // [V, selector_rank]
    succ: []f32,
    layers: []Layer,

    pub fn close(w: *Weights, io: std.Io) void {
        w.file.close(io);
    }
};

fn bf16s(t: st.Tensor) ![]const u16 {
    if (t.dtype != .bf16) return error.NotBf16;
    return @alignCast(std.mem.bytesAsSlice(u16, t.bytes));
}

fn tensor(f: *const st.File, name: []const u8) !st.Tensor {
    return f.get(name) orelse {
        std.log.err("drafter: no tensor {s}", .{name});
        return error.MissingTensor;
    };
}

/// Rows [r0, r0 + n) of a [rows, cols] bf16 matrix appended to `out`.
/// [rows, cols] の bf16 行列の行 [r0, r0 + n) を `out` の後ろに加える。
fn rows(out: *std.ArrayList(u16), gpa: std.mem.Allocator, t: st.Tensor, r0: usize, n: usize) !void {
    const width = t.dim(1);
    const all = try bf16s(t);
    try out.appendSlice(gpa, all[r0 * width ..][0 .. n * width]);
}

/// Columns [c0, c0 + n) of a [rows, cols] bf16 matrix (each row's slice).
/// [rows, cols] の bf16 行列の列 [c0, c0 + n) (各行から切り出す)。
fn cols(gpa: std.mem.Allocator, t: st.Tensor, c0: usize, n: usize) ![]u16 {
    const r = t.dim(0);
    const cc = t.dim(1);
    const all = try bf16s(t);
    const out = try gpa.alloc(u16, r * n);
    for (0..r) |i| @memcpy(out[i * n ..][0..n], all[i * cc + c0 ..][0..n]);
    return out;
}

fn quantRows(gpa: std.mem.Allocator, f: *const st.File, parts: []const struct { name: []const u8, r0: usize, n: usize }) !q4.Packed {
    var buf: std.ArrayList(u16) = .empty;
    defer buf.deinit(gpa);
    var k: usize = 0;
    var n: usize = 0;
    for (parts) |p| {
        const t = try tensor(f, p.name);
        k = t.dim(1);
        try rows(&buf, gpa, t, p.r0, p.n);
        n += p.n;
    }
    return q4.quantize(gpa, buf.items, n, k);
}

fn quantCols(gpa: std.mem.Allocator, f: *const st.File, name: []const u8, c0: usize, n: usize) !q4.Packed {
    const t = try tensor(f, name);
    const m = try cols(gpa, t, c0, n);
    defer gpa.free(m);
    return q4.quantize(gpa, m, t.dim(0), n);
}

fn quantAll(gpa: std.mem.Allocator, f: *const st.File, name: []const u8) !q4.Packed {
    const t = try tensor(f, name);
    return q4.quantize(gpa, try bf16s(t), t.dim(0), t.dim(1));
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, rank: usize, world: usize) !Weights {
    const cfg = try Config.read(gpa, io, dir, world);
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(path);
    var f = try st.File.open(gpa, io, path);
    errdefer f.close(io);
    const H = cfg.heads;
    const KV = cfg.kv_heads;
    const hd = cfg.head_dim;
    const it = cfg.inter;
    const layers = try gpa.alloc(Layer, cfg.layers);
    var name: [128]u8 = undefined;
    for (layers, 0..) |*L, i| {
        const p = struct {
            fn n(b: []u8, li: usize, s: []const u8) []const u8 {
                return std.fmt.bufPrint(b, "layers.{d}.{s}", .{ li, s }) catch unreachable;
            }
        };
        var qn: [128]u8 = undefined;
        var kn: [128]u8 = undefined;
        var vn: [128]u8 = undefined;
        const qname = p.n(&qn, i, "self_attn.q_proj.weight");
        const kname = p.n(&kn, i, "self_attn.k_proj.weight");
        const vname = p.n(&vn, i, "self_attn.v_proj.weight");
        var gn: [128]u8 = undefined;
        var un: [128]u8 = undefined;
        L.* = .{
            .in_norm = try bf16s(try tensor(&f, p.n(&name, i, "input_layernorm.weight"))),
            .post_norm = try bf16s(try tensor(&f, p.n(&name, i, "post_attention_layernorm.weight"))),
            .a_base = try bf16s(try tensor(&f, p.n(&name, i, "attention_conv.base_kernel"))),
            .m_base = try bf16s(try tensor(&f, p.n(&name, i, "mlp_conv.base_kernel"))),
            .q_norm = try bf16s(try tensor(&f, p.n(&name, i, "self_attn.q_norm.weight"))),
            .k_norm = try bf16s(try tensor(&f, p.n(&name, i, "self_attn.k_norm.weight"))),
            .a_kp = try quantAll(gpa, &f, p.n(&name, i, "attention_conv.kernel_projection.weight")),
            .m_kp = try quantAll(gpa, &f, p.n(&name, i, "mlp_conv.kernel_projection.weight")),
            .qkv = try quantRows(gpa, &f, &.{ .{ .name = qname, .r0 = rank * H * hd, .n = H * hd }, .{ .name = kname, .r0 = rank * KV * hd, .n = KV * hd }, .{ .name = vname, .r0 = rank * KV * hd, .n = KV * hd } }),
            .kv = try quantRows(gpa, &f, &.{ .{ .name = kname, .r0 = rank * KV * hd, .n = KV * hd }, .{ .name = vname, .r0 = rank * KV * hd, .n = KV * hd } }),
            .o = try quantCols(gpa, &f, p.n(&name, i, "self_attn.o_proj.weight"), rank * H * hd, H * hd),
            .gu = try quantRows(gpa, &f, &.{ .{ .name = p.n(&gn, i, "mlp.gate_proj.weight"), .r0 = rank * it, .n = it }, .{ .name = p.n(&un, i, "mlp.up_proj.weight"), .r0 = rank * it, .n = it } }),
            .down = try quantCols(gpa, &f, p.n(&name, i, "mlp.down_proj.weight"), rank * it, it),
        };
    }
    const pred_t = try tensor(&f, "candidate_selector.predecessor_codebook");
    const succ_t = try tensor(&f, "candidate_selector.successor_codebook");
    return .{
        .cfg = cfg,
        .file = f,
        .fc = try quantAll(gpa, &f, "fc.weight"),
        .hidden_norm = try bf16s(try tensor(&f, "hidden_norm.weight")),
        .norm = try bf16s(try tensor(&f, "norm.weight")),
        .hproj = try bf16s(try tensor(&f, "candidate_selector.hidden_projection.weight")),
        .pred = try toF32(gpa, pred_t),
        .succ = try toF32(gpa, succ_t),
        .layers = layers,
    };
}

/// A codebook as fp32 (bf16 or fp32 on disk).
/// codebook を fp32 にする (ディスク上は bf16 か fp32)。
fn toF32(gpa: std.mem.Allocator, t: st.Tensor) ![]f32 {
    const out = try gpa.alloc(f32, t.numel());
    switch (t.dtype) {
        .bf16 => for (try bf16s(t), out) |h, *o| {
            o.* = @bitCast(@as(u32, h) << 16);
        },
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes),
        else => return error.CodebookDtype,
    }
    return out;
}
