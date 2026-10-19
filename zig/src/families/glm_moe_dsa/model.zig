//! One rank's GLM-5.3 layers on the GPU: bf16 norms and indexer projections, EXL3 linears (linear.zig), and kv_b split
//! into the absorb (wk) and expand (wv) blocks.
//! 1 ランク分の GLM-5.3 の層を GPU に置く: bf16 の norm と indexer の projection、EXL3 の linear (linear.zig)、
//! absorb 用 (wk) と expand 用 (wv) に分けた kv_b。
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const linear = @import("linear.zig");
const moe = @import("moe.zig");

pub const Buf = cuda.DeviceBuffer;

pub const Indexer = struct { wq_b: linear.Linear, wk: Buf, weights_proj: Buf, k_norm_w: Buf, k_norm_b: Buf };
pub const Mlp = struct { gate: linear.Linear, up: linear.Linear, down: linear.Linear };

pub const Layer = struct {
    index: usize,
    input_norm: Buf,
    post_attn_norm: Buf,
    q_a: linear.Linear,
    q_a_norm: Buf,
    kv_a: linear.Linear,
    kv_a_norm: Buf,
    q_b: linear.Linear,
    wk: Buf, // bf16 [H, nope, lw]: kv_b's first nope rows a head / 各ヘッドの kv_b の先頭 nope 行
    wv: Buf, // bf16 [H, v, lw]: kv_b's last v rows a head / 各ヘッドの kv_b の後ろ v 行
    o_proj: linear.Linear,
    indexer: ?Indexer,
    router_w: ?Buf, // bf16 [E, D] (MoE layers) / bf16 [E, D] (MoE 層)
    router_bias: ?Buf, // fp32 [E] (e_score_correction_bias as float) / fp32 [E] (e_score_correction_bias を float に)
    // the dense MLP (layers below first_dense) or the shared expert
    // dense MLP (first_dense より下の層) か shared expert
    mlp: Mlp,
    // routed experts (MoE layers), loaded by moe.load
    // routed expert (MoE 層)。moe.load が読む
    experts: ?moe.Routed = null,
    /// DCP 2 (TF_GLM53_PEER_Q): the group partner's q_b and wk, so this rank computes the partner's packed queries of
    /// a prompt chunk itself (from q_a's rows, which every rank holds) instead of receiving them.
    /// DCP 2 (TF_GLM53_PEER_Q): グループの相手の q_b と wk。これで prompt チャンクに対する相手の packed query を、
    /// 受け取らずにこのランクで計算する (q_a の行は全ランクが持つ)。
    peer_q_b: ?linear.Linear = null,
    peer_wk: ?Buf = null,
};

fn bf(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *weights.Reader, name: []const u8) !Buf {
    var s = try r.get(name);
    defer s.deinit(gpa);
    return Buf.fromHost(d, s.bytes);
}

/// The partner's q_b and wk of `layer`: `r` is a reader opened for the partner's rank.
/// `layer` の相手の q_b と wk。`r` は相手のランク用に開いた reader。
pub fn loadPeerQuery(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *weights.Reader, cfg: Config, layer: usize, heads: usize) !struct { q_b: linear.Linear, wk: Buf } {
    var nb: [256]u8 = undefined;
    var kvb = try r.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.kv_b_proj.weight", .{layer}));
    defer kvb.deinit(gpa);
    const per = cfg.qk_nope + cfg.v_head;
    const row = cfg.kv_lora * 2;
    if (kvb.bytes.len != heads * per * row) return error.BadKvB;
    const wk = try gpa.alloc(u8, heads * cfg.qk_nope * row);
    defer gpa.free(wk);
    for (0..heads) |h| @memcpy(wk[h * cfg.qk_nope * row ..][0 .. cfg.qk_nope * row], kvb.bytes[h * per * row ..][0 .. cfg.qk_nope * row]);
    return .{
        .q_b = try linear.load(gpa, d, r, try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.q_b_proj", .{layer})),
        .wk = try Buf.fromHost(d, wk),
    };
}

/// The tensors of `layer` (routed experts are loaded by moe.load, not here).
/// `layer` のテンソルを読む (routed expert はここでなく moe.load が読む)。
pub fn loadLayer(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *weights.Reader, cfg: Config, layer: usize, heads: usize) !Layer {
    return loadLayerAs(gpa, d, r, cfg, layer, heads, cfg.isMoe(layer));
}

/// `loadLayer` with the layer's kind given: the MTP layer (index cfg.layers) is an MoE layer past the decoder's.
/// 層の種類を指定する `loadLayer`。MTP 層 (index cfg.layers) はデコーダの層の後に続く MoE 層。
pub fn loadLayerAs(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *weights.Reader, cfg: Config, layer: usize, heads: usize, is_moe: bool) !Layer {
    var b: [192]u8 = undefined;
    const p = try std.fmt.bufPrint(&b, "model.layers.{d}", .{layer});
    var nb: [256]u8 = undefined;
    const n = struct {
        fn f(buf: []u8, pre: []const u8, rest: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}.{s}", .{ pre, rest }) catch unreachable;
        }
    }.f;
    // kv_b: this rank's heads x (nope + v) rows of lw -> wk [H, nope, lw] and wv [H, v, lw], each contiguous
    // kv_b: このランクのヘッド x (nope + v) 行 (各 lw) -> wk [H, nope, lw] と wv [H, v, lw]、それぞれ連続
    var kvb = try r.get(n(&nb, p, "self_attn.kv_b_proj.weight"));
    defer kvb.deinit(gpa);
    const per = cfg.qk_nope + cfg.v_head;
    const row = cfg.kv_lora * 2;
    if (kvb.bytes.len != heads * per * row) return error.BadKvB;
    const wk = try gpa.alloc(u8, heads * cfg.qk_nope * row);
    defer gpa.free(wk);
    const wv = try gpa.alloc(u8, heads * cfg.v_head * row);
    defer gpa.free(wv);
    for (0..heads) |h| {
        @memcpy(wk[h * cfg.qk_nope * row ..][0 .. cfg.qk_nope * row], kvb.bytes[h * per * row ..][0 .. cfg.qk_nope * row]);
        @memcpy(wv[h * cfg.v_head * row ..][0 .. cfg.v_head * row], kvb.bytes[(h * per + cfg.qk_nope) * row ..][0 .. cfg.v_head * row]);
    }
    var idx: ?Indexer = null;
    if (r.has(n(&nb, p, "self_attn.indexer.wq_b.trellis"))) idx = .{
        .wq_b = try linear.load(gpa, d, r, n(&nb, p, "self_attn.indexer.wq_b")),
        .wk = try bf(gpa, d, r, n(&nb, p, "self_attn.indexer.wk.weight")),
        .weights_proj = try bf(gpa, d, r, n(&nb, p, "self_attn.indexer.weights_proj.weight")),
        .k_norm_w = try bf(gpa, d, r, n(&nb, p, "self_attn.indexer.k_norm.weight")),
        .k_norm_b = try bf(gpa, d, r, n(&nb, p, "self_attn.indexer.k_norm.bias")),
    };
    var router_bias: ?Buf = null;
    if (is_moe) {
        var bias = try r.get(n(&nb, p, "mlp.gate.e_score_correction_bias"));
        defer bias.deinit(gpa);
        router_bias = try toF32(gpa, d, bias);
    }
    const mp = if (is_moe) "mlp.shared_experts" else "mlp";
    var mb: [128]u8 = undefined;
    return .{
        .index = layer,
        .input_norm = try bf(gpa, d, r, n(&nb, p, "input_layernorm.weight")),
        .post_attn_norm = try bf(gpa, d, r, n(&nb, p, "post_attention_layernorm.weight")),
        .q_a = try linear.load(gpa, d, r, n(&nb, p, "self_attn.q_a_proj")),
        .q_a_norm = try bf(gpa, d, r, n(&nb, p, "self_attn.q_a_layernorm.weight")),
        .kv_a = try linear.load(gpa, d, r, n(&nb, p, "self_attn.kv_a_proj_with_mqa")),
        .kv_a_norm = try bf(gpa, d, r, n(&nb, p, "self_attn.kv_a_layernorm.weight")),
        .q_b = try linear.load(gpa, d, r, n(&nb, p, "self_attn.q_b_proj")),
        .wk = try Buf.fromHost(d, wk),
        .wv = try Buf.fromHost(d, wv),
        .o_proj = try linear.load(gpa, d, r, n(&nb, p, "self_attn.o_proj")),
        .indexer = idx,
        .router_w = if (is_moe) try bf(gpa, d, r, n(&nb, p, "mlp.gate.weight")) else null,
        .router_bias = router_bias,
        .mlp = .{
            .gate = try linear.load(gpa, d, r, n(&nb, p, try std.fmt.bufPrint(&mb, "{s}.gate_proj", .{mp}))),
            .up = try linear.load(gpa, d, r, n(&nb, p, try std.fmt.bufPrint(&mb, "{s}.up_proj", .{mp}))),
            .down = try linear.load(gpa, d, r, n(&nb, p, try std.fmt.bufPrint(&mb, "{s}.down_proj", .{mp}))),
        },
    };
}

/// bf16 / fp16 / fp32 -> fp32 on the host (an exact widening).
/// ホスト上で bf16 / fp16 / fp32 -> fp32 (誤差のない拡張)。
fn toF32(gpa: std.mem.Allocator, d: *const cuda.Driver, s: weights.Share) !Buf {
    var n: usize = 1;
    for (s.shape[0..s.rank]) |x| n *= x;
    const out = try gpa.alloc(u8, n * 4);
    defer gpa.free(out);
    for (0..n) |i| {
        const v: f32 = switch (s.dtype) {
            .f32 => @bitCast(std.mem.readInt(u32, s.bytes[i * 4 ..][0..4], .little)),
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, s.bytes[i * 2 ..][0..2], .little)) << 16),
            .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, s.bytes[i * 2 ..][0..2], .little)))),
            else => return error.BadRouterBias,
        };
        std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(v), .little);
    }
    return Buf.fromHost(d, out);
}
