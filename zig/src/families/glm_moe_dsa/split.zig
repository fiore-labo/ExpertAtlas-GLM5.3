//! Which part of each checkpoint tensor a rank holds, for any world size. EXL3 linears are cut along whole 16x16
//! tiles and whole 128-wide Hadamard blocks, so a share is a valid EXL3 tensor.
//! 各ランクがチェックポイントの各テンソルのどの部分を持つか (world サイズは任意)。EXL3 の linear は 16x16 タイルと
//! 128 幅の Hadamard ブロックの境目で切るので、分けた部分もそのまま有効な EXL3 テンソルになる。
const std = @import("std");

pub const had = 128;
pub const tile = 16;

/// rep: every rank holds all of it; row: the leading axis is cut; dim1: the second axis is cut.
/// rep: 全ランクが全体を持つ。row: 先頭の軸を切る。dim1: 2 番目の軸を切る。
pub const Rule = enum { rep, row, dim1 };

const Proj = enum { out, in, rep };

/// The EXL3 linear a tensor belongs to (by its projection), or null for a plain tensor.
/// テンソルが属する EXL3 linear (projection で判定)。普通のテンソルなら null。
fn exl3Proj(name: []const u8) ?struct { proj: Proj, part: []const u8 } {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return null;
    const part = name[dot + 1 ..];
    const parts = [_][]const u8{ "trellis", "suh", "svh", "mul1", "mcg" };
    for (parts) |p| {
        if (std.mem.eql(u8, part, p)) break;
    } else return null;
    const stem = name[0..dot];
    const replicated = [_][]const u8{ ".self_attn.q_a_proj", ".self_attn.kv_a_proj_with_mqa", ".self_attn.indexer.wq_b" };
    for (replicated) |s| if (std.mem.endsWith(u8, stem, s)) return .{ .proj = .rep, .part = part };
    const outs = [_][]const u8{ "gate_proj", "up_proj", ".self_attn.q_b_proj" };
    const ins = [_][]const u8{ "down_proj", ".self_attn.o_proj" };
    if (!isLinearStem(stem)) return null;
    for (outs) |s| if (std.mem.endsWith(u8, stem, s)) return .{ .proj = .out, .part = part };
    for (ins) |s| if (std.mem.endsWith(u8, stem, s)) return .{ .proj = .in, .part = part };
    return null;
}

/// The EXL3 linears that are split: routed / shared experts, the dense MLP, attention projections.
/// 分割する EXL3 linear: routed / shared expert、dense MLP、attention の projection。
fn isLinearStem(stem: []const u8) bool {
    const marks = [_][]const u8{ ".mlp.experts.", ".mlp.shared_experts.", ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".self_attn." };
    for (marks) |m| if (std.mem.indexOf(u8, stem, m) != null) return true;
    return false;
}

pub fn rule(name: []const u8) !Rule {
    if (exl3Proj(name)) |e| {
        if (e.proj == .rep) return .rep;
        const p = e.part;
        if (std.mem.eql(u8, p, "mul1") or std.mem.eql(u8, p, "mcg")) return .rep;
        return switch (e.proj) {
            .out => if (std.mem.eql(u8, p, "trellis")) .dim1 else if (std.mem.eql(u8, p, "svh")) .row else .rep,
            .in => if (std.mem.eql(u8, p, "trellis") or std.mem.eql(u8, p, "suh")) .row else .rep,
            .rep => .rep,
        };
    }
    if (std.mem.endsWith(u8, name, ".self_attn.kv_b_proj.weight") or std.mem.eql(u8, name, "lm_head.weight")) return .row;
    const reps = [_][]const u8{ "model.norm.weight", "_layernorm.weight", ".mlp.gate.weight", ".mlp.gate.e_score_correction_bias", ".self_attn.indexer.wk.weight", ".self_attn.indexer.weights_proj.weight", ".self_attn.indexer.k_norm.weight", ".self_attn.indexer.k_norm.bias", ".eh_proj.weight", ".enorm.weight", ".hnorm.weight", ".shared_head.norm.weight" };
    if (std.mem.startsWith(u8, name, "model.embed_tokens.")) return .rep;
    for (reps) |s| if (std.mem.endsWith(u8, name, s)) return .rep;
    return error.NoSplitRule;
}

/// Checks a cut: the cut axis divides evenly, and a share never straddles a Hadamard block.
/// 切り方を確かめる: 切る軸が割り切れ、どの部分も Hadamard ブロックをまたがない。
pub fn checkCut(name: []const u8, r: Rule, shape: []const usize, world: usize) !void {
    if (r == .rep) return;
    const axis: usize = if (r == .row) 0 else 1;
    if (shape.len <= axis or shape[axis] % world != 0) return error.UnevenSplit;
    if (std.mem.endsWith(u8, name, ".trellis")) {
        if ((shape[axis] / world) % (had / tile) != 0) return error.StraddlesHadamard;
    } else if (std.mem.endsWith(u8, name, ".suh") or std.mem.endsWith(u8, name, ".svh")) {
        if ((shape[0] / world) % had != 0) return error.StraddlesHadamard;
    }
}

/// Rank `rank`'s share of the cut axis: [lo, hi).
/// ランク `rank` が持つ、切った軸の範囲 [lo, hi)。
pub fn share(r: Rule, shape: []const usize, rank: usize, world: usize) struct { lo: usize, hi: usize } {
    if (r == .rep) return .{ .lo = 0, .hi = shape[0] };
    const n = shape[if (r == .row) 0 else 1] / world;
    return .{ .lo = rank * n, .hi = (rank + 1) * n };
}

test "split rules match split.py on GLM-5.3's tensor names" {
    const t = std.testing;
    try t.expectEqual(Rule.dim1, try rule("model.layers.3.mlp.experts.7.gate_proj.trellis"));
    try t.expectEqual(Rule.row, try rule("model.layers.3.mlp.experts.7.up_proj.svh"));
    try t.expectEqual(Rule.rep, try rule("model.layers.3.mlp.experts.7.up_proj.suh"));
    try t.expectEqual(Rule.row, try rule("model.layers.3.mlp.experts.7.down_proj.trellis"));
    try t.expectEqual(Rule.row, try rule("model.layers.3.mlp.experts.7.down_proj.suh"));
    try t.expectEqual(Rule.rep, try rule("model.layers.3.mlp.experts.7.down_proj.svh"));
    try t.expectEqual(Rule.rep, try rule("model.layers.3.mlp.experts.7.down_proj.mcg"));
    try t.expectEqual(Rule.rep, try rule("model.layers.0.self_attn.q_a_proj.trellis"));
    try t.expectEqual(Rule.dim1, try rule("model.layers.0.self_attn.q_b_proj.trellis"));
    try t.expectEqual(Rule.row, try rule("model.layers.0.self_attn.o_proj.trellis"));
    try t.expectEqual(Rule.rep, try rule("model.layers.0.self_attn.indexer.wq_b.svh"));
    try t.expectEqual(Rule.row, try rule("model.layers.0.self_attn.kv_b_proj.weight"));
    try t.expectEqual(Rule.dim1, try rule("model.layers.0.mlp.gate_proj.trellis"));
    try t.expectEqual(Rule.row, try rule("lm_head.weight"));
    try t.expectEqual(Rule.rep, try rule("model.embed_tokens.weight"));
    try t.expectEqual(Rule.rep, try rule("model.layers.5.post_attention_layernorm.weight"));
    try t.expectError(error.NoSplitRule, rule("model.layers.5.mystery"));
    try checkCut("x.trellis", .dim1, &.{ 128, 128, 48 }, 4);
    try t.expectError(error.StraddlesHadamard, checkCut("x.trellis", .dim1, &.{ 128, 24, 48 }, 4));
    const s = share(.row, &.{ 2048, 6144 }, 1, 4);
    try t.expectEqual(@as(usize, 512), s.lo);
}
