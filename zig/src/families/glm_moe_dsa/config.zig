//! GLM-5.3's settings (model_type glm_moe_dsa) from config.json; other models and unsupported options are refused.
//! GLM-5.3 の設定 (model_type glm_moe_dsa) を config.json から読む。他のモデルや未対応の設定は拒否する。
const std = @import("std");

pub const max_layers = 96;

pub const Indexer = enum { full, shared };

pub const Config = struct {
    hidden: usize,
    layers: usize,
    mtp_layers: usize,
    first_dense: usize, // first_k_dense_replace: layers below it are dense MLPs / これより下の層は dense MLP
    inter: usize, // dense MLP width / dense MLP の幅
    moe_inter: usize,
    experts: usize,
    shared_experts: usize,
    top_k: usize,
    routed_scaling: f32,
    norm_topk: bool,
    heads: usize,
    q_lora: usize,
    kv_lora: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head: usize,
    rope_theta: f64,
    rope_interleave: bool,
    index_heads: usize,
    index_dim: usize,
    index_topk: usize,
    indexer: [max_layers]Indexer = undefined,
    max_positions: usize,
    eps: f32,
    vocab: usize,
    eos: [8]u32 = undefined, // eos_token_id (an int or a list) / eos_token_id (整数かリスト)
    n_eos: usize = 0,

    pub fn isEos(c: *const Config, t: u32) bool {
        return std.mem.indexOfScalar(u32, c.eos[0..c.n_eos], t) != null;
    }

    /// One token's MLA cache entry: the kv latent and the shared RoPE key.
    /// 1 トークン分の MLA キャッシュの項目: kv latent と共有の RoPE キー。
    pub fn latentWidth(c: Config) usize {
        return c.kv_lora + c.qk_rope;
    }

    pub fn isMoe(c: Config, layer: usize) bool {
        return layer >= c.first_dense and layer < c.layers;
    }

    /// Whether a layer computes its own top-k (else it reuses the last full layer's).
    /// 層が自分で top-k を計算するか (しなければ直前の full 層の結果を使う)。
    pub fn fullIndexer(c: Config, layer: usize) bool {
        return c.indexer[layer] == .full;
    }

    /// Reads `dir`/config.json.
    /// `dir`/config.json を読む。
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        return parse(gpa, text);
    }
};

pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var o = parsed.value.object;
    if (o.get("text_config")) |t| if (t == .object) {
        o = t.object;
    };
    const mt = o.get("model_type") orelse return error.BadConfig;
    if (mt != .string or !std.mem.eql(u8, mt.string, "glm_moe_dsa")) return error.NotGlmMoeDsa;
    if (intOr(o, "n_group", 1) != 1 or intOr(o, "topk_group", 1) != 1) return error.GroupedRoutingUnsupported;
    if (!strIs(o, "scoring_func", "sigmoid") or !strIs(o, "topk_method", "noaux_tc")) return error.RoutingUnsupported;
    const rope = o.get("rope_parameters") orelse o.get("rope_scaling");
    var theta: f64 = if (o.get("rope_theta")) |v| try number(v) else 10000.0;
    if (rope) |r| if (r == .object) {
        if (r.object.get("rope_type")) |t| if (t == .string and !std.mem.eql(u8, t.string, "default")) return error.RopeScalingUnsupported;
        if (r.object.get("rope_theta")) |v| theta = try number(v);
    };
    var c: Config = .{
        .hidden = try int(o, "hidden_size"),
        .layers = try int(o, "num_hidden_layers"),
        .mtp_layers = intOr(o, "num_nextn_predict_layers", 0),
        .first_dense = try int(o, "first_k_dense_replace"),
        .inter = try int(o, "intermediate_size"),
        .moe_inter = try int(o, "moe_intermediate_size"),
        .experts = try int(o, "n_routed_experts"),
        .shared_experts = try int(o, "n_shared_experts"),
        .top_k = try int(o, "num_experts_per_tok"),
        .routed_scaling = @floatCast(try number(o.get("routed_scaling_factor") orelse return error.BadConfig)),
        .norm_topk = (o.get("norm_topk_prob") orelse return error.BadConfig).bool,
        .heads = try int(o, "num_attention_heads"),
        .q_lora = try int(o, "q_lora_rank"),
        .kv_lora = try int(o, "kv_lora_rank"),
        .qk_nope = try int(o, "qk_nope_head_dim"),
        .qk_rope = try int(o, "qk_rope_head_dim"),
        .v_head = try int(o, "v_head_dim"),
        .rope_theta = theta,
        .rope_interleave = if (o.get("rope_interleave")) |v| v.bool else true,
        .index_heads = try int(o, "index_n_heads"),
        .index_dim = try int(o, "index_head_dim"),
        .index_topk = try int(o, "index_topk"),
        .max_positions = try int(o, "max_position_embeddings"),
        .eps = @floatCast(try number(o.get("rms_norm_eps") orelse return error.BadConfig)),
        .vocab = try int(o, "vocab_size"),
    };
    if (o.get("eos_token_id")) |e| switch (e) {
        .integer => |v| {
            c.eos[0] = @intCast(v);
            c.n_eos = 1;
        },
        .array => |arr| for (arr.items) |v| {
            if (v != .integer or c.n_eos == c.eos.len) return error.BadConfig;
            c.eos[c.n_eos] = @intCast(v.integer);
            c.n_eos += 1;
        },
        else => return error.BadConfig,
    };
    if (c.layers > max_layers) return error.TooManyLayers;
    const types = o.get("indexer_types") orelse return error.BadIndexerTypes;
    if (types != .array or types.array.items.len != c.layers) return error.BadIndexerTypes;
    for (types.array.items, 0..) |t, i| {
        if (t != .string) return error.BadIndexerTypes;
        c.indexer[i] = std.meta.stringToEnum(Indexer, t.string) orelse return error.BadIndexerTypes;
    }
    if (c.indexer[0] != .full) return error.BadIndexerTypes;
    return c;
}

fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadConfig;
    if (v != .integer or v.integer < 0) return error.BadConfig;
    return @intCast(v.integer);
}

fn intOr(o: std.json.ObjectMap, key: []const u8, default: usize) usize {
    const v = o.get(key) orelse return default;
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else default;
}

fn strIs(o: std.json.ObjectMap, key: []const u8, want: []const u8) bool {
    const v = o.get(key) orelse return true; // absent: the default, accepted / 無ければ既定値として受け入れる
    return v == .string and std.mem.eql(u8, v.string, want);
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

test "GLM-5.3's config parses with its indexer pattern and refuses other families" {
    const text =
        \\{"model_type": "glm_moe_dsa", "hidden_size": 6144, "num_hidden_layers": 4, "num_nextn_predict_layers": 1,
        \\ "first_k_dense_replace": 3, "intermediate_size": 12288, "moe_intermediate_size": 2048, "n_routed_experts": 256,
        \\ "n_shared_experts": 1, "num_experts_per_tok": 8, "routed_scaling_factor": 2.5, "norm_topk_prob": true,
        \\ "num_attention_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
        \\ "v_head_dim": 256, "rope_parameters": {"rope_type": "default", "rope_theta": 8000000}, "index_n_heads": 32,
        \\ "index_head_dim": 128, "index_topk": 2048, "indexer_types": ["full", "full", "full", "shared"],
        \\ "max_position_embeddings": 1048576, "rms_norm_eps": 1e-05, "vocab_size": 154880}
    ;
    const c = try parse(std.testing.allocator, text);
    try std.testing.expectEqual(@as(usize, 576), c.latentWidth());
    try std.testing.expect(c.fullIndexer(2) and !c.fullIndexer(3));
    try std.testing.expect(!c.isMoe(2) and c.isMoe(3));
    try std.testing.expectEqual(@as(f64, 8000000), c.rope_theta);
    try std.testing.expectError(error.NotGlmMoeDsa, parse(std.testing.allocator, "{\"model_type\": \"glm5_next\"}"));
}
