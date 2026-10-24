// Fiore-lab: routed experts loaded at start from banks by a plan (TF_GLM53_EXPERT_PLAN), not in TensorFold 1.0.2.
//! An expert plan (glm53-expert-plan v1): for every routed expert of the MoE layers
//! (layers 3..77 of GLM-5.3; the MTP layer is not in a plan), the bank it is loaded from. A bank is a directory of
//! experts-layer-NNN.safetensors files at one width, identified by its manifest (glm53-expert-bank v1: every layer
//! file's size, header and sha256). Before anything heavy is allocated every bank's manifest is checked against the
//! plan's sha256, and the header and size of every layer file the plan reads against the manifest; the plan digest
//! (sha256 of the plan file) goes into the start fingerprint, so ranks given different plans do not serve.
//! エキスパートの計画: MoE 層 (GLM-5.3 の第 3..77 層。MTP 層は計画に入らない) の routed expert ごとに、読み込むバンク。
//! バンクは 1 つの幅の experts-layer-NNN.safetensors のディレクトリで、manifest (glm53-expert-bank v1) で同定する。
//! 重い確保の前に、各バンクの manifest を計画の sha256 と、計画が読む層ファイルの header と大きさを manifest と照らす。
//! 計画のダイジェスト (計画ファイルの sha256) は起動時の指紋に入り、違う計画を渡されたランクは serve しない。
const std = @import("std");
const core = @import("core");
const Config = @import("config.zig").Config;

const dio = core.direct_io;

/// Banks a plan may name.
pub const max_banks = 16;

pub const LayerFile = struct {
    layer: usize,
    file: []const u8,
    size: u64,
    header_bytes: u64,
    header_sha256: [32]u8,
};

pub const Bank = struct {
    id: []const u8,
    bits: u8,
    dir: []const u8,
    manifest_sha256: [32]u8,
    calibration: []const u8,
    files: []LayerFile, // by layer - first_layer
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    name: []const u8,
    first_layer: usize,
    layers: usize,
    experts: usize,
    banks: []Bank,
    bank_of: []u8, // layer-major from first_layer: an index into banks
    digest: [32]u8,

    /// The bank index of every expert of `layer`, or null for a layer the plan does not cover (dense, MTP).
    /// `layer` の各 expert のバンク番号。計画の外の層 (dense、MTP) は null。
    pub fn row(p: *const Plan, layer: usize) ?[]const u8 {
        if (layer < p.first_layer or layer >= p.first_layer + p.layers) return null;
        return p.bank_of[(layer - p.first_layer) * p.experts ..][0..p.experts];
    }

    /// The banks `layer` reads (a mask over banks).
    pub fn uses(p: *const Plan, layer: usize, bank: usize) bool {
        const r = p.row(layer) orelse return false;
        return std.mem.indexOfScalar(u8, r, @intCast(bank)) != null;
    }

    pub fn deinit(p: *Plan) void {
        p.arena.deinit();
        p.* = undefined;
    }
};

const PlanJson = struct {
    format: []const u8,
    version: u32,
    name: []const u8 = "",
    first_layer: usize,
    layers: usize,
    experts: usize,
    banks: []const struct { id: []const u8, bits: u8, manifest_sha256: []const u8, dir: []const u8, manifest: []const u8 = "" },
    bank_of: []const u8,
};

const ManifestJson = struct {
    format: []const u8,
    version: u32,
    id: []const u8,
    bits: u8,
    calibration: []const u8 = "",
    first_layer: usize,
    layer_count: usize,
    layers: []const struct { layer: usize, file: []const u8, size: u64, header_bytes: u64, header_sha256: []const u8 },
};

/// TF_GLM53_EXPERT_PLAN=<plan.json>: the plan, its banks' manifests (a bank's `manifest` in the plan, else <id>.json in
/// TF_GLM53_BANK_MANIFESTS=<dir>, else in the bank's directory, where a downloaded bank has it) and the bank directories
/// (the plan's, or TF_GLM53_PLAN_BANK_DIRS=id:dir,...), checked against `cfg`; null without the variable.
/// TF_GLM53_EXPERT_PLAN=<plan.json>: 計画と、そのバンクの manifest (計画の `manifest`、無ければ TF_GLM53_BANK_MANIFESTS の
/// <id>.json、それも無ければバンクのディレクトリの <id>.json。ダウンロードしたバンクはそこに持つ) とディレクトリを読み、
/// `cfg` と照らす。変数が無ければ null。
pub fn fromEnv(gpa: std.mem.Allocator, io: std.Io, cfg: Config) !?Plan {
    const path = std.mem.span(std.c.getenv("TF_GLM53_EXPERT_PLAN") orelse return null);
    const mdir: ?[]const u8 = if (std.c.getenv("TF_GLM53_BANK_MANIFESTS")) |v| std.mem.span(v) else null;
    const dirs = if (std.c.getenv("TF_GLM53_PLAN_BANK_DIRS")) |v| std.mem.span(v) else "";
    return try load(gpa, io, cfg, path, mdir, dirs);
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, cfg: Config, path: []const u8, mdir: ?[]const u8, dirs: []const u8) !Plan {
    var p: Plan = .{ .arena = .init(gpa), .name = "", .first_layer = 0, .layers = 0, .experts = 0, .banks = &.{}, .bank_of = &.{}, .digest = undefined };
    errdefer p.arena.deinit();
    const a = p.arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    std.crypto.hash.sha2.Sha256.hash(text, &p.digest, .{});
    const parsed = try std.json.parseFromSlice(PlanJson, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const v = parsed.value;
    if (!std.mem.eql(u8, v.format, "glm53-expert-plan") or v.version != 1) return error.ExpertPlanFormat;
    // the plan covers exactly the routed-expert layers below the MTP layer
    // 計画はちょうど MTP 層より下の routed expert の層を覆う
    var first: ?usize = null;
    var moe_layers: usize = 0;
    for (0..cfg.layers) |i| if (cfg.isMoe(i)) {
        if (first == null) first = i;
        moe_layers += 1;
    };
    if (first == null or v.first_layer != first.? or v.layers != moe_layers or v.experts != cfg.experts) {
        std.log.err("glm53: plan {s}: layers {d}+{d} x {d} experts, the model has {?d}+{d} x {d}", .{ path, v.first_layer, v.layers, v.experts, first, moe_layers, cfg.experts });
        return error.ExpertPlanShape;
    }
    if (v.banks.len == 0 or v.banks.len > max_banks) return error.ExpertPlanBanks;
    p.name = try a.dupe(u8, v.name);
    p.first_layer = v.first_layer;
    p.layers = v.layers;
    p.experts = v.experts;
    const dec = std.base64.standard.Decoder;
    p.bank_of = try a.alloc(u8, try dec.calcSizeForSlice(v.bank_of));
    try dec.decode(p.bank_of, v.bank_of);
    if (p.bank_of.len != p.layers * p.experts) return error.ExpertPlanShape;
    for (p.bank_of) |b| if (b >= v.banks.len) return error.ExpertPlanBankIndex;
    p.banks = try a.alloc(Bank, v.banks.len);
    for (v.banks, p.banks) |bj, *b| {
        b.* = .{ .id = try a.dupe(u8, bj.id), .bits = bj.bits, .dir = try a.dupe(u8, overrideDir(dirs, bj.id) orelse bj.dir), .manifest_sha256 = undefined, .calibration = "", .files = &.{} };
        if (bj.manifest_sha256.len != 64) return error.ExpertPlanManifestDigest;
        _ = try std.fmt.hexToBytes(&b.manifest_sha256, bj.manifest_sha256);
        try readManifest(gpa, io, a, b, bj.manifest, mdir orelse b.dir, p.first_layer, p.layers);
    }
    return p;
}

fn overrideDir(dirs: []const u8, id: []const u8) ?[]const u8 {
    var it = std.mem.tokenizeScalar(u8, dirs, ',');
    while (it.next()) |item| {
        const c = std.mem.indexOfScalar(u8, item, ':') orelse continue;
        if (std.mem.eql(u8, item[0..c], id)) return item[c + 1 ..];
    }
    return null;
}

/// The bank's manifest, checked against the plan's sha256, its id and bits, and the plan's layers.
fn readManifest(gpa: std.mem.Allocator, io: std.Io, a: std.mem.Allocator, b: *Bank, given: []const u8, mdir: []const u8, first: usize, n: usize) !void {
    const name = try std.fmt.allocPrint(gpa, "{s}.json", .{b.id});
    defer gpa.free(name);
    const mpath = if (given.len > 0) try gpa.dupe(u8, given) else try std.fs.path.join(gpa, &.{ mdir, name });
    defer gpa.free(mpath);
    const text = std.Io.Dir.cwd().readFileAlloc(io, mpath, gpa, .limited(256 << 20)) catch |err| {
        std.log.err("glm53: bank {s}: manifest {s}: {s}", .{ b.id, mpath, @errorName(err) });
        return error.BankManifestMissing;
    };
    defer gpa.free(text);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
    if (!std.mem.eql(u8, &d, &b.manifest_sha256)) {
        std.log.err("glm53: bank {s}: manifest {s} is not the one the plan names (sha256 {x})", .{ b.id, mpath, d });
        return error.BankManifestDiffers;
    }
    const parsed = try std.json.parseFromSlice(ManifestJson, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    if (!std.mem.eql(u8, m.format, "glm53-expert-bank") or m.version != 1) return error.BankManifestFormat;
    if (!std.mem.eql(u8, m.id, b.id) or m.bits != b.bits) return error.BankManifestIdentity;
    if (m.first_layer != first or m.layer_count != n or m.layers.len != n) return error.BankManifestLayers;
    b.calibration = try a.dupe(u8, m.calibration);
    b.files = try a.alloc(LayerFile, n);
    for (m.layers, b.files, 0..) |x, *f, i| {
        if (x.layer != first + i or x.header_sha256.len != 64) return error.BankManifestLayers;
        f.* = .{ .layer = x.layer, .file = try a.dupe(u8, x.file), .size = x.size, .header_bytes = x.header_bytes, .header_sha256 = undefined };
        _ = try std.fmt.hexToBytes(&f.header_sha256, x.header_sha256);
    }
}

/// Every layer file the plan reads: its size and header bytes equal the manifest's (direct reads, no page cache).
/// Cheap (headers only, ~2.8 MiB a file); run before the weights are loaded.
/// 計画が読む層ファイルすべてで、大きさと header のバイトが manifest と同じこと (direct read、ページキャッシュなし)。
pub fn checkFiles(p: *const Plan, gpa: std.mem.Allocator) !void {
    const buf: dio.Buffer = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), 8 << 20);
    defer gpa.free(buf);
    for (p.banks, 0..) |*b, bi| for (b.files) |*f| {
        if (!p.uses(f.layer, bi)) continue;
        const path = try std.fs.path.joinZ(gpa, &.{ b.dir, f.file });
        defer gpa.free(path);
        var file = dio.File.open(path) catch {
            std.log.err("glm53: bank {s}: {s} missing", .{ b.id, path });
            return error.BankFileMissing;
        };
        defer file.close();
        const end = std.c.lseek(file.fd, 0, std.c.SEEK.END);
        if (end < 0 or @as(u64, @intCast(end)) != f.size) {
            std.log.err("glm53: bank {s}: {s} is {d} bytes, the manifest says {d}", .{ b.id, path, end, f.size });
            return error.BankFileDiffers;
        }
        const head = try file.read(buf, 0, 8);
        const n = std.mem.readInt(u64, head[0..8], .little);
        if (n != f.header_bytes or n + 8 > dio.fits(buf.len)) return error.BankFileDiffers;
        const raw = try file.read(buf, 8, @intCast(n));
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(raw, &d, .{});
        if (!std.mem.eql(u8, &d, &f.header_sha256)) {
            std.log.err("glm53: bank {s}: {s}: the header differs from the manifest", .{ b.id, path });
            return error.BankFileDiffers;
        }
    };
}

/// Counts of the plan's experts by bits (index = bits).
pub fn counts(p: *const Plan) [9]u64 {
    var c: [9]u64 = @splat(0);
    for (p.bank_of) |bi| c[@min(p.banks[bi].bits, 8)] += 1;
    return c;
}
