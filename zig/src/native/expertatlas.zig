// Fiore-lab, Expert Atlas.
//! The engines Expert Atlas's server opens: GLM-5.3 (glm_moe_dsa, EXL3) on CUDA, one process a GPU, the ranks of one
//! tensor-parallel replica (glm53_host.zig). The server asks for `backends`, `families`, `chip` and `open`; nothing
//! else is registered (TensorFold's native/cuda.zig also registered Nemotron behind its lane host).
//! Expert Atlas のサーバーが開くエンジン: CUDA の GLM-5.3 だけ。サーバーが使うのは backends・families・chip・open。
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const glm53 = @import("glm53_host.zig");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"cuda"};
pub const families: []const api.Family = &.{.{ .model_type = "glm_moe_dsa", .formats = &.{"exl3"} }};

/// Positions a request may use when the start names no --context.
const default_window: i64 = 8192;

/// The gate class of the GPU the engine would use (TF_CUDA_DEVICE, else 0): "nvidia-sm121" on a GB10; null without one.
pub fn chip(a: Allocator) ?[]const u8 {
    const device = deviceNumber() catch return null;
    var driver = cuda.Driver.open() catch return null;
    defer driver.close();
    const count = driver.deviceCount() catch return null;
    if (device >= count) return null;
    const capability = driver.capability(@intCast(device)) catch return null;
    return std.fmt.allocPrint(a, "nvidia-sm{d}", .{capability}) catch null;
}

/// TF_CUDA_DEVICE as a whole number; 0 when it is unset or blank.
fn deviceNumber() error{Invalid}!u32 {
    const raw = std.c.getenv("TF_CUDA_DEVICE") orelse return 0;
    const text = std.mem.trim(u8, std.mem.span(raw), " ");
    if (text.len == 0) return 0;
    return std.fmt.parseInt(u32, text, 10) catch error.Invalid;
}

/// The model's own window: config.json's max_position_embeddings (text_config's first); 0 when it names none.
fn modelWindow(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    return windowIn(a, bytes);
}

/// config.json's text as modelWindow reads it.
fn windowIn(a: Allocator, bytes: []const u8) i64 {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const v = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    if (v != .integer or v.integer <= 0) return 0;
    return v.integer;
}

/// The engine for the checkpoint in `o.dir`, or null with `problem` set.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (!std.mem.eql(u8, o.model_type, "glm_moe_dsa")) {
        problem.* = try std.fmt.allocPrint(a, "Expert Atlas serves GLM-5.3 (glm_moe_dsa) checkpoints; {s} is not one", .{o.model_type});
        return null;
    }
    const model = modelWindow(a, io, o.dir);
    const window: i64 = o.context orelse default_window;
    if (window <= 1 or (model > 0 and window > model)) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: GLM-5.3 takes 2 .. {d} positions", .{ window, model });
        return null;
    }
    const h = glm53.open(gpa, io, o.dir, @intCast(window)) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native GLM-5.3 engine cannot load {s} ({s}; set TF_GLM53_RANK / _WORLD / _MASTER / _AOT for each rank)", .{ o.dir, @errorName(e) });
        return null;
    };
    return .{ .engine = h.engine(), .close = glm53.close, .ctx = h };
}

test {
    _ = glm53; // its own tests (the think block, TF_GLM53_IDLE_PARK_MS, structured output)
}

test "one family, GLM-5.3 in EXL3, on CUDA" {
    try std.testing.expectEqual(@as(usize, 1), families.len);
    try std.testing.expectEqualStrings("glm_moe_dsa", families[0].model_type);
    try std.testing.expectEqualStrings("exl3", families[0].formats[0]);
    try std.testing.expectEqualStrings("cuda", backends[0]);
}

test "the model's window: max_position_embeddings, text_config's first, 0 when absent or not positive" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { json: []const u8, want: i64 }{
        .{ .json = "{\"max_position_embeddings\": 1048576}", .want = 1048576 },
        .{ .json = "{\"text_config\": {\"max_position_embeddings\": 4096}, \"max_position_embeddings\": 8}", .want = 4096 },
        .{ .json = "{\"text_config\": {}, \"max_position_embeddings\": 8}", .want = 8 },
        .{ .json = "{\"text_config\": 3, \"max_position_embeddings\": 9}", .want = 9 },
        .{ .json = "{\"max_position_embeddings\": 0}", .want = 0 },
        .{ .json = "{}", .want = 0 },
        .{ .json = "not json", .want = 0 },
    };
    for (cases) |c| try std.testing.expectEqual(c.want, windowIn(a, c.json));
}
