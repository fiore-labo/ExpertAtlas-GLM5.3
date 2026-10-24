// Fiore-lab: rank 0's memory for the control panel (GET /v1/memory), not in TensorFold 1.0.2.
//! What this rank's start-up allocated step by step (the DeviceBuffers memMark saw), the routed experts' share of the
//! weights, and a snapshot the engine thread takes between requests (the process's tracked bytes and the driver's
//! free / total). /proc/meminfo is read when asked (no GPU). The driver's used bytes minus the tracked ones mix CUDA's
//! own allocations, graphs, other processes and GB10's page cache, so they are shown as they are, never as one thing.
//! 制御パネル用の rank 0 のメモリ (GET /v1/memory)。このランクの起動が段階ごとに確保したもの (memMark が見た
//! DeviceBuffer)、重みのうち routed expert の分、要求の合間にエンジンのスレッドが取る写し (プロセスが数えている量と
//! ドライバの free / total)。/proc/meminfo は問われたときに読む (GPU に触れない)。ドライバの使用量から数えている量を
//! 引いた差には CUDA 自身の確保、グラフ、他のプロセス、GB10 のページキャッシュが混ざるので、そのまま出し、1 つの
//! ものとしては扱わない。
const std = @import("std");

pub const Step = struct { label: []const u8, bytes: u64 };

pub const MemLive = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    rank: u32,
    steps: []Step,
    experts: u64,
    /// The snapshot (engine thread, between requests): tracked device / pinned host bytes, the device peak, the
    /// driver's free and total, and when (ms since start).
    /// 写し (エンジンのスレッド、要求の合間): 数えている device / pinned host のバイト、device のピーク、ドライバの
    /// free と total、その時刻 (起動からの ms)。
    tracked_device: u64 = 0,
    tracked_host: u64 = 0,
    peak: u64 = 0,
    free: u64 = 0,
    total: u64 = 0,
    at_ms: u64 = 0,
    snaps: u64 = 0,
    t0: std.Io.Timestamp,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, rank: u32, steps: []const Step, experts: u64) !*MemLive {
        const m = try gpa.create(MemLive);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .io = io, .rank = rank, .steps = try gpa.dupe(Step, steps), .experts = experts, .t0 = std.Io.Clock.awake.now(io) };
        return m;
    }

    /// The routed experts' bytes after a change of their bits (expert_swap).
    /// expert の bit を変えた後の routed expert のバイト (expert_swap)。
    pub fn setExperts(m: *MemLive, bytes: u64) void {
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        m.experts = bytes;
    }

    pub fn snap(m: *MemLive, device: u64, host: u64, peak: u64, free: u64, total: u64) void {
        const ms = m.t0.durationTo(std.Io.Clock.awake.now(m.io)).toMilliseconds();
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        m.tracked_device = device;
        m.tracked_host = host;
        m.peak = peak;
        m.free = free;
        m.total = total;
        m.at_ms = if (ms < 0) 0 else @intCast(ms);
        m.snaps += 1;
    }

    /// /proc/meminfo's fields in bytes (0 when missing).
    /// /proc/meminfo の項目 (バイト。無ければ 0)。
    pub fn meminfo(io: std.Io, names: []const []const u8, out: []u64) void {
        @memset(out, 0);
        var buf: [8192]u8 = undefined;
        const text = std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return;
        for (names, out) |name, *v| {
            const at = std.mem.indexOf(u8, text, name) orelse continue;
            var it = std.mem.tokenizeAny(u8, text[at + name.len ..], " \n");
            v.* = (std.fmt.parseInt(u64, it.next() orelse continue, 10) catch continue) * 1024;
        }
    }

    pub fn json(m: *MemLive, a: std.mem.Allocator) ![]u8 {
        const names = [_][]const u8{ "MemTotal:", "MemAvailable:", "MemFree:", "Cached:" };
        var mi: [names.len]u64 = undefined;
        meminfo(m.io, &names, &mi);
        var s: MemLive = undefined;
        {
            m.mutex.lockUncancelable(m.io);
            defer m.mutex.unlock(m.io);
            s = m.*;
        }
        const now = m.t0.durationTo(std.Io.Clock.awake.now(m.io)).toMilliseconds();
        var out: std.ArrayList(u8) = .empty;
        try out.print(a, "{{\"rank\":{d},\"tracked\":{{\"device\":{d},\"host\":{d},\"peak\":{d}}}," ++
            "\"driver\":{{\"free\":{d},\"total\":{d},\"age_ms\":{d},\"snapshots\":{d}}}," ++
            "\"meminfo\":{{\"total\":{d},\"available\":{d},\"free\":{d},\"cached\":{d}}},\"experts\":{d},\"startup\":[", .{
            s.rank, s.tracked_device, s.tracked_host, s.peak, s.free, s.total, if (now > s.at_ms) @as(u64, @intCast(now)) - s.at_ms else 0, s.snaps,
            mi[0], mi[1], mi[2], mi[3], s.experts,
        });
        for (s.steps, 0..) |st, i| try out.print(a, "{s}{{\"label\":\"{s}\",\"bytes\":{d}}}", .{ if (i == 0) "" else ",", st.label, st.bytes });
        try out.appendSlice(a, "]}");
        return out.toOwnedSlice(a);
    }
};
