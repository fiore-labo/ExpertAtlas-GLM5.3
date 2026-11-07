//! Fiore-lab: llguidance (v1.9.1, MIT) loaded at run time for the structured output: its C API's structs, the table of
//! the functions rank 0 calls, the version check and the vocabulary built from the server's tokenizer. Host code only
//! (no cuda), so it runs in the Mac's unit tests. llguidance is the grammar engine since it refuses the keywords it
//! does not implement and intersects those of one node, where xgrammar v0.2.8 weakened such schemas without an error.
//! Fiore-lab: 構造化出力のために実行時に読む llguidance (v1.9.1、MIT): C API の構造体、rank 0 が呼ぶ関数の表、版の確かめ、
//! サーバーの tokenizer から作る語彙。host だけのコード (cuda なし) なので Mac の単体試験で動く。

const std = @import("std");

pub const Token = u32;

/// The version the structs below copy; another version may lay them out differently, so loading it fails.
/// 下の構造体が写した版。版が違うと形が違いうるので、読み込みを失敗にする。
pub const version_prefix = "llguidance@1.9.1 ";

/// LlgParserLimits (llguidance.h:82-133).
pub const ParserLimits = extern struct {
    max_items_in_row: usize,
    initial_lexer_fuel: u64,
    step_lexer_fuel: u64,
    step_max_items: usize,
    max_lexer_states: usize,
    max_grammar_size: usize,
    precompute_large_lexemes: bool,
    verbose_errors: bool,
};

pub const TokenizeFn = *const fn (user_data: ?*const anyopaque, bytes: [*]const u8, bytes_len: usize, out: [*]Token, out_len: usize) callconv(.c) usize;

/// LlgTokenizerInitV2 (llguidance.h:338-407).
pub const TokenizerInitV2 = extern struct {
    struct_size: usize,
    vocab_size: u32,
    tok_eos: Token,
    token_lens: ?[*]const u32,
    token_bytes: ?[*]const u8,
    tokenizer_json: ?[*:0]const u8,
    tokenize_assumes_string: bool,
    tokenize_fn: ?TokenizeFn,
    use_approximate_greedy_tokenize_fn: bool,
    tokenize_user_data: ?*const anyopaque,
    slices: ?[*]const ?[*:0]const u8,
    tok_eos_extra: ?[*]const Token,
    tok_eos_extra_count: u32,
};

pub const Tokenizer = opaque {};
pub const Matcher = opaque {};

/// LlgConstraintInit (llguidance.h:146-176).
pub const ConstraintInit = extern struct {
    tokenizer: ?*const Tokenizer,
    log_buffer_level: u32,
    log_stderr_level: u32,
    ff_tokens_ok: bool,
    backtrack_ok: bool,
    limits: ParserLimits,
};

/// The functions rank 0 calls, looked up by name.
/// rank 0 が呼ぶ関数。名前で引く。
pub const Api = struct {
    llg_get_version: *const fn () callconv(.c) [*:0]const u8,
    llg_new_tokenizer_v2: *const fn (init: *const TokenizerInitV2, err: [*]u8, err_len: usize) callconv(.c) ?*Tokenizer,
    llg_free_tokenizer: *const fn (t: *Tokenizer) callconv(.c) void,
    llg_constraint_init_set_defaults: *const fn (init: *ConstraintInit, t: ?*const Tokenizer) callconv(.c) void,
    llg_new_matcher: *const fn (init: *const ConstraintInit, kind: [*:0]const u8, data: [*:0]const u8) callconv(.c) ?*Matcher,
    llg_matcher_get_error: *const fn (m: *Matcher) callconv(.c) ?[*:0]const u8,
    llg_matcher_is_error: *const fn (m: *const Matcher) callconv(.c) bool,
    llg_matcher_get_mask_byte_size: *const fn (m: *const Matcher) callconv(.c) usize,
    llg_matcher_compute_mask_into: *const fn (m: *Matcher, dest: [*]u32, byte_len: usize) callconv(.c) i32,
    llg_matcher_consume_token: *const fn (m: *Matcher, token: Token) callconv(.c) i32,
    llg_matcher_consume_tokens: *const fn (m: *Matcher, tokens: [*]const Token, n: usize) callconv(.c) i32,
    llg_matcher_is_accepting: *const fn (m: *Matcher) callconv(.c) bool,
    llg_matcher_is_stopped: *const fn (m: *const Matcher) callconv(.c) bool,
    llg_matcher_validate_tokens: *const fn (m: *Matcher, tokens: [*]const Token, n: usize) callconv(.c) i32,
    llg_matcher_rollback: *const fn (m: *Matcher, n: usize) callconv(.c) i32,
    llg_clone_matcher: *const fn (m: *const Matcher) callconv(.c) ?*Matcher,
    llg_free_matcher: *const fn (m: *Matcher) callconv(.c) void,
};

pub const Error = error{ LibraryUnavailable, MissingSymbol, WrongVersion };

pub const Library = struct {
    lib: std.DynLib,
    api: Api,

    /// Opens `path` (TF_GRAMMAR_LIB) and checks the version before anything else is called.
    /// `path` (TF_GRAMMAR_LIB) を開き、ほかを呼ぶ前に版を確かめる。
    pub fn openPath(path: []const u8) Error!Library {
        var lib = std.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        const v = std.mem.span(api.llg_get_version());
        if (!std.mem.startsWith(u8, v, version_prefix)) {
            std.log.err("{s} is {s}; the structured output needs {s}", .{ path, v, version_prefix });
            return error.WrongVersion;
        }
        return .{ .lib = lib, .api = api };
    }

    pub fn close(l: *Library) void {
        l.lib.close();
    }
};

/// The server tokenizer's view of a token (engine_api.TokenVocab's shape): its raw bytes (null: none) and whether it
/// is an added token.
/// サーバーの tokenizer から見た token (engine_api.TokenVocab と同じ形): 生のバイト (null はなし) と added かどうか。
pub const VocabSource = struct {
    ctx: *anyopaque,
    bytes: *const fn (ctx: *anyopaque, a: std.mem.Allocator, id: u32) ?[]const u8,
    added: *const fn (ctx: *anyopaque, id: u32) bool,
};

/// Every id's length and bytes as llguidance takes them: an added token is 0xFF and its name (llguidance's way of
/// writing a special token, never reachable from a JSON or regex grammar since 0xFF is no UTF-8 byte), an id without
/// bytes (the model's padding) is empty and so never allowed.
/// llguidance が受け取る形の、全 id の長さとバイト: added は 0xFF + 名前 (llguidance の special token の書き方。0xFF は UTF-8
/// に出ないので JSON と regex の文法からは届かない)、バイトのない id (モデルの詰め物) は空で、決して許されない。
pub const Vocab = struct {
    lens: []u32,
    bytes: []u8,

    pub fn build(a: std.mem.Allocator, src: VocabSource, vocab_size: u32) !Vocab {
        const lens = try a.alloc(u32, vocab_size);
        errdefer a.free(lens);
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        var scratch: std.heap.ArenaAllocator = .init(a);
        defer scratch.deinit();
        for (0..vocab_size) |i| {
            const id: u32 = @intCast(i);
            const b = src.bytes(src.ctx, scratch.allocator(), id) orelse &.{};
            const start = out.items.len;
            if (b.len > 0 and src.added(src.ctx, id)) try out.append(a, 0xFF);
            try out.appendSlice(a, b);
            lens[i] = @intCast(out.items.len - start);
        }
        return .{ .lens = lens, .bytes = try out.toOwnedSlice(a) };
    }

    pub fn deinit(v: *Vocab, a: std.mem.Allocator) void {
        a.free(v.lens);
        a.free(v.bytes);
    }
};

/// A tokenizer for the vocabulary with the model's end tokens (the first as tok_eos, the rest as extras) and the
/// approximate tokenize function (not the exact BPE, so llguidance forces no token: a span's BPE alone can split it
/// otherwise than the model does in context). llguidance copies the arrays: `v` may be freed after. On failure the
/// message is in `err`.
/// モデルの終わりの token (最初を tok_eos、残りを extra) と近似の tokenize の関数 (本物の BPE は渡さないので llguidance は
/// token を強制しない: 区間だけの BPE の切り方は、文脈の中でモデルが出す切り方と違いうる) で語彙の tokenizer を作る。
/// llguidance は列を写すので `v` は後で放してよい。失敗の文は `err`。
pub fn newTokenizer(api: *const Api, v: *const Vocab, eos: []const Token, err: []u8) ?*Tokenizer {
    std.debug.assert(eos.len >= 1 and err.len > 0);
    const init: TokenizerInitV2 = .{
        .struct_size = @sizeOf(TokenizerInitV2),
        .vocab_size = @intCast(v.lens.len),
        .tok_eos = eos[0],
        .token_lens = v.lens.ptr,
        .token_bytes = v.bytes.ptr,
        .tokenizer_json = null,
        .tokenize_assumes_string = false,
        .tokenize_fn = null,
        .use_approximate_greedy_tokenize_fn = true,
        .tokenize_user_data = null,
        .slices = null,
        .tok_eos_extra = if (eos.len > 1) eos[1..].ptr else null,
        .tok_eos_extra_count = @intCast(eos.len - 1),
    };
    @memset(err, 0);
    return api.llg_new_tokenizer_v2(&init, err.ptr, err.len);
}

/// The constraint settings put on top of llguidance's defaults: nothing to rank 0's stderr, errors
/// without the parser's state, tokens or grammar in them.
/// llguidance の既定に重ねる設定: rank 0 の stderr に何も出さない、エラーの文に状態・token・文法を入れない。
pub fn constraintInit(api: *const Api, t: *const Tokenizer) ConstraintInit {
    var init: ConstraintInit = undefined;
    api.llg_constraint_init_set_defaults(&init, t);
    init.log_buffer_level = 0;
    init.log_stderr_level = 0;
    init.limits.verbose_errors = false;
    return init;
}

/// The words of one mask row for `vocab_size` ids (llguidance's mask is 32-bit words, bit id % 32 of word id / 32).
/// `vocab_size` 個の id の mask の 1 行の語数 (llguidance の mask は 32 bit の語で、id は語 id / 32 の bit id % 32)。
pub fn maskWords(vocab_size: u32) usize {
    return (vocab_size + 31) / 32;
}

pub fn allowed(row: []const u32, id: Token) bool {
    return (row[id / 32] >> @intCast(id % 32)) & 1 != 0;
}
