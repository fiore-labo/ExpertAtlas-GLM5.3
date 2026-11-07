//! TF_GLM53_MULTI: a drafted conversation's window in a round. Alone it keeps its full window (DFlash2's verify rows,
//! MTP's 1 + k); beside another it takes TF_GLM53_MULTI_DRAFT_ROWS, 4 by default with DFlash2: the window is padded to
//! that width whatever the drafts, and on prose four conversations at 4 rows ran 12 % faster than at 8, code 2 % slower
//! (measured on eight nodes, prose; code on four nodes at 130K tokens). Every rank picks the same: the round's
//! conversations are rank 0's event and the start's settings are checked alike on every rank.
//! TF_GLM53_MULTI: ラウンドでの、draft する会話のウィンドウ。1 会話だけならウィンドウいっぱい (DFlash2 の verify の行、
//! MTP の 1 + k)。ほかの会話と並ぶときは TF_GLM53_MULTI_DRAFT_ROWS で、DFlash2 の既定は 4: ウィンドウは draft の数に
//! よらずこの幅まで詰め物される。文章の 4 会話では 4 行が 8 行より 12 % 速く、コードでは 2 % 遅かった
//! (文章は 8 台、コードは 4 台・130K トークンで測った)。全ランクが同じ値を選ぶ: ラウンドの会話は rank 0 の
//! 出来事から来て、起動の設定は全ランクで同じことを確かめている。

const std = @import("std");

/// DFlash2's window beside other conversations when TF_GLM53_MULTI_DRAFT_ROWS is not set.
/// TF_GLM53_MULTI_DRAFT_ROWS がないときの、ほかの会話と並ぶ DFlash2 のウィンドウ。
pub const dflash_beside = 4;

/// The window beside other conversations by default: DFlash2's 4 rows (its rows if fewer), else the drafter's rows.
/// ほかの会話と並ぶときのウィンドウの既定: DFlash2 は 4 行 (rows がそれより少なければ rows)、ほかは drafter の rows。
pub fn besideDefault(dflash: bool, rows: usize) usize {
    return if (dflash) @min(rows, dflash_beside) else rows;
}

/// A drafted conversation's window in a round of `live` conversations: `rows` alone, `beside` with others.
/// `live` 会話のラウンドでの、draft する会話のウィンドウ: 1 会話だけなら `rows`、ほかと並ぶなら `beside`。
pub fn width(live: usize, rows: usize, beside: usize) usize {
    return if (live >= 2) beside else rows;
}

test "multi_width: DFlash2 takes 4 rows beside others, MTP and no drafter keep their rows" {
    const cases = [_]struct { bool, usize, usize }{
        .{ true, 8, 4 }, .{ true, 4, 4 }, .{ true, 3, 3 }, .{ true, 2, 2 }, .{ true, 1, 1 },
        .{ false, 4, 4 }, .{ false, 8, 8 }, .{ false, 1, 1 },
    };
    for (cases) |c| try std.testing.expectEqual(c[2], besideDefault(c[0], c[1]));
}

test "multi_width: alone the full window, from two conversations the width beside" {
    const cases = [_]struct { usize, usize, usize, usize }{ // live, rows, beside, width
        .{ 1, 8, 4, 8 }, .{ 2, 8, 4, 4 }, .{ 3, 8, 4, 4 }, .{ 4, 8, 4, 4 },
        // TF_GLM53_MULTI_DRAFT_ROWS set: 2 or the old 8 rows; MTP (1 + 3 rows) at 2 from two conversations on
        .{ 2, 8, 2, 2 }, .{ 2, 8, 8, 8 }, .{ 2, 4, 2, 2 }, .{ 1, 4, 2, 4 },
        .{ 2, 1, 1, 1 }, // no drafter: one row
    };
    for (cases) |c| try std.testing.expectEqual(c[3], width(c[0], c[1], c[2]));
}
