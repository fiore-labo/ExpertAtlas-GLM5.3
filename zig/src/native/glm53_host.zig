//! GLM-5.3 behind the native server: one process a GPU, the ranks of one tensor-parallel replica. Rank 0 serves HTTP
//! and runs one request at a time (foreground first) on every rank. A request samples or decodes greedily, may resume
//! from the slot cache and draft (MTP, DFlash2); a cancel or stop string stops every rank at the next agreement point.
//! ネイティブサーバーで動く GLM-5.3: GPU ごとに 1 プロセス、1 つの tensor-parallel レプリカのランク群。rank 0 が
//! HTTP を受け、要求を 1 つずつ (foreground を先に) 全ランクで実行する。要求はサンプルか greedy で decode し、
//! スロットキャッシュからの再開と draft (MTP、DFlash2) もできる。cancel や stop 文字列は次の合意点で全ランクを止める。
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const glm = @import("glm_moe_dsa");
const Allocator = std.mem.Allocator;

/// What rank 0 stopped a decode for in the think block (Job.thinkCut): the text to write into the reply, and a fold
/// of the context the model goes on from.
/// rank 0 が思考ブロックで decode を止めた理由 (Job.thinkCut): 応答に書き込む文と、モデルが続ける文脈の畳み方。
const Cut = struct {
    text: []const u32 = &.{},
    fold: Fold = .none,

    // canon: no text and no fold, the think block closed with ids remade before the answer (Host.closeCut);
    // branch_open: the summary beside the reply asked for (think_fold branch), branch_close: it ended, the block folds
    // canon: 文も畳み込みもない。閉じた思考ブロックの id を作り直してから答える (Host.closeCut)。branch_open: 応答の横での
    // 要約を求める (think_fold branch)、branch_close: それが終わり、ブロックを畳む
    const Fold = enum { none, recent, checkpoint, canon, branch_open, branch_close };
};

const Job = struct {
    id: api.Id,
    request: *const api.Request,
    sink: api.Sink,
    emitted: std.ArrayList(u32) = .empty,
    // a stop string matched or the client left: the rest of the reply is dropped
    // stop 文字列に一致したか、クライアントが去った: 応答の残りは捨てる
    quiet: bool = false,
    queued: i96 = 0,
    began: i96 = 0,
    prefilled: ?i96 = null,
    // its progress for /metrics (the engine thread writes it under the host's mutex): when the prefill started, the
    // prompt rows the cache resumed, those in the caches so far and when they were (the last chunk's end: the prefill
    // seconds move with the rows, so tokens over seconds is the chunks' own rate whenever a monitor reads it), the
    // decode rounds and drafts
    // /metrics 用の進み具合 (エンジンのスレッドがホストの mutex の下で書く): prefill を始めた時刻、キャッシュから
    // 再開した prompt の行数、いまキャッシュにある行数とその時刻 (最後のチャンクの終わり: prefill の秒数は行と一緒に
    // 進むので、モニタがいつ読んでもトークン ÷ 秒はチャンクそのものの速さになる)、decode のラウンド数と draft の数
    prefill_at: ?i96 = null,
    cached: u64 = 0,
    done: u64 = 0,
    done_at: i96 = 0,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    // TF_GLM53_MULTI: decoded as one of the conversations of the batched rounds; `stop`: rank 0's stop for it (the
    // client left, a stop string matched, a cancel), sent with the next round
    // TF_GLM53_MULTI: まとめたラウンドの会話の 1 つとして decode する。`stop`: それへの rank 0 の停止 (クライアントが
    // 去った、stop 文字列に一致、cancel)。次のラウンドで送る
    multi: bool = false,
    stop: std.atomic.Value(bool) = .init(false),
    // TF_GLM53_MULTI: when the pool first had no room for it beside the conversations in flight (its wait's start)
    // TF_GLM53_MULTI: 処理中の会話の横に pool の空きが初めて無かった時刻 (待ちの始まり)
    room_since: ?i96 = null,
    // TF_GLM53_MULTI: the rows it would compute past the cache (rank 0's scheduler's estimate, counted at pool_gen
    // `fresh_gen`), and whether it was admitted as a short prompt (its prefill then runs without the decode's rounds)
    // TF_GLM53_MULTI: キャッシュより後ろで計算する行数 (rank 0 のスケジューラの見積もり。pool_gen が `fresh_gen` のときに
    // 数えた) と、短い prompt として受け入れたか (その prefill は decode のラウンドを挟まずに進む)
    fresh: u64 = 0,
    fresh_gen: ?u64 = null,
    short: bool = false,
    // TF_GLM53_MULTI: its prefill gives way to a short prompt (the scheduler thread's): when it ends, the chunks done
    // are committed and the job goes back to the queue, its stream open, to resume from them
    // TF_GLM53_MULTI: その prefill は短い prompt に譲る (スケジューラのスレッドのもの): prefill が終わると、終わった
    // チャンクを確定し、job はストリームを開いたまま待ち行列へ戻り、そこから再開する
    yielding: bool = false,
    yields: u32 = 0,
    yield_row: u64 = 0,
    // /health's prefill_ms: its chunks' own time so far (the waits left out), the part from admissions before it gave
    // way, and this admission's first chunk (the host's mid_paused_ns counts this admission's pauses)
    // /health の prefill_ms: ここまでのチャンク自体の時間 (待ちは除く)、譲る前の受け入れの分、この受け入れの最初のチャンク
    // (この受け入れで止まっていた時間はホストの mid_paused_ns)
    prefill_ns: i96 = 0,
    prefill_base_ns: i96 = 0,
    adm_t0: ?i96 = null,
    // /health's progress.requests: its slot of the decode window (TF_GLM53_MULTI; 0 alone), its conversation's pool
    // slot (from its plan, set by the engine thread's callbacks) and its reply tokens so far (under the host's mutex)
    // /health の progress.requests 用: decode ウィンドウの枠 (TF_GLM53_MULTI。単独なら 0)、会話の pool のスロット
    // (計画から。エンジンのスレッドのコールバックが設定)、ここまでの応答のトークン数 (ホストの mutex の下)
    conv: u8 = 0,
    pool_id: ?u64 = null,
    // Fiore-lab: the pool slot the reply's last segment wrote (notePool): its next segment's admission (a fold or a
    // cut) passes it on as Header.own_slot, so the pool continues it from where the context changed
    // Fiore-lab: 応答の直前の区間が書いた pool のスロット (notePool)。次の区間の受け入れ (畳み込みや打ち切り) が
    // Header.own_slot として渡し、pool は文脈が変わった所からそれを続ける
    pool_x: ?u64 = null,
    generated: u64 = 0,
    // the think block (Request.think_budget, think_notice_every; rank 0's engine thread): its tokens so far (the
    // notices written in it count), whether it closed, the next notice's number, and a notice or the budget's cut due,
    // waiting for a line break (since which think token). `cut`: the text rank 0 stopped the decode for, written when
    // the engine returns; the reply then goes on as a new admission from the prompt, the reply so far and that text
    // (the cache resumes all but the text): `segs` of them so far, `seg_prompt` the prompt of the one to run
    // 思考ブロック (Request.think_budget、think_notice_every。rank 0 のエンジンのスレッド): ここまでのトークン数 (書き込んだ
    // お知らせも数える)、閉じたか、次のお知らせの番号、期限が来て改行を待っているお知らせか予算の打ち切り (どの思考トークン
    // からか)。`cut`: rank 0 が decode を止めた理由の文。エンジンが戻ったら書き込む。応答はそのあと、prompt・ここまでの応答・
    // その文からの新しい受け入れとして続く (キャッシュは文以外を再開する): ここまでの数が `segs`、次に実行するものの prompt が
    // `seg_prompt`
    think_n: u64 = 0,
    think_shut: bool = false,
    notice_k: u32 = 1,
    want: ?[]const u32 = null,
    want_wrap: bool = false,
    want_at: u64 = 0,
    cut: ?Cut = null,
    want_fold: enum { none, recent, open, branch } = .none,
    // the fold (Request.think_fold): the reply as the model goes on from it (`live`; the client gets `emitted`, which
    // equals it until a fold), its think tokens since the last fold, the folds so far, and a checkpoint being asked
    // for or written (from which index of `live`, how many tokens so far)
    // 畳み込み (Request.think_fold): モデルが続ける元の応答 (`live`。クライアントが受け取るのは `emitted` で、畳むまでは同じ)、
    // 最後に畳んでからの思考トークン数、ここまで畳んだ回数、求めている・書いているチェックポイント (`live` のどこから、何トークン)
    live: std.ArrayList(u32) = .empty,
    fold_n: u64 = 0,
    folds: u32 = 0,
    // the think tokens the last fold kept in the context, and the thinking filled its share again with no fold left
    // (Request.think_fold_max), so it closed as at its budget's end
    // 最後の畳み込みが文脈に残した思考トークン数と、畳み込みが残っていないのに思考がまた割合を埋めた (Request.think_fold_max)
    // ので予算の終わりと同じに閉じたこと
    fold_kept: u64 = 0,
    room_full: bool = false,
    // the thinking budget left no room for a summary midway and the summary at the close (summaryWontFit), so the
    // block closed at the fold instead / 思考の予算に途中の要約と思考完了時要約の余地がなく、畳む所で閉じた
    budget_near: bool = false,
    // the summary beside the reply (Request.think_fold branch): being written, its tokens so far (not in `live`), when
    // it was asked for, and where in `live` the reasoning not yet summarized starts (past the head and the summaries)
    // 応答の横での要約 (Request.think_fold branch): 書いている最中か、ここまでのトークン (`live` には入れない)、求めた時刻、
    // `live` のどこからがまだ要約していない思考か (先頭と要約の後ろ)
    br_writing: bool = false,
    br_text: std.ArrayList(u32) = .empty,
    br_t0: i96 = 0,
    // where in br_text the answer begins after the thinking it wrote first (its last </think>; 0: none)
    // br_text のどこから答えか (先に書いた思考の後、最後の </think>。0 はなし)
    br_split: usize = 0,
    // which step the branch being written is (Request.think_step_*): midway goes on thinking after it, budget and close
    // shut the block; and the reply ended in its think block, which the engine closed (Request.think_unclosed)
    // 書いている枝がどの手順か (Request.think_step_*): midway はそのあと思考を続け、budget と close はブロックを閉じる。
    // および応答が思考ブロックの中で終わり、エンジンが閉じた (Request.think_unclosed)
    br_step: BrStep = .midway,
    br_cap: u64 = 0,
    // the budget's step left the block open: the model goes on until the thinking's memory is full (memoryFull); and
    // the room its rewrite has (budgetRoom), while that is asked for
    // 予算の手順がブロックを開いたままにした: 思考のメモリが埋まるまでモデルが続ける (memoryFull)。およびその書き換えに使える
    // 場所 (budgetRoom)。求めている間
    went_on: bool = false,
    br_room: ?u64 = null,
    unclosed: bool = false,
    fold_from: usize = 0,
    ck: enum { none, opening, writing } = .none,
    ck_start: usize = 0,
    ck_n: u64 = 0,
    // the stage of the checkpoint asked for (an index of the request's think_fold_opens)
    // 求めたチェックポイントの段階 (要求の think_fold_opens の添字)
    ck_stage: u8 = 0,
    // the summary at the close (Request.think_fold_at_close): the checkpoint asked for or written is it (it ends by
    // closing the block, not by going on), it was asked for once already, and the close token just taken waits for it
    // (Out.tokens takes that token back out of the reply and the context)
    // 閉じる時の要約 (Request.think_fold_at_close): 求めている・書いているチェックポイントがそれである (続けずにブロックを
    // 閉じて終わる)、すでに一度求めた、いま受け取った閉じるトークンが要約を待つ (Out.tokens がそのトークンを応答と文脈から
    // 取り除く)
    ck_close: bool = false,
    close_asked: bool = false,
    drop_close: bool = false,
    segs: u32 = 0,
    seg_prompt: ?[]u32 = null,
    // Host.byteResume: the prompt with the slot's ids where the bytes agree, and that the match was tried
    // Host.byteResume: バイトが一致する所をスロットの id にした prompt と、照合を試したこと
    matched: ?[]u32 = null,
    byte_tried: bool = false,
    // the context window the segments run in (the engine's capacity, set by serve; 0: max_tokens alone limits them)
    // 区間が動く文脈の長さ (エンジンの容量。serve が設定する。0 なら max_tokens だけが制限する)
    context: usize = 0,
    // the model closed the think block with the token just taken (thinkCut; Out.tokens checks its ids), and how many
    // times the context's think ids were remade (Host.canonThink)
    // モデルがいま受け取ったトークンで思考ブロックを閉じた (thinkCut。Out.tokens がその id を確かめる)、文脈の思考の id を
    // 作り直した回数 (Host.canonThink)
    closed_now: bool = false,
    canons: u32 = 0,
    // the segments before: their rounds and drafts, the first one's prefill, the others' decode and re-admissions
    // それより前の区間: ラウンドと draft、最初の区間の prefill、ほかの区間の decode と受け入れ直し
    acc_rounds: u64 = 0,
    acc_drafted: u64 = 0,
    acc_accepted: u64 = 0,
    acc_prefill_ns: u64 = 0,
    acc_decode_ns: u64 = 0,

    /// The prompt the reply was asked with: the request's, or that with the slot's ids where the bytes agree (Host.byteResume).
    /// 応答を求められた prompt: 要求のもの、またはバイトが一致する所をスロットの id にしたもの (Host.byteResume)。
    fn prompt0(job: *const Job) []const u32 {
        return job.matched orelse job.request.prompt;
    }

    /// The prompt of the segment to run: the request's, or the continuation's (the prompt and the reply so far).
    /// 実行する区間の prompt: 要求のもの、または続きのもの (prompt とここまでの応答)。
    fn promptNow(job: *const Job) []const u32 {
        return job.seg_prompt orelse job.prompt0();
    }

    /// The reply tokens the segment may still write: what max_tokens leaves, and no more than the context leaves past
    /// the segment's prompt (with folds the reply can outgrow the context; the context itself never does).
    /// その区間がまだ書ける応答のトークン数: max_tokens の残り。ただし区間の prompt の後ろに文脈が残す分まで (畳み込みが
    /// あると応答は文脈より長くなりうるが、文脈そのものは超えない)。
    fn maxNow(job: *const Job) u32 {
        const left = job.request.max_tokens -| @as(u32, @intCast(@min(job.emitted.items.len, std.math.maxInt(u32))));
        if (job.context == 0) return left;
        return @intCast(@min(left, job.context -| job.promptNow().len));
    }

    /// After the reply token `t` (rank 0's engine thread): what to do in the think block now, at a line break (or
    /// think_grace think tokens after it fell due), if the text and a token after it fit in max_tokens; null otherwise.
    /// In order: a checkpoint being written ends at its marker, or past think_fold_cap (the engine ends it); the
    /// budget's cut; a fold once think_fold_window think tokens are in the context past the last one (recent: the
    /// context folds at once; checkpoint: the model is asked for its checkpoint first); a notice.
    /// 応答のトークン `t` のあと (rank 0 のエンジンのスレッド): 思考ブロックでいますること。改行のところで (または期限から
    /// think_grace 個の思考トークンが過ぎて)、文とその後の 1 トークンが max_tokens に収まるとき。それ以外は null。順に:
    /// 書いているチェックポイントは終わりの印で終わる (think_fold_cap を過ぎたらエンジンが終える)。予算の打ち切り。最後に畳んで
    /// から think_fold_window 個の思考トークンが文脈に入ったら畳む (recent はすぐ畳む。checkpoint はまずチェックポイントを
    /// 求める)。お知らせ。
    fn thinkCut(job: *Job, t: u32) ?Cut {
        const r = job.request;
        const end = r.think_end orelse return null;
        const folding = r.think_fold != .off and r.think_fold_window > 0;
        const closing = r.think_fold_at_close > 0 and r.think_fold_close_open.len > 0;
        const stepping = r.think_step_budget != null or r.think_step_close != null or r.think_unclosed == .close;
        if (job.think_shut or (r.think_budget == 0 and r.think_notice_every == 0 and !folding and !closing and !stepping)) return null;
        if (t == end) {
            if (job.stepAtClose()) |c| return c;
            // the summary at the close: a close inside it ends it there (folded, then closed); a long enough block
            // waits for it before it closes (its close token taken back), once a reply
            // 閉じる時の要約: その中で閉じたらそこで終える (畳んでから閉じる)。十分長いブロックは閉じる前にそれを待つ
            // (閉じるトークンは取り除く)。応答に 1 回だけ
            if (job.ck == .writing and job.ck_close) {
                job.ck = .none;
                job.want = null;
                if (job.closeFold(&.{})) |c| return c;
            } else if (closing and !job.close_asked and job.ck == .none and !job.want_wrap and job.fold_n >= r.think_fold_at_close) {
                if (job.fitsSummary(.{ .text = r.think_fold_close_open })) |c| {
                    job.want = null;
                    job.want_fold = .none;
                    job.ck = .opening;
                    job.ck_close = true;
                    job.close_asked = true;
                    job.drop_close = true;
                    return c;
                }
            }
            job.think_shut = true;
            job.want = null;
            job.ck = .none;
            job.closed_now = true;
            return null;
        }
        job.think_n += 1;
        job.fold_n += 1;
        const at_break = isBreak(r.think_breaks, t);
        // past the budget's step the limit is the thinking's memory, and no text is needed: the model has organised its
        // thoughts
        // 予算の手順の後の上限は思考のメモリ。文は要らない: モデルは考えを整理したあと
        const over = if (job.went_on) job.memoryFull() else r.think_budget > 0 and job.think_n + job.budgetReserve() >= r.think_budget;
        // the summary at the close runs on past the budget (the budget's cut may be what asked for it)
        // 閉じる時の要約は予算を過ぎても続ける (予算の打ち切りがそれを求めたこともある)
        if (job.ck == .writing and (job.ck_close or !over)) {
            job.ck_n += 1;
            if (r.think_fold_end) |s| if (s.check(s.ctx, job.live.items[job.ck_start..])) {
                job.ck = .none;
                if (job.ck_close) return job.closeFold(r.think_close);
                return job.fits(.{ .text = r.think_fold_resumes[job.ck_stage], .fold = .checkpoint });
            };
            const ck_cap = api.foldCapOf(r.think_fold_cap_share, r.think_fold_cap, job.fold_n -| job.ck_n);
            if (job.ck_n >= ck_cap and (at_break or job.ck_n - ck_cap >= think_grace)) {
                job.ck = .none;
                if (job.ck_close) return job.closeFold(r.think_close);
                return job.fits(.{ .text = r.think_fold_closes[job.ck_stage], .fold = .checkpoint });
            }
            return null;
        }
        // the budget's cut takes the place of a checkpoint, a fold or a notice waiting
        // 予算の打ち切りは、書きかけのチェックポイント、待っている畳み込みやお知らせに代わる
        job.ck = .none;
        if (over and !job.want_wrap) {
            job.want = if (r.think_wrap.len > 0 and !job.went_on) r.think_wrap else r.think_close;
            job.want_wrap = true;
            job.want_fold = .none;
            job.want_at = job.think_n;
        } else if (job.want == null and folding and !job.went_on and ((job.fold_n >= r.think_fold_window and !job.foldsSpent()) or job.regionFull())) {
            // a fold is due: think_fold_window tokens since the last one, or the block filling its room in the context
            // (think_fold_region, where the summaries stack); with no fold left only the full room closes it
            // 畳み込みの時: 前回から think_fold_window 個、またはブロックが文脈の中の領域 (think_fold_region、要約はここに
            // 積み上がる) を埋めた。畳み込みが残っていなければ、領域が埋まった時だけ閉じる
            job.budget_near = job.summaryWontFit();
            if (job.foldsSpent() or job.budget_near) {
                // no fold left (think_fold_max, or the summaries kept and the next would pass the region): the
                // thinking filled its room in the context, so it closes as at its budget's end
                // 畳み込みが残っていない (think_fold_max、または積んだ要約と次の要約が領域を越える): 思考が文脈の中の領域を
                // 埋めたので、予算の終わりと同じに閉じる
                job.want = if (r.think_wrap.len > 0) r.think_wrap else r.think_close;
                job.want_wrap = true;
                job.want_fold = .none;
                job.want_at = job.think_n;
                job.room_full = true;
            } else {
                const ck = r.think_fold == .checkpoint;
                if (ck) job.ck_stage = job.foldStage();
                job.want = if (ck) r.think_fold_opens[job.ck_stage] else &.{};
                job.want_fold = if (ck) .open else if (r.think_fold == .branch) .branch else .recent;
                job.want_at = job.think_n;
            }
        } else if (job.want == null and r.think_notice_every > 0 and job.notice_k <= r.think_notices.len and job.think_n >= @as(u64, job.notice_k) * r.think_notice_every) {
            job.want = r.think_notices[job.notice_k - 1];
            job.want_at = job.think_n;
        }
        const w = job.want orelse return null;
        if (!at_break and job.think_n - job.want_at < think_grace) return null;
        job.want = null;
        const kind = job.want_fold;
        job.want_fold = .none;
        if (job.want_wrap) {
            // a step takes the place of the budget's closing sentence: the model writes beside the reply, then the
            // block closes after it (goOn)
            // 手順が予算の締めの文に代わる: モデルが応答の横に書き、そのあとブロックを閉じる (goOn)
            if (r.think_step_budget) |st| if (!job.close_asked and job.fold_n >= st.min) {
                if (job.stepOpen(.budget, st)) |c| {
                    job.close_asked = true;
                    return c;
                }
            };
            // the summary at the close takes the place of the budget's closing sentence: the model writes it, then
            // the block folds to it and closes (closeFold), as when the model closes it itself
            // 閉じる時の要約が予算の締めの文に代わる: モデルがそれを書き、ブロックはそこへ畳んで閉じる (closeFold)。
            // モデルが自分で閉じた時と同じ
            if (closing and !job.close_asked and r.think_fold_close_budget_open.len > 0 and job.fold_n >= r.think_fold_at_close) {
                if (job.fitsSummary(.{ .text = r.think_fold_close_budget_open })) |c| {
                    job.ck = .opening;
                    job.ck_close = true;
                    job.close_asked = true;
                    return c;
                }
            }
            job.think_shut = true;
            return job.fits(.{ .text = w });
        }
        switch (kind) {
            .recent => return job.fits(.{ .fold = .recent }),
            .branch => return job.branchOpen(),
            .open => {
                const c = job.fits(.{ .text = w }) orelse return null;
                job.ck = .opening;
                return c;
            },
            .none => {
                job.notice_k += 1;
                return job.fits(.{ .text = w });
            },
        }
    }

    /// The summary beside the reply asked for (think_fold branch), when its request and the most the answer may write
    /// (its thinking and a summary, twice think_fold_cap and the line break it may wait for) fit past the context, and
    /// the label, that and the resume line in max_tokens; else the recent fold.
    /// 応答の横での要約を求める (think_fold branch)。その依頼と答えが書きうる最大 (思考と要約で think_fold_cap の 2 倍と
    /// 待ちうる改行) が文脈の後ろに、見出し・それ・続きの一行が max_tokens に収まるとき。収まらなければ recent で畳む。
    fn branchOpen(job: *Job) ?Cut {
        return job.stepOpen(.midway, job.midwayStep()) orelse job.fits(.{ .fold = .recent });
    }

    /// The branch for step `st` asked for, when its request and the most the model may write in it fit past the context,
    /// and the texts around it and that in max_tokens; null otherwise.
    /// 手順 `st` の枝を求める。その依頼とモデルが書きうる最大が文脈の後ろに、前後の文とそれが max_tokens に収まるとき。
    /// 収まらなければ null。
    fn stepOpen(job: *Job, which: BrStep, st: api.ThinkStep) ?Cut {
        const r = job.request;
        if (which != .midway) {
            const room = job.budgetRoom(st);
            if (room < 256) return null;
            job.br_room = room;
        }
        const most: usize = job.stepMost(st) + 1;
        const shut: usize = if (which == .midway) 0 else r.think_close.len;
        const in_reply = job.emitted.items.len + st.label.len + st.heading.len + most + st.after.len + shut <= r.max_tokens;
        const in_context = job.context == 0 or job.prompt0().len + job.live.items.len + st.ask.len + most <= job.context;
        if (!in_reply or !in_context) {
            job.br_room = null;
            return null;
        }
        job.br_step = which;
        return .{ .fold = .branch_open };
    }

    /// The most the model may write in a branch for `st`: think_fold_cap and the line break it may wait for up to its
    /// think_end (take before), or twice think_fold_cap (the thinking it wrote first and the answer).
    /// `st` の枝でモデルが書きうる最大: think_end までの think_fold_cap と待ちうる改行 (take before)、または
    /// think_fold_cap の 2 倍 (先に書いた思考と答え)。
    fn stepMost(job: *const Job, st: api.ThinkStep) u64 {
        const cap = job.stepCap();
        return (if (st.take == .before) cap else 2 * cap) + think_grace;
    }

    /// The tokens a branch's result may take: think_fold_cap, or with 0 the thinking it rewrites (a rewrite
    /// is never longer than what it rewrites; one that does not shrink only folds again soon).
    /// 枝の結果の上限: think_fold_cap。0 なら書き直す元の思考の長さ (元より大きくなることはない。小さくならなければ
    /// すぐ次にまとめるだけ)。
    fn stepCap(job: *const Job) u64 {
        const own: u64 = @max(job.fold_n, 1024);
        const cap: u64 = api.foldCapOf(job.request.think_fold_cap_share, job.request.think_fold_cap, own);
        // a rewrite at the block's end has no size of its own to keep to, only the room there is
        // ブロックの終わりの書き換えに守るべき大きさはない。あるのは使える場所だけ
        if (job.br_room) |room| return @min(own, room);
        return if (cap > 0) cap else own;
    }

    /// The midway step: the request's own, or the server's summary beside the reply.
    /// 途中の手順: 要求のもの、なければサーバーの「応答の横での要約」。
    fn midwayStep(job: *const Job) api.ThinkStep {
        const r = job.request;
        return r.think_step_midway orelse .{ .ask = r.think_fold_branch_ask, .label = r.think_fold_branch_label, .after = r.think_fold_branch_resume };
    }

    /// The step of the branch asked for or being written.
    /// 求めている・書いている枝の手順。
    fn step(job: *const Job) api.ThinkStep {
        return switch (job.br_step) {
            .midway => job.midwayStep(),
            .budget => job.request.think_step_budget.?,
            .close => job.request.think_step_close.?,
        };
    }

    /// The step at the model's own close, when one is set and enough was thought since the last rewrite: the close
    /// token leaves the reply and the context again, and goes back after the step (goOn).
    /// モデル自身が閉じた時の手順。設定があり、前回の書き換えから十分考えたとき: 閉じるトークンを応答と文脈から取り除き、
    /// 手順のあとで閉じ直す (goOn)。
    fn stepAtClose(job: *Job) ?Cut {
        const st = job.request.think_step_close orelse return null;
        if (job.close_asked or job.ck != .none or job.want_wrap or job.fold_n < st.min) return null;
        const c = job.stepOpen(.close, st) orelse return null;
        job.want = null;
        job.want_fold = .none;
        job.close_asked = true;
        job.drop_close = true;
        return c;
    }

    /// Whether the thinking may add no more after the budget's step: what it holds in the context fills its memory
    /// (think_budget, not a fixed size), or only the answer's part is left of the context or the reply.
    /// 予算の手順のあと、思考がもう足せないか: 文脈の中の思考がメモリ (think_budget。固定サイズではない) を埋めた、または
    /// 文脈か応答の残りが答えの分だけになった。
    fn memoryFull(job: *const Job) bool {
        const r = job.request;
        const n: u64 = job.live.items.len;
        if (n >= r.think_budget) return true;
        if (job.context > 0 and job.prompt0().len + n + r.think_answer >= job.context) return true;
        return job.emitted.items.len + r.think_answer >= r.max_tokens;
    }

    /// The most the budget's rewrite may write: what the context has past its request, at most half of what the reply
    /// has left. A rewrite this long is cut there, and the thinking goes on from what it wrote.
    /// 予算の書き換えが書ける最大: 依頼の後ろに文脈が持つ分、ただし応答の残りの半分まで。ここまで書いたら切り、書いたものから
    /// 思考が続く。
    fn budgetRoom(job: *const Job, st: api.ThinkStep) u64 {
        const r = job.request;
        const around: u64 = st.label.len + st.heading.len + st.after.len + r.think_close.len + think_grace + 1;
        const in_reply: u64 = (@as(u64, r.max_tokens) -| (job.emitted.items.len + around)) / 2;
        if (job.context == 0) return in_reply;
        return @min(in_reply, @as(u64, job.context) -| (job.prompt0().len + job.live.items.len + st.ask.len + think_grace + 1));
    }

    /// How far before the budget's end the budget's step starts (5% by default, a setting); 0 when it will
    /// not run.
    /// 予算の手順が予算の終わりのどれだけ手前で始まるか (既定 5%、設定項目)。動かないなら 0。
    fn budgetReserve(job: *const Job) u64 {
        const r = job.request;
        const st = r.think_step_budget orelse return 0;
        if (job.close_asked or job.fold_n < st.min) return 0;
        return @as(u64, r.think_budget) * r.think_step_margin / 100;
    }

    /// The reply ended with its think block open (the model wrote its answer there and its EOS): the block closes, so
    /// the model writes the answer (Request.think_unclosed close; max_tokens or the context left: null).
    /// 応答が思考ブロックを開いたまま終わった (モデルが答えをそこに書いて EOS): ブロックを閉じ、モデルに答えを書かせる
    /// (Request.think_unclosed close。max_tokens か残りの文脈が足りなければ null)。
    fn unclosedCut(job: *Job) ?Cut {
        const r = job.request;
        if (r.think_unclosed != .close or r.think_end == null or job.think_shut or job.br_writing or job.cut != null or job.quiet or job.emitted.items.len == 0) return null;
        const c = job.fits(.{ .text = r.think_close }) orelse return null;
        job.think_shut = true;
        job.unclosed = true;
        return c;
    }

    /// The summary beside the reply written: `live` to the block's head and the summaries before (fold_from), the
    /// label, this summary and the resume line (the client got the label and the summary as they came, and gets the
    /// line too). The summary is the answer after the thinking it wrote first (br_split), or, when that is too short
    /// to stand for the reasoning, the thinking itself; `.recent` when both are: the head and summaries kept, then the
    /// reasoning's last think_fold_keep tokens.
    /// 応答の横での要約を書き終えた: `live` をブロックの先頭とそれまでの要約 (fold_from)、見出し、この要約、続きの一行に
    /// する (クライアントは見出しと要約を届いたとおりに受け取っており、続きの一行も受け取る)。短すぎて思考の代わりに
    /// ならないときは `.recent`: 先頭と要約を残し、その後に思考の最後の think_fold_keep 個。
    const BrStep = enum { midway, budget, close };
    const BranchFold = enum { summary, recent, kept };

    fn foldBranch(job: *Job, gpa: Allocator) !BranchFold {
        const r = job.request;
        const st = job.step();
        defer job.br_text.clearRetainingCapacity();
        job.br_writing = false;
        const n = job.live.items.len;
        // a request's own step rewrites the block from its start (<think>aaabbb -> <think>ab), the server's
        // summary keeps the block's head
        // 要求の手順はブロックを先頭から書き換える (<think>aaabbb -> <think>ab)。サーバーの要約はブロックの先頭を残す
        const head: usize = if (job.br_step == .midway and r.think_step_midway == null) r.think_fold_head else 0;
        const from = @min(n, @max(job.fold_from, head));
        const answer = job.br_text.items[job.br_split..];
        const notes = job.br_text.items[0..job.br_split];
        const summary = if (st.take == .before)
            (if (notes.len >= 16) notes else answer)
        else
            (if (answer.len >= 16) answer else notes);
        if (summary.len < 16) {
            // nothing came back: midway the recent fold, at the block's end the block as it is
            // 何も返らなかった: 途中なら recent で畳み、ブロックの終わりならブロックはそのまま
            if (job.br_step != .midway) return .kept;
            const tail = @max(from, n -| r.think_fold_keep);
            std.mem.copyForwards(u32, job.live.items[from..], job.live.items[tail..n]);
            job.live.shrinkRetainingCapacity(from + (n - tail));
            job.fold_n = job.live.items.len;
            return .recent;
        }
        // shown only: the client has read it, the context stays as it was
        // 見せるだけ: クライアントは読んだ。文脈はそのまま
        if (st.place == .show) return .summary;
        if (st.place == .replace) job.live.shrinkRetainingCapacity(from);
        // with no text around it, the summary still starts and ends a line
        // 前後に文がなくても、要約は行の頭で始まり行末で終わる
        if (st.label.len == 0) try job.endLine(gpa);
        try job.live.appendSlice(gpa, st.label);
        try job.live.appendSlice(gpa, st.heading);
        try job.live.appendSlice(gpa, summary);
        if (st.after.len == 0) try job.endLine(gpa);
        try job.live.appendSlice(gpa, st.after);
        try job.emitted.appendSlice(gpa, st.after);
        job.fold_from = job.live.items.len;
        job.fold_n = st.after.len;
        return .summary;
    }

    /// A line break after the context's last token unless it holds one (the break's token: think_close's lead).
    /// 文脈の最後のトークンが改行を持たなければ改行を足す (改行のトークンは think_close の先頭)。
    fn endLine(job: *Job, gpa: Allocator) !void {
        const r = job.request;
        const l = job.live.items;
        if (l.len == 0 or r.think_close.len < 2 or isBreak(r.think_breaks, l[l.len - 1])) return;
        try job.live.append(gpa, r.think_close[0]);
    }

    /// Whether a summary midway (the checkpoint, or the summary beside the reply with the thinking its answer may write
    /// first) and then the summary at the close no longer fit in the thinking budget left:
    /// the thinking then closes now, as at the budget's end (with the summary
    /// at the close when it is on). Never for the recent fold, which writes nothing.
    /// 途中の要約 (チェックポイント、または応答の横での要約と答えが先に書きうる思考) とその後の思考完了時要約が、思考の
    /// 予算の残りに収まらないか: そのときはいま思考を
    /// 閉じる。予算の終わりと同じ (思考完了時要約がオンならそれを書く)。何も書かない recent では使わない。
    fn summaryWontFit(job: *const Job) bool {
        const r = job.request;
        if (r.think_budget == 0 or r.think_fold == .recent or r.think_fold_cap == 0) return false;
        const cap: u64 = api.foldCapOf(r.think_fold_cap_share, r.think_fold_cap, job.fold_n);
        const midway: u64 = if (r.think_fold == .branch) job.stepMost(job.midwayStep()) else cap + think_grace + (if (r.think_fold_opens.len > 0) r.think_fold_opens[0].len else 0);
        const at_close: u64 = if (r.think_step_budget) |st| job.stepMost(st) + r.think_close.len else if (r.think_fold_at_close > 0 and r.think_fold_close_budget_open.len > 0) cap + think_grace + r.think_fold_close_budget_open.len else 0;
        return job.think_n + midway + at_close > r.think_budget;
    }

    /// Whether the reply may fold no more: think_fold_max folds done (0: no limit), or what the last fold kept (the
    /// head and the summaries) and one more summary would pass the block's room in the context (think_fold_region).
    /// もう畳めないか: think_fold_max 回畳んだ (0 は制限なし)、または前回の畳み込みが残したもの (先頭と要約) ともう 1 つの
    /// 要約が、文脈の中のブロックの領域 (think_fold_region) を越える。
    fn foldsSpent(job: *const Job) bool {
        const r = job.request;
        if (r.think_fold_max > 0 and job.folds >= r.think_fold_max) return true;
        if (job.folds == 0 or r.think_fold_region == 0) return false;
        const st = job.midwayStep();
        const next: u64 = api.foldCapOf(r.think_fold_cap_share, r.think_fold_cap, job.fold_n) + think_grace + st.label.len + st.heading.len + st.after.len;
        return job.fold_kept + next > r.think_fold_region;
    }

    /// Whether the block's tokens in the context (the head, the summaries kept and the thinking since) fill its room
    /// there (think_fold_region; 0: none).
    /// 文脈の中のブロックのトークン (先頭、残した要約、その後の思考) が、そこでの領域 (think_fold_region。0 はなし) を埋めたか。
    fn regionFull(job: *const Job) bool {
        const r = job.request;
        return r.think_fold_region > 0 and job.live.items.len >= r.think_fold_region;
    }

    /// The checkpoint's stage now: 0 organise; 1 converge (after a fold, or half the thinking budget spent); 2 commit
    /// (a quarter of the budget left). At most think_fold_stages - 1.
    /// いまのチェックポイントの段階: 0 整理。1 収束 (一度畳んだあと、または思考の予算の半分を使ったあと)。2 確定 (予算の
    /// 残りが 4 分の 1)。最大で think_fold_stages - 1。
    fn foldStage(job: *const Job) u8 {
        const r = job.request;
        const n = @min(r.think_fold_stages, r.think_fold_opens.len, r.think_fold_resumes.len, r.think_fold_closes.len);
        if (n <= 1) return 0;
        const b: u64 = r.think_budget;
        const left = b -| job.think_n;
        if (n >= 3 and b > 0 and left * 4 <= b) return 2;
        if (job.folds >= 1 or (b > 0 and left * 2 <= b)) return 1;
        return 0;
    }

    /// The summary at the close written: `ending` (the block's end; empty when the model closed it) goes in after it and
    /// the block folds to it, the think block then shut; null when that does not fit (the model then closes it itself).
    /// 閉じる時の要約を書き終えた: その後に `ending` (ブロックの終わり。モデルが閉じた時は空) を入れて要約へ畳み、思考ブロックを
    /// 閉じる。収まらなければ null (モデルが自分で閉じる)。
    fn closeFold(job: *Job, ending: []const u32) ?Cut {
        job.ck_close = false;
        const c = job.fits(.{ .text = ending, .fold = .checkpoint }) orelse return null;
        job.think_shut = true;
        return c;
    }

    /// `c`, the request for the summary at the close, if it, a summary of think_fold_cap tokens and an answer as long
    /// still fit in max_tokens; null otherwise (the block then closes as it would have).
    /// `c` (閉じる時の要約を求める文) とその要約 think_fold_cap 個、同じ長さの答えがまだ max_tokens に収まれば `c`。
    /// 収まらなければ null (ブロックはそのまま閉じる)。
    fn fitsSummary(job: *const Job, c: Cut) ?Cut {
        const room = c.text.len + 2 * @as(usize, @intCast(api.foldCapOf(job.request.think_fold_cap_share, job.request.think_fold_cap, job.fold_n))) + 1;
        if (job.emitted.items.len + room > job.request.max_tokens) return null;
        return job.fits(c);
    }

    /// `c` if its text and a token after it fit in max_tokens (otherwise the reply ends as it would have).
    /// `c` の文とその後の 1 トークンが max_tokens に収まれば `c` (収まらなければ応答はそのまま終わる)。
    fn fits(job: *const Job, c: Cut) ?Cut {
        if (c.text.len == 0 and c.fold == .none) return null;
        if (job.emitted.items.len + c.text.len + 1 > job.request.max_tokens) return null;
        if (job.context > 0 and job.prompt0().len + job.live.items.len + c.text.len + 1 > job.context) return null;
        return c;
    }

    /// The recent fold: `live` to the think block's first think_fold_head tokens and its last think_fold_keep.
    /// recent の畳み込み: `live` を思考ブロックの最初の think_fold_head 個と最後の think_fold_keep 個にする。
    fn foldRecent(job: *Job) void {
        const r = job.request;
        const n = job.live.items.len;
        const head = @min(r.think_fold_head, n);
        const from = @max(head, n -| r.think_fold_keep);
        std.mem.copyForwards(u32, job.live.items[head..], job.live.items[from..n]);
        job.live.shrinkRetainingCapacity(head + (n - from));
        job.fold_n = n - from;
    }

    /// The checkpoint fold: `live` to the think block's first think_fold_head tokens, think_fold_label and the
    /// checkpoint just written (with the line the thinking goes on after); the request for it and the reasoning
    /// before are dropped.
    /// checkpoint の畳み込み: `live` を思考ブロックの最初の think_fold_head 個、think_fold_label、いま書いたチェックポイント
    /// (続きの一行を含む) にする。チェックポイントを求めた文とそれより前の思考は外す。
    fn foldCheckpoint(job: *Job, gpa: Allocator) !void {
        const r = job.request;
        const l = job.live.items;
        const head = @min(r.think_fold_head, job.ck_start);
        const body = l[job.ck_start..];
        const out = try gpa.alloc(u32, head + r.think_fold_label.len + body.len);
        defer gpa.free(out);
        @memcpy(out[0..head], l[0..head]);
        @memcpy(out[head..][0..r.think_fold_label.len], r.think_fold_label);
        @memcpy(out[head + r.think_fold_label.len ..], body);
        job.live.clearRetainingCapacity();
        try job.live.appendSlice(gpa, out);
        job.fold_n = body.len;
    }
};

/// A branch's moment in the log: nothing midway, as the log always read.
/// ログでの枝の場面: 途中はこれまでのログのとおり何も付けない。
fn stepName(which: anytype) []const u8 {
    return switch (which) {
        .midway => "",
        .budget => " at the budget's end",
        .close => " at the close",
    };
}

/// How many think tokens a notice or the budget's cut waits at most for a line break.
/// お知らせや予算の打ち切りが改行を待つ思考トークンの最大数。
const think_grace: u64 = 256;

/// Whether `t` is in `sorted` (the tokens holding a line break, ascending).
/// `t` が `sorted` (改行を含むトークン、昇順) にあるか。
fn isBreak(sorted: []const u32, t: u32) bool {
    var lo: usize = 0;
    var hi = sorted.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (sorted[mid] < t) lo = mid + 1 else hi = mid;
    }
    return lo < sorted.len and sorted[lo] == t;
}

pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    driver: cuda.Driver,
    ctx: cuda.Context,
    nccl: cuda.nccl.Library,
    group: cuda.tp.Group,
    eng: *glm.engine.Engine,
    ch: glm.lockstep.Channel,
    info_: api.Info,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(api.Id) = .empty,
    running: ?*Job = null,
    closing: bool = false,
    // requests that may wait beside the running one (TF_GLM53_QUEUE); past it the server answers 503 (busy)
    // 実行中のもののほかに待てる要求の数 (TF_GLM53_QUEUE)。超えるとサーバーは 503 (busy) を返す
    queue_max: usize = 32,
    // rank 0: the engine thread; others: the follower
    // rank 0 ではエンジンのスレッド、他のランクではフォロワー
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty,
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_prompt: u64 = 0,
    live_generated: u64 = 0,
    // the requests that ran: their totals, which status adds the running request's progress to (under the mutex, so
    // each total only grows)
    // 実行を終えた要求の合計。status はこれに実行中の要求の進み具合を足す (mutex の下なので、どの合計も減らない)
    totals: api.Progress = .{},
    labels: []u8 = &.{},
    // TF_GLM53_MULTI: conversations decoded together (0: one request at a time), the job of each, its outputs, and
    // the job whose prompt is being prefilled
    // TF_GLM53_MULTI: 一緒に decode する会話の数 (0 なら 1 要求ずつ)、それぞれの job と出力、prompt を prefill 中の job
    multi_n: usize = 0,
    // TF_GLM53_MULTI_WAIT_S (default 300; 0: no limit): how long a request may wait for room in the pool beside the
    // conversations in flight before it is refused as busy (503 / overloaded)
    // TF_GLM53_MULTI_WAIT_S (既定 300、0 は上限なし): 処理中の会話の横に pool の空きができるのを要求が待てる時間。
    // 超えると busy として断る (503 / overloaded)
    room_wait_ns: i96 = 0,
    // TF_GLM53_CANON_CLOSE (near, the default; all; off) and TF_GLM53_CANON_NEAR (default 4096): when the model closes
    // the think block, its ids are remade before the answer if they differ within the last canon_near ids (near) or
    // anywhere (all); a difference further back is left for the next turn to compute (closeCut)
    // TF_GLM53_CANON_CLOSE (既定 near、all、off) と TF_GLM53_CANON_NEAR (既定 4096): モデルが思考ブロックを閉じた時、
    // id が最後の canon_near の中で違えば (near)、またはどこかで違えば (all)、答えの前に作り直す。それより前の違いは次の
    // ターンの計算に任せる (closeCut)
    canon_close: enum { off, near, all } = .near,
    canon_near: usize = 4096,
    // TF_GLM53_BYTE_RESUME: off: nothing; log: each reply's first admission logs how far the prompt's bytes agree with a
    // kept slot past the first id that differs; on (default): the same, and the prompt takes the slot's ids to there
    // TF_GLM53_BYTE_RESUME: off は何もしない。log は応答の最初の受け入れで、最初に違う id より先で prompt のバイトが
    // スロットとどこまで一致するかをログに出す。on (既定) はそれに加え、そこまでの prompt をスロットの id にする
    byte_resume: enum { off, log, on } = .on,
    // TF_GLM53_MULTI_DECODE_SHARE / POST /v1/schedule: the decode's percent of the GPU's time while a prompt waits
    // or is computed beside live conversations (set from the HTTP threads), and rank 0's counting of that time (the
    // scheduler thread's, glm.pace)
    // TF_GLM53_MULTI_DECODE_SHARE / POST /v1/schedule: 使用中の会話の横で prompt が待つか計算されている間の、GPU の時間の
    // decode の取り分 (%、HTTP のスレッドから設定する)。その時間を数える rank 0 の Pace (スケジューラのスレッドのもの、glm.pace)
    share: std.atomic.Value(u8) = .init(30),
    pace: glm.pace.Pace = .{},
    // TF_GLM53_MULTI_SHORT_PROMPT / POST /v1/schedule: a prompt with at most this many rows to compute past the cache
    // goes before the longer ones waiting and is not held by the share (0: none is short); and the pool's changes
    // counted (an admission or a release, the scheduler thread's), so a queued job's estimate is made again after one
    // TF_GLM53_MULTI_SHORT_PROMPT / POST /v1/schedule: キャッシュより後ろで計算する行がこれ以下の prompt は、待っている
    // 長いものより先に入り、取り分で止めない (0 は短いものなし)。pool の変化の数 (受け入れと返却、スケジューラのスレッドの
    // もの): 待ち行列の job の見積もりは、変化の後に作り直す
    short: std.atomic.Value(u32) = .init(8192),
    pool_gen: u64 = 0,
    // the admission in flight: the time its prefill spent paused for the others (lockstep.Live.paused; the scheduler
    // thread's, read under the lock in Out.chunk / prefilled)
    // 実行中の受け入れ: その prefill がほかのために止まっていた時間 (lockstep.Live.paused。スケジューラのスレッドのもの)
    mid_paused_ns: i96 = 0,
    // TF_GLM53_MULTI_SHORT_PREEMPT / POST /v1/schedule (default off): a longer prefill running gives way to a short
    // prompt waiting (yieldFor)
    // TF_GLM53_MULTI_SHORT_PREEMPT / POST /v1/schedule (既定はオフ): 実行中の長い prefill も、待っている短い prompt に
    // 譲る (yieldFor)
    preempt: std.atomic.Value(bool) = .init(false),
    // the three at start (GET / POST /v1/schedule's start)
    // 起動時の 3 つ (GET / POST /v1/schedule の start)
    sched_start: api.ScheduleStart = .{},
    // the last admission sent its job back to the queue (the scheduler thread's)
    // 直前の受け入れが job を待ち行列へ戻した (スケジューラのスレッドのもの)
    requeued: bool = false,
    /// rank 0: the memory the control panel shows (GET /v1/memory), its snapshot refreshed between requests.
    /// rank 0: 制御パネルが出すメモリ (GET /v1/memory)。写しは要求の合間に更新する。
    mem: ?*glm.memory_live.MemLive = null,
    /// rank 0: a change of the routed experts' bits asked for (POST /v1/experts/swap; the engine thread runs it between
    /// requests), its state for GET /v1/experts/swap, and the checkpoint the bits are saved against (under the mutex)
    /// rank 0: 求められた routed expert の bit の変更 (POST /v1/experts/swap。エンジンのスレッドが要求の合間に実行)、
    /// GET /v1/experts/swap 用のその状態、保存する bit の基のチェックポイント (mutex の下)
    swap_want: ?[]u8 = null,
    /// saves so far, for each save's own temporary file (under the mutex) / 保存の回数 (一時ファイルの名前に。mutex の下)
    save_n: u64 = 0,
    swap: Swap = .{},
    ckpt: []const u8 = "",
    act: [glm.engine.multi_max]?*Job = @splat(null),
    outs: [glm.engine.multi_max]Out = undefined,
    prefilling: ?*Job = null,
    // TF_GLM53_MULTI: the job the scheduler holds between the queue and its prefill (taken from the queue, or back from
    // a plan that found no room): a cancel for it is recorded, one for no job held anywhere is not
    // TF_GLM53_MULTI: 待ち行列と prefill の間でスケジューラが持つ job (待ち行列から取った、または空きのなかった計画から
    // 戻った): それへの取り消しは記録し、どこにも無い job への取り消しは記録しない
    admitting: ?*Job = null,

    const Mark = struct { at: i96, tokens: u64 };
    const Swap = struct {
        seq: u32 = 0,
        state: enum { idle, waiting, running, done, failed } = .idle,
        done: u32 = 0,
        layers: u32 = 0,
        stats: glm.expert_swap.Stats = .{},
        err: []const u8 = "",
    };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    pub fn engine(h: *Host) api.Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .experts = expertsFn, .experts_control = controlFn, .schedule = scheduleFn } };
    }

    fn self(ctx: *anyopaque) *Host {
        return @ptrCast(@alignCast(ctx));
    }

    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }

    fn now(h: *Host) i96 {
        return std.Io.Clock.awake.now(h.io).toNanoseconds();
    }

    fn infoFn(ctx: *anyopaque) api.Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h = self(ctx);
        // only rank 0 takes requests: the others follow it and never run a queue of their own
        // 要求を受けるのは rank 0 だけ。他のランクは rank 0 に従い、自分の列を処理しない
        if (h.group.rank != 0) return error.Closed;
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .id = id, .request = request, .sink = sink, .queued = h.now() };
        h.lock();
        defer h.unlock();
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        if (h.queued.items.len >= h.queue_max) {
            h.gpa.destroy(job);
            return error.Busy;
        }
        // foreground before background, each in arrival order
        // foreground を background より先に、それぞれ到着順に並べる
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: api.Id) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        for (h.queued.items, 0..) |job, i| if (job.id == id) {
            _ = h.queued.orderedRemove(i);
            h.unlock();
            h.finish(job, .cancelled, .{}, "");
            h.lock();
            return;
        };
        // TF_GLM53_MULTI: a cancel for a request no stage holds (it ended, or never came) leaves nothing behind
        // TF_GLM53_MULTI: どの段階にも無い要求 (終わった、または来ていない) への取り消しは何も残さない
        if (h.multi_n > 0 and !h.holds(id)) return;
        h.cancels.append(h.gpa, id) catch {};
        // the running request: ask every rank to stop at its next agreement point (a prompt chunk, a decode round)
        // 実行中の要求: 次の合意点 (prompt チャンク、decode ラウンド) で止まるよう全ランクに求める
        if (h.running) |job| if (job.id == id) h.eng.stop_req.store(true, .monotonic);
        if (h.prefilling) |job| if (job.id == id) h.stopJob(job);
        for (h.act[0..h.multi_n]) |a| if (a) |job| if (job.id == id) h.stopJob(job);
    }

    /// TF_GLM53_MULTI: whether the scheduler holds the request `id` past the queue (between the queue and its prefill,
    /// in its prefill, or live). Under the lock.
    /// TF_GLM53_MULTI: 要求 `id` をスケジューラが待ち行列より先で持っているか (待ち行列と prefill の間、prefill 中、
    /// 使用中)。ロックの下で。
    fn holds(h: *const Host, id: api.Id) bool {
        if (h.admitting) |job| if (job.id == id) return true;
        if (h.prefilling) |job| if (job.id == id) return true;
        for (h.act[0..h.multi_n]) |a| if (a) |job| if (job.id == id) return true;
        return false;
    }

    /// Every cancel recorded for `id` dropped (a request ends once; a cancel may come twice). Under the lock.
    /// `id` に記録した取り消しをすべて外す (要求は 1 回で終わるが、取り消しは 2 回来うる)。ロックの下で。
    fn dropCancels(h: *Host, id: api.Id) void {
        var i: usize = 0;
        while (i < h.cancels.items.len) {
            if (h.cancels.items[i] == id) _ = h.cancels.swapRemove(i) else i += 1;
        }
    }

    /// Under the lock: whether a cancel came for a job not yet decoding (its stop bit, or a cancel recorded for its
    /// id), its records then dropped. Before its decode only a cancel (or the close) sets the stop bit.
    /// ロックの下で: まだ decode していない job に取り消しが来たか (停止ビット、またはその id の記録)。来ていれば記録を
    /// 外す。decode の前に停止ビットを立てるのは取り消し (か終了) だけ。
    fn takeCancel(h: *Host, job: *Job) bool {
        const c = job.stop.load(.monotonic) or std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null;
        if (c) h.dropCancels(job.id);
        return c;
    }

    /// A job's stop: its conversation's bit in the next batched round, and the prefill's stop while its prompt runs
    /// (the agreement points of every prompt chunk read the engine's stop_req).
    /// job の停止: 次のまとめたラウンドでのその会話のビット。prompt の実行中は prefill の停止も (各 prompt チャンクの
    /// 合意点がエンジンの stop_req を読む)。
    fn stopJob(h: *Host, job: *Job) void {
        job.stop.store(true, .monotonic);
        if (h.multi_n == 0 or h.prefilling == job) h.eng.stop_req.store(true, .monotonic);
    }

    fn statusFn(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        const t = h.now();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= t - window_ns) tokens += m.tokens;
        }
        var n: usize = 0;
        var generation_tokens: u64 = 0;
        if (h.running != null and stream_tokens.len > 0) {
            stream_tokens[0] = @intCast(h.live_prompt + h.live_generated);
            generation_tokens = h.live_generated;
            n = 1;
        } else if (h.multi_n > 0 and stream_tokens.len > 0) {
            // TF_GLM53_MULTI: the conversation being admitted and the live ones, each a stream
            // TF_GLM53_MULTI: 受け付け中の会話と使用中の会話、それぞれが 1 本の stream
            if (h.prefilling) |job| {
                stream_tokens[n] = @intCast(job.prompt0().len + job.generated);
                generation_tokens += job.generated;
                n += 1;
            }
            for (h.act[0..h.multi_n]) |a| if (a) |job| if (n < stream_tokens.len) {
                stream_tokens[n] = @intCast(job.prompt0().len + job.generated);
                generation_tokens += job.generated;
                n += 1;
            };
        }
        out.* = .{
            .pool = if (h.eng.pool) |p| blk: {
                const s = p.status();
                break :blk .{
                    .blocks = s.blocks, .free_blocks = s.free_blocks, .slots = s.slots, .shared_blocks = s.shared_blocks, .evicted_total = s.evicted_total, .cold_total = s.cold_total,
                    .block_positions = s.block_positions, .continued_total = s.continued_total, .shared_total = s.shared_total, .new_total = s.new_total,
                    .check_seconds_total = s.check_seconds_total, .hash_seconds_total = s.hash_seconds_total,
                };
            } else null,
            .running = @intCast(@as(usize, @intFromBool(h.running != null)) + @intFromBool(h.prefilling != null) + blk: {
                var a: usize = 0;
                for (h.act[0..h.multi_n]) |x| a += @intFromBool(x != null);
                break :blk a;
            }),
            .waiting = @intCast(h.queued.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (t - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = generation_tokens,
            .progress = h.progress(t),
        };
    }

    /// The totals with the running requests' progress in them, and each one's position (progress.requests, the
    /// oldest admitted first; the single fields tell the first of them). Under the mutex.
    /// 実行中の要求の進み具合を足した合計と、それぞれの位置 (progress.requests、受け付けの古い順。単独の項目は
    /// その最初のもの)。mutex の下で。
    fn progress(h: *Host, t: i96) api.Progress {
        var p = h.totals;
        p.phase = .idle;
        p.prompt = 0;
        p.cached = 0;
        p.prompt_done = 0;
        p.generated = 0;
        p.queued_prompt = 0;
        p.n_requests = 0;
        for (h.queued.items) |q| p.queued_prompt += q.prompt0().len;
        // the queue: how many wait, the oldest's wait; TF_GLM53_MULTI's settings as this engine runs them
        // 待ち行列: 待つ数と、最も古いものの待ち時間。このエンジンが動かしている TF_GLM53_MULTI の設定
        p.waiting = h.queued.items.len;
        p.waiting_ms = 0;
        for (h.queued.items) |q| p.waiting_ms = @max(p.waiting_ms, @as(u64, @intCast(@divTrunc(@max(0, t - q.queued), std.time.ns_per_ms))));
        p.multi = @intCast(h.multi_n);
        p.multi_wait_s = if (h.multi_n > 0) @intCast(@divTrunc(h.room_wait_ns, std.time.ns_per_s)) else 0;
        p.multi_schedule = h.scheduleNow();
        // where the RoCE proxies poll (TF_GLM53_ROCE_PROXY_CORES): rank 0's from /proc now, the others' from the start
        // RoCE の proxy が待つ所 (TF_GLM53_ROCE_PROXY_CORES): rank 0 のものは今 /proc から、ほかは起動時から
        p.engine_cores = glm.roce.engineCores(h.io);
        if (h.eng.roceView(h.io)) |v| {
            p.roce.mode = switch (v.mode) {
                .a725 => .a725,
                .x925 => .x925,
                .os => .os,
            };
            p.roce.world = @intCast(v.ranks);
            for (0..v.ranks) |r| for (0..2) |j| {
                p.roce.proxies[r][j] = .{ .present = v.present[r][j], .cpu = v.cpu[r][j], .live = v.live[r][j], .core = switch (v.kind[r][j]) {
                    .unknown => .unknown,
                    .a725 => .a725,
                    .x925 => .x925,
                } };
            };
        }
        if (h.running) |job| {
            h.addRequest(&p, job, h.live_generated, t);
        } else if (h.multi_n > 0) {
            // TF_GLM53_MULTI: the conversation being admitted and the live ones (at most multi_max: a prefill runs
            // only beside fewer live ones), the oldest admitted first
            // TF_GLM53_MULTI: 受け付け中の会話と使用中の会話 (多くて multi_max: prefill は使用中がそれより少ないときだけ)、
            // 受け付けの古い順
            var jobs: [api.max_requests]*Job = undefined;
            var n: usize = 0;
            if (h.prefilling) |job| {
                jobs[n] = job;
                n += 1;
            }
            for (h.act[0..h.multi_n]) |a| if (a) |job| if (n < jobs.len) {
                jobs[n] = job;
                n += 1;
            };
            std.mem.sort(*Job, jobs[0..n], {}, struct {
                fn older(_: void, x: *Job, y: *Job) bool {
                    return x.began < y.began;
                }
            }.older);
            for (jobs[0..n]) |job| h.addRequest(&p, job, job.generated, t);
        }
        if (p.n_requests > 0) {
            const q = p.requests[0];
            p.phase = q.phase;
            p.prompt = q.prompt;
            p.cached = q.cached;
            p.prompt_done = q.prompt_done;
            p.generated = q.generated;
        }
        return p;
    }

    /// A running request's position into `p.requests`, and its progress into the totals once its prefill started
    /// (`generated`: its reply tokens so far). Under the mutex.
    /// 実行中の要求の位置を `p.requests` へ、prefill が始まっていれば進み具合を合計へ (`generated`: ここまでの応答の
    /// トークン数)。mutex の下で。
    fn addRequest(h: *Host, p: *api.Progress, job: *const Job, generated: u64, t: i96) void {
        var q: api.RequestProgress = .{
            .conv = job.conv, .pool_id = job.pool_id, .phase = if (job.prefilled == null) .prefill else .decode,
            .prompt = job.prompt0().len, .elapsed_ms = @intCast(@divTrunc(@max(0, t - job.began), std.time.ns_per_ms)),
        };
        // still planning: nothing counted yet / まだ計画中: 何も数えない
        if (job.prefill_at) |start| {
            q.cached = job.cached;
            q.prompt_done = job.done;
            q.generated = generated;
            q.prefill_ms = @intCast(@divTrunc(@max(0, job.prefill_ns), std.time.ns_per_ms));
            h.addRunning(p, job, start, generated, t);
        }
        if (p.n_requests < api.max_requests) {
            p.requests[p.n_requests] = q;
            p.n_requests += 1;
        }
    }

    /// A running request's progress at time `t` added to the totals in `p` (`generated`: its reply tokens so far).
    /// 時刻 `t` での実行中の要求の進み具合を `p` の合計に足す (`generated`: ここまでの応答のトークン数)。
    fn addRunning(h: *Host, p: *api.Progress, job: *const Job, start: i96, generated: u64, t: i96) void {
        _ = h;
        p.prompt_total += job.prompt0().len;
        p.cached_total += job.cached;
        p.computed_total += job.done -| job.cached;
        p.generated_total += generated;
        p.prefill_seconds_total += secs(job.done_at - start);
        if (job.prefilled) |f| p.decode_seconds_total += secs(t - f);
        p.rounds_total += job.rounds;
        p.drafted_total += job.drafted;
        p.accepted_total += job.accepted;
    }

    /// A finished TF_GLM53_MULTI job's progress added to the totals (as addRunning, with its own tokens).
    /// 終わった TF_GLM53_MULTI の job の進み具合を合計に足す (addRunning と同じ、トークンはその job のもの)。
    fn addFinished(h: *Host, p: *api.Progress, job: *const Job, start: i96, t: i96) void {
        _ = h;
        p.prompt_total += job.prompt0().len;
        p.cached_total += job.cached;
        p.computed_total += job.done -| job.cached;
        p.generated_total += job.emitted.items.len;
        p.prefill_seconds_total += secs(job.done_at - start);
        if (job.prefilled) |f| p.decode_seconds_total += secs(t - f);
        p.rounds_total += job.rounds;
        p.drafted_total += job.drafted;
        p.accepted_total += job.accepted;
    }

    fn secs(ns: i96) f64 {
        return @max(0, @as(f64, @floatFromInt(ns)) / 1e9);
    }

    /// TF_GLM53_EXPERT_LIVE: the counts the engine thread keeps on rank 0 (its own mutex; no GPU).
    /// TF_GLM53_EXPERT_LIVE: エンジンのスレッドが rank 0 に残す数 (専用の mutex。GPU には触れない)。
    fn expertsFn(ctx: *anyopaque, a: Allocator, q: api.ExpertsQuery) ?[]u8 {
        const h: *Host = @ptrCast(@alignCast(ctx));
        if (q == .memory) return if (h.mem) |m| m.json(a) catch null else null;
        if (q == .swap) return h.swapJson(a) catch null;
        // the pool's snapshot (pool/live.zig, its own mutex) / pool の写し (pool/live.zig、専用の mutex)
        if (q == .pool) return if (h.eng.pool) |p| (if (p.live) |l| l.json(a) catch null else null) else null;
        const lv = h.eng.live orelse return null;
        return switch (q) {
            .usage => |w| lv.usageJson(a, w) catch null,
            .recent => |r| lv.recentJson(a, r.after, r.max, if (r.text) |tx| .{ .ctx = tx.ctx, .bytes = tx.bytes } else null) catch null,
            .prefill => |r| lv.prefillJson(a, r.after, r.max) catch null,
            .memory, .swap, .pool => null,
        };
    }

    /// GET /v1/experts/swap: whether a change can be asked for, the banks, and the last change's state.
    /// GET /v1/experts/swap: 変更を求められるか、バンク、最後の変更の状態。
    fn swapJson(h: *Host, a: Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(a);
        const w = &out.writer;
        h.lock();
        defer h.unlock();
        const s = h.swap;
        try w.print("{{\"available\": {}, \"plan\": ", .{h.eng.banks != null});
        try planJson(h, w);
        try w.writeAll(", \"banks\": {");
        var first = true;
        if (h.eng.banks) |b| for (b.dirs, 0..) |d, bits| if (d) |dir| {
            try w.print("{s}\"{d}\": ", .{ if (first) "" else ", ", bits });
            try std.json.Stringify.value(dir, .{}, w);
            first = false;
        };
        try w.print("}}, \"seq\": {d}, \"state\": \"{s}\", \"done\": {d}, \"layers\": {d}, \"experts\": {d}, \"up\": {d}, \"down\": {d}, \"read_bytes\": {d}, \"ms\": {d}, \"error\": \"{s}\"}}", .{ s.seq, @tagName(s.state), s.done, s.layers, s.stats.experts, s.stats.up, s.stats.down, s.stats.read_bytes, s.stats.ms, s.err });
        return out.toOwnedSlice();
    }

    /// The expert plan the routed experts were loaded by (null without one): name, digest, banks, counts by bits.
    /// routed expert を読み込んだ計画 (無ければ null): 名前、ダイジェスト、バンク、bit ごとの数。
    fn planJson(h: *Host, w: *std.Io.Writer) !void {
        const p = h.eng.plan orelse return w.writeAll("null");
        try w.writeAll("{\"name\": ");
        try std.json.Stringify.value(p.name, .{}, w);
        try w.print(", \"digest\": \"{x}\", \"banks\": [", .{p.digest});
        for (p.banks, 0..) |b, i| {
            try w.print("{s}{{\"id\": ", .{if (i == 0) "" else ", "});
            try std.json.Stringify.value(b.id, .{}, w);
            try w.print(", \"bits\": {d}, \"calibration\": ", .{b.bits});
            try std.json.Stringify.value(b.calibration, .{}, w);
            try w.writeAll("}");
        }
        const c = glm.expert_plan.counts(&p);
        try w.writeAll("], \"counts\": {");
        var first = true;
        for (c, 0..) |n, b| if (n > 0) {
            try w.print("{s}\"{d}\": {d}", .{ if (first) "" else ", ", b, n });
            first = false;
        };
        try w.writeAll("}}");
    }

    fn scheduleNow(h: *Host) api.Schedule {
        const m = h.pace.shown.load(.monotonic);
        return .{ .decode_share = h.share.load(.monotonic), .short_prompt = h.short.load(.monotonic), .short_preempt = h.preempt.load(.monotonic), .measured = if (m <= 100) m else null, .contended = h.pace.both.load(.monotonic), .start = h.sched_start };
    }

    /// GET / POST /v1/schedule (TF_GLM53_MULTI, rank 0): the schedule in use, the fields given set first (both checked
    /// before either is set). A new share holds from the scheduler's next turn and a running prefill's next pause or
    /// chunk (glm.pace); a new short prompt limit from the scheduler's next turn (a prefill already running keeps how it
    /// was admitted). Nothing waits for the request in flight.
    /// GET / POST /v1/schedule (TF_GLM53_MULTI、rank 0): 使用中のスケジュール。渡された項目を先に設定する (両方を確かめて
    /// から設定する)。新しい取り分は、スケジューラの次の番と、実行中の prefill の次に止まる所かチャンクから効く
    /// (glm.pace)。新しい短い prompt の上限はスケジューラの次の番から (実行中の prefill は受け入れたときのまま)。実行中の
    /// 要求の終わりは待たない。
    fn scheduleFn(ctx: *anyopaque, set: api.ScheduleSet) api.ControlError!api.Schedule {
        const h: *Host = @ptrCast(@alignCast(ctx));
        if (h.group.rank != 0 or h.multi_n == 0) return error.Unsupported;
        if (set.decode_share) |v| if (v > 100) return error.BadRequest;
        if (set.short_prompt) |v| if (v > api.short_prompt_max) return error.BadRequest;
        if (set.decode_share) |v| {
            if (h.share.swap(v, .monotonic) != v) std.log.info("glm53: schedule: the decode's share {d}% while a prompt waits or is computed", .{v});
        }
        if (set.short_prompt) |v| {
            if (h.short.swap(v, .monotonic) != v) logShort(v);
        }
        if (set.short_preempt) |v| {
            if (h.preempt.swap(v, .monotonic) != v) logPreempt(v);
        }
        return h.scheduleNow();
    }

    fn logPreempt(v: bool) void {
        if (v) return std.log.info("glm53: schedule: a longer prefill running gives way to a short prompt waiting", .{});
        std.log.info("glm53: schedule: a prefill running does not give way (a short prompt waits for its end)", .{});
    }

    fn logShort(v: u32) void {
        if (v == 0) return std.log.info("glm53: schedule: no prompt is short (every prompt waits its turn)", .{});
        std.log.info("glm53: schedule: a prompt with at most {d} rows to compute past the cache is short (it goes first, not held by the share)", .{v});
    }

    /// POST /v1/experts/swap: taken only while nothing runs or waits (the engine thread runs it next), with
    /// TF_GLM53_MULTI too: no conversation live or being admitted;
    /// POST /v1/experts/save: the bits in use, as a named set (TF_GLM53_EXPERT_SETS/<name>.json).
    /// POST /v1/experts/swap: 何も実行・待機していないときだけ受ける (次にエンジンのスレッドが実行)。
    /// TF_GLM53_MULTI でも同じく、使用中の会話も受け入れ中の会話もないとき。
    /// POST /v1/experts/save: 使用中の bit を名前つきのセットとして (TF_GLM53_EXPERT_SETS/<name>.json)。
    fn controlFn(ctx: *anyopaque, a: Allocator, c: api.ExpertsControl) api.ControlError![]u8 {
        const h: *Host = @ptrCast(@alignCast(ctx));
        if (h.group.rank != 0) return error.Unsupported;
        switch (c) {
            .swap => |bits| {
                if (h.eng.banks == null) return error.Unsupported;
                if (bits.len != h.eng.xbits.len) return error.BadRequest;
                for (bits) |b| if (b < 2 or b > 8) return error.BadRequest;
                const want = try h.gpa.dupe(u8, bits);
                h.lock();
                defer h.unlock();
                if (h.running != null or h.admitting != null or h.prefilling != null or h.liveCount() > 0 or h.queued.items.len > 0 or h.swap_want != null or h.swap.state == .running or h.closing) {
                    h.gpa.free(want);
                    return error.Busy;
                }
                h.swap_want = want;
                h.swap = .{ .seq = h.swap.seq + 1, .state = .waiting };
                h.wake.signal(h.io);
                return std.fmt.allocPrint(a, "{{\"seq\": {d}, \"state\": \"waiting\"}}", .{h.swap.seq});
            },
            .save => |name| return h.saveSet(a, name),
        }
    }

    fn saveSet(h: *Host, a: Allocator, name: []const u8) api.ControlError![]u8 {
        const dir = std.mem.span(std.c.getenv("TF_GLM53_EXPERT_SETS") orelse return error.Unsupported);
        if (name.len == 0 or name.len > 64) return error.BadRequest;
        for (name) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '-' and ch != '_' and ch != '.') return error.BadRequest;
        if (name[0] == '.') return error.BadRequest;
        const path = try std.fmt.allocPrint(a, "{s}/{s}.json", .{ dir, name });
        const cwd = std.Io.Dir.cwd();
        if (cwd.openFile(h.io, path, .{})) |f| {
            f.close(h.io);
            return error.Exists;
        } else |_| {}
        h.lock();
        // the bits change only in a swap, which the engine thread runs with the state at .running
        // bit が変わるのは swap の中だけ。エンジンのスレッドはその間 state を .running にしている
        if (h.swap.state == .running or h.swap_want != null) {
            h.unlock();
            return error.Busy;
        }
        const bits = a.dupe(u8, h.eng.xbits) catch |err| {
            h.unlock();
            return err;
        };
        h.unlock();
        const body = setJson(h, a, name, bits) catch return error.OutOfMemory;
        cwd.createDirPath(h.io, dir) catch return error.Failed;
        // written whole to a file of its own, then linked to the name, which fails if the name exists: two saves of
        // one name never both win, and no reader sees half a file
        // 自分だけのファイルに全部書いてから名前へ link する (名前があれば失敗): 同じ名前の 2 つの保存が両方勝つことは
        // なく、読み手が書きかけを見ることもない
        h.lock();
        h.save_n += 1;
        const n = h.save_n;
        h.unlock();
        const tmpz = std.fmt.allocPrintSentinel(a, "{s}/.{s}.json.tmp{d}", .{ dir, name, n }, 0) catch return error.OutOfMemory;
        const pathz = a.dupeSentinel(u8, path, 0) catch return error.OutOfMemory;
        cwd.writeFile(h.io, .{ .sub_path = tmpz, .data = body }) catch return error.Failed;
        defer _ = std.c.unlink(tmpz.ptr);
        if (std.c.link(tmpz.ptr, pathz.ptr) != 0) return if (std.c._errno().* == @intFromEnum(std.c.E.EXIST)) error.Exists else error.Failed;
        std.log.info("glm53: expert bits saved as {s}", .{path});
        return std.fmt.allocPrint(a, "{{\"saved\": \"{s}\"}}", .{name});
    }

    /// glm53-expert-bits v1: the bits (base64, one byte an expert, the MoE layers from first_layer), their counts, the
    /// checkpoint they were loaded from and the banks.
    /// glm53-expert-bits v1: bit (base64、expert ごとに 1 バイト、first_layer からの MoE 層)、その数、読み込んだ
    /// チェックポイントとバンク。
    fn setJson(h: *Host, a: Allocator, name: []const u8, bits: []const u8) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(a);
        const w = &out.writer;
        const E = h.eng.cfg.experts;
        var counts: [9]u64 = @splat(0);
        var sum: u64 = 0;
        for (bits) |b| {
            counts[@min(b, 8)] += 1;
            sum += b;
        }
        const t = std.Io.Clock.real.now(h.io).toSeconds();
        try w.print("{{\n  \"format\": \"glm53-expert-bits\",\n  \"version\": 1,\n  \"name\": \"{s}\",\n  \"saved_unix\": {d},\n  \"checkpoint\": ", .{ name, t });
        try std.json.Stringify.value(h.ckpt, .{}, w);
        try w.print(",\n  \"first_layer\": {d},\n  \"layers\": {d},\n  \"experts\": {d},\n  \"mean_bits\": {d:.4},\n  \"counts\": {{", .{ h.eng.xfirst, bits.len / E, E, @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(@max(bits.len, 1))) });
        var first = true;
        for (counts, 0..) |n, b| if (n > 0) {
            try w.print("{s}\"{d}\": {d}", .{ if (first) "" else ", ", b, n });
            first = false;
        };
        try w.writeAll("},\n  \"banks\": {");
        first = true;
        if (h.eng.banks) |bk| for (bk.dirs, 0..) |d, b| if (d) |dir| {
            try w.print("{s}\"{d}\": ", .{ if (first) "" else ", ", b });
            try std.json.Stringify.value(dir, .{}, w);
            first = false;
        };
        try w.writeAll("},\n  \"bits\": \"");
        const enc = std.base64.standard.Encoder;
        const b64 = try a.alloc(u8, enc.calcSize(bits.len));
        _ = enc.encode(b64, bits);
        try w.writeAll(b64);
        try w.writeAll("\"\n}\n");
        return out.toOwnedSlice();
    }

    /// Engine thread: the change asked for, on every rank (lockstep.swap), its progress and outcome in `swap`.
    /// エンジンのスレッド: 求められた変更を全ランクで (lockstep.swap)。進み具合と結果は `swap` に。
    fn runSwap(h: *Host, want: []u8) void {
        defer h.gpa.free(want);
        const Prog = struct {
            fn f(ctx: *anyopaque, done: u32, layers: u32) void {
                const hh: *Host = @ptrCast(@alignCast(ctx));
                hh.lock();
                defer hh.unlock();
                hh.swap.done = done;
                hh.swap.layers = layers;
            }
        };
        std.log.info("glm53: expert swap {d}: start", .{h.swap.seq});
        var st: glm.expert_swap.Stats = .{};
        const prog: glm.expert_swap.Progress = .{ .ctx = h, .f = Prog.f };
        const r = if (h.multi_n > 0) glm.lockstep.swapMulti(&h.ch, h.eng, want, prog, &st) else glm.lockstep.swap(&h.ch, h.eng, want, prog, &st);
        h.lock();
        h.swap.stats = st;
        if (r) |_| {
            h.swap.state = .done;
        } else |err| {
            h.swap.state = .failed;
            h.swap.err = @errorName(err);
        }
        const s = h.swap;
        h.unlock();
        // a swap that may have left the ranks apart (or the cached prefixes stale): every rank stops, as after a
        // failed request (the followers return the error from follow and stop too)
        // ランクが食い違ったかもしれない (または保存した prefix が古い) 入れ替え: 失敗した要求の後と同じく全ランクが
        // 止まる (フォロワーも follow からエラーを返して止まる)
        if (r) |_| {} else |err| if (!st.alike) fatal("an expert swap that may have left the ranks apart", err);
        if (r) |_| {
            std.log.info("glm53: expert swap {d}: {d} experts in {d} layers ({d} up, {d} down), {d} MiB read, {d} ms", .{ s.seq, s.stats.experts, s.stats.layers, s.stats.up, s.stats.down, s.stats.read_bytes >> 20, s.stats.ms });
        } else |err| std.log.err("glm53: expert swap {d}: {s}", .{ s.seq, @errorName(err) });
        if (h.mem) |m| m.setExperts(expertBytes(h.eng));
    }

    /// rank 0, engine thread: the memory snapshot (tracked bytes and the driver's free / total).
    /// rank 0、エンジンのスレッド: メモリの写し (数えているバイトとドライバの free / total)。
    fn memSnap(h: *Host) void {
        const m = h.mem orelse return;
        const u = cuda.usage(false);
        const mi = h.ctx.memInfo() catch return;
        m.snap(u.device, u.host, u.peak, mi.free, mi.total);
    }

    /// The process's tracked device bytes (DeviceBuffers) and their peak; no GPU call (/health, /metrics).
    /// プロセスが数えている device のバイト (DeviceBuffer) とそのピーク。GPU は呼ばない (/health、/metrics)。
    fn memoryFn(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .peak = u.peak };
    }

    fn emit(job: *Job, event: api.Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A request refused or failed before its prompt was in the cache: the server hears "prefilled" once a request, so
    /// a continuation (segs > 0; its first segment told it) says nothing here.
    /// prompt がキャッシュに入る前に断った・失敗した要求: サーバーが聞く "prefilled" は要求に 1 回なので、続き (segs > 0。
    /// 最初の区間が伝えた) はここで何も言わない。
    fn noPrefill(job: *Job) void {
        if (job.segs == 0) emit(job, .{ .prefilled = 0 });
    }

    fn finish(h: *Host, job: *Job, reason: api.Reason, stats: api.Stats, message: []const u8) void {
        h.lock();
        // refused for want of room within TF_GLM53_MULTI_WAIT_S (/metrics busy_total)
        // TF_GLM53_MULTI_WAIT_S のうちに空きがなく断った (/metrics の busy_total)
        if (reason == .busy) h.totals.busy_total += 1;
        // the request's progress joins the totals as it leaves `running`, so the totals status reads never fall back
        // 要求の進み具合は `running` から外すときに合計へ移す。status が読む合計は後戻りしない
        if (h.running == job) {
            if (job.prefill_at) |start| {
                h.addRunning(&h.totals, job, start, h.live_generated, h.now());
                h.totals.requests_total += 1;
            }
        }
        // (TF_GLM53_MULTI: a live conversation's progress joined the totals as releaseConv took it off the live ones)
        // (TF_GLM53_MULTI: 使用中の会話の進み具合は、releaseConv が使用中から外したときに合計へ移した)
        // before finished is out: status must not see a freed job
        // finished を出す前に外す: status が解放済みの job を見てはならない
        if (h.running == job) h.running = null;
        if (h.admitting == job) h.admitting = null;
        // its cancels are spent: a request ends once / その取り消しは使い終わった: 要求は 1 回で終わる
        h.dropCancels(job.id);
        h.unlock();
        // a reply the engine edited (a fold, a notice, the budget's cut, remade ids): the server tells the client the
        // reply as the context held it, to send back next turn
        // エンジンが手を入れた応答 (畳み込み、お知らせ、予算の打ち切り、作り直した id): 文脈が持っていた形の応答をサーバーが
        // クライアントに伝え、次のターンで送り返してもらう
        var st = stats;
        if (job.segs > 0) st.context_reply = job.live.items;
        emit(job, .{ .finished = .{ .reason = reason, .stats = st, .message = message } });
        job.emitted.deinit(h.gpa);
        job.live.deinit(h.gpa);
        job.br_text.deinit(h.gpa);
        if (job.seg_prompt) |p| h.gpa.free(p);
        if (job.matched) |p| h.gpa.free(p);
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *Host, n: usize) void {
        const t = h.now();
        h.lock();
        defer h.unlock();
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < t - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = t, .tokens = n }) catch {};
    }

    fn run(h: *Host) void {
        h.ctx.makeCurrent() catch |err| fatal("binding the CUDA context", err);
        while (true) {
            h.lock();
            while (h.queued.items.len == 0 and h.swap_want == null and !h.closing) {
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            }
            if (h.swap_want) |want| if (!h.closing) {
                h.swap_want = null;
                h.swap.state = .running;
                h.unlock();
                h.runSwap(want);
                h.memSnap();
                continue;
            };
            if (h.closing) {
                if (h.swap_want) |want| h.gpa.free(want);
                h.swap_want = null;
                const left = h.gpa.dupe(*Job, h.queued.items) catch &.{};
                h.queued.clearRetainingCapacity();
                h.unlock();
                for (left) |job| h.finish(job, .cancelled, .{}, "");
                h.gpa.free(left);
                glm.lockstep.shutdown(&h.ch) catch |err| std.log.err("glm53: telling the other ranks to stop failed: {s}", .{@errorName(err)});
                return;
            }
            const job = h.queued.orderedRemove(0);
            h.running = job;
            // with `running`, under the lock: a cancel from now on is kept
            // `running` と一緒にロックの下でリセットする: これ以降の cancel は失われない
            h.eng.stop_req.store(false, .monotonic);
            h.live_prompt = job.prompt0().len;
            h.live_generated = job.emitted.items.len; // a continuation's reply so far / 続きならここまでの応答
            h.cancels.clearRetainingCapacity();
            h.unlock();
            h.serve(job);
            h.memSnap();
        }
    }

    /// The reply's callbacks from Engine.generate (engine thread).
    /// Engine.generate から呼ばれる応答のコールバック (エンジンのスレッド)。
    const Out = struct {
        h: *Host,
        job: *Job,
        // TF_GLM53_MULTI: its conversation (null: the request alone) / TF_GLM53_MULTI: その会話 (null なら単独の要求)
        k: ?u8 = null,

        // the conversation's pool slot, once its plan is made (the engine thread's, as this callback)
        // 会話の pool のスロット。計画ができてから (このコールバックと同じくエンジンのスレッドのもの)
        fn notePool(c: *Out) void {
            const k: ?usize = if (c.k) |k| @as(usize, k) else null;
            if (c.job.pool_id == null) c.job.pool_id = c.h.eng.poolIdOf(k);
            if (c.h.eng.poolSlotOf(k)) |x| c.job.pool_x = x;
        }

        fn prefilled(ctx: *anyopaque) void {
            const c: *Out = @ptrCast(@alignCast(ctx));
            const h = c.h;
            // a continuation (after a notice or the cut): the request's prefill and progress were its first segment's
            // 続き (お知らせや打ち切りのあと): 要求の prefill と進み具合は最初の区間のもの
            if (c.job.segs > 0) {
                h.lock();
                c.notePool();
                h.unlock();
                return;
            }
            const done = h.now();
            h.lock();
            // the slot cache's resume point (this engine thread set it for this prefill)
            // スロットキャッシュの再開位置 (このエンジンのスレッドがこの prefill のために設定した)
            const cached = h.eng.start;
            if (done > c.job.began) h.prefill_rate = @as(f64, @floatFromInt(c.job.prompt0().len - cached)) / (@as(f64, @floatFromInt(done - c.job.began)) / 1e9);
            h.prefill_at = done;
            c.job.prefilled = done;
            c.notePool();
            if (c.job.prefill_at == null) {
                c.job.prefill_at = c.job.began;
                c.job.cached = cached;
            }
            c.job.done = c.job.prompt0().len;
            c.job.done_at = done;
            c.noteCompute(done);
            h.unlock();
            emit(c.job, .{ .prefilled = @intCast(cached) });
        }

        fn chunk(ctx: *anyopaque, done: usize) void {
            const c: *Out = @ptrCast(@alignCast(ctx));
            const h = c.h;
            const t = h.now();
            h.lock();
            defer h.unlock();
            if (c.job.segs > 0) return c.notePool(); // a continuation's (as prefilled) / 続きのもの (prefilled と同じ)
            // the first chunk: the prefill starts at the rows the cache resumed
            // 最初のチャンク: prefill はキャッシュから再開した行から始まる
            if (c.job.prefill_at == null) {
                c.job.prefill_at = t;
                c.job.cached = done;
            }
            c.job.done = done;
            c.job.done_at = t;
            c.noteCompute(t);
            c.notePool();
        }

        /// The prefill's own time up to `t` (under the host's lock, with job.done): from this admission's first chunk,
        /// less its pauses for the others, plus the admissions before it gave way.
        /// `t` までの prefill 自体の時間 (ホストのロックの下で job.done と一緒に): この受け入れの最初のチャンクから、ほかの
        /// ために止まっていた時間を引き、譲る前の受け入れの分を足す。
        fn noteCompute(c: *Out, t: i96) void {
            const job = c.job;
            const t0 = job.adm_t0 orelse blk: {
                job.adm_t0 = t;
                break :blk t;
            };
            job.prefill_ns = job.prefill_base_ns + @max(0, t - t0 - c.h.mid_paused_ns);
        }

        fn round(ctx: *anyopaque, drafted: usize, accepted: usize) void {
            const c: *Out = @ptrCast(@alignCast(ctx));
            const h = c.h;
            h.lock();
            defer h.unlock();
            c.job.rounds += 1;
            c.job.drafted += drafted;
            c.job.accepted += accepted;
        }

        fn tokens(ctx: *anyopaque, toks: []const u32) void {
            const c: *Out = @ptrCast(@alignCast(ctx));
            const job = c.job;
            const h = c.h;
            h.noteDecoded(toks.len);
            // a stop, or the decode stopping for a notice or the cut: the rows past it are not the reply's
            // 停止、またはお知らせや打ち切りのために止めている decode: その先の行は応答のものではない
            if (job.quiet or job.cut != null) return;
            h.lock();
            const gone = h.closing or std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null;
            h.unlock();
            if (gone) {
                job.quiet = true;
                h.stopJob(job);
                return;
            }
            if (job.br_writing) return c.branchTokens(toks);
            var n: usize = 0;
            // stop strings are checked after each token
            // stop 文字列はトークンごとに確かめる
            for (toks) |t| {
                // the reply's EOS with its think block open: the EOS stays out of the reply (the server ends a reply
                // at it) and the block closes, so the model answers (Request.think_unclosed)
                // 思考ブロックを開いたままの EOS: EOS は応答に入れない (サーバーはそこで応答を終える)。ブロックを閉じ、
                // モデルに答えを書かせる (Request.think_unclosed)
                if (std.mem.indexOfScalar(u32, job.request.eos, t) != null) if (job.unclosedCut()) |cut| {
                    job.cut = cut;
                    h.stopJob(job);
                    break;
                };
                // out of host memory: the reply is lost, so every rank stops
                // ホストのメモリ不足: 応答は失われるので全ランクを止める
                job.live.append(h.gpa, t) catch {
                    job.quiet = true;
                    h.stopJob(job);
                    break;
                };
                job.emitted.append(h.gpa, t) catch {
                    job.quiet = true;
                    h.stopJob(job);
                    break;
                };
                n += 1;
                if (job.request.stop) |s| if (s.check(s.ctx, job.emitted.items)) {
                    job.quiet = true;
                    // the reply is complete: the ranks need not go on
                    // 応答は完成した: ランクは続ける必要がない
                    h.stopJob(job);
                    break;
                };
                // a notice or the budget's cut at this line break: every rank stops, then the host writes it (goOn)
                // この改行でお知らせか予算の打ち切り: 全ランクを止め、そのあとホストが書き込む (goOn)
                if (job.thinkCut(t)) |cut| {
                    // the close waits for its summary: the close token leaves the reply and the context again
                    // 閉じるのは要約を待つ: 閉じるトークンを応答と文脈から取り除く
                    if (job.drop_close) {
                        job.drop_close = false;
                        _ = job.live.pop();
                        _ = job.emitted.pop();
                        n -= 1;
                    }
                    job.cut = cut;
                    h.stopJob(job);
                    break;
                }
                // the model closed the think block: its ids as the tokenizer cuts its text, or the answer waits for a
                // re-admission from them (TF_GLM53_CANON_CLOSE)
                // モデルが思考ブロックを閉じた: その id をトークナイザーが文を切るとおりにする。違えば、そこからの受け入れ
                // 直しまで答えを待たせる (TF_GLM53_CANON_CLOSE)
                if (job.closed_now) {
                    job.closed_now = false;
                    if (h.closeCut(job)) |cut| {
                        job.cut = cut;
                        h.stopJob(job);
                        break;
                    }
                }
            }
            if (n > 0) emit(job, .{ .tokens = toks[0..n] });
            h.lock();
            h.live_generated = @intCast(job.emitted.items.len);
            job.generated = job.emitted.items.len;
            h.unlock();
        }

        /// The summary beside the reply's tokens (think_fold branch): to the client as reasoning and into br_text,
        /// not into the context. The answer's EOS, another turn or a tool call ends it (the token left out); the
        /// answer may think first (11-02: the model wrote notes, </think>, then the summary):
        /// its <think> is dropped, a </think> marks where the answer begins (br_split, the last
        /// one), and think_fold_cap tokens of the answer, or twice that in all, end it at a line break (at most
        /// think_grace tokens later).
        /// 応答の横での要約のトークン (think_fold branch): 思考としてクライアントへ、br_text へ (文脈には入れない)。
        /// 答えの EOS、ほかの番、道具呼び出しで終わる (そのトークンは入れない)。答えは先に考えることがある (11-02:
        /// モデルがメモ、</think>、要約と書いた): <think> は捨て、</think> は
        /// 答えの始まりの印 (br_split、最後のもの)。答えの think_fold_cap 個、または全体でその 2 倍で、改行のところで
        /// 終える (遅くとも think_grace 個後)。
        fn branchTokens(c: *Out, toks: []const u32) void {
            const job = c.job;
            const h = c.h;
            const r = job.request;
            var shown: std.ArrayList(u32) = .empty;
            defer shown.deinit(h.gpa);
            for (toks) |t| {
                if (std.mem.indexOfScalar(u32, r.eos, t) != null or std.mem.indexOfScalar(u32, r.think_fold_branch_stops, t) != null) {
                    job.cut = .{ .fold = .branch_close };
                    h.stopJob(job);
                    break;
                }
                if (r.think_fold_branch_open) |o| if (t == o) continue;
                if (r.think_end) |e| if (t == e) {
                    job.br_split = job.br_text.items.len;
                    // take before: what precedes the close is the result, so the branch ends there
                    // take before: 閉じる前が結果なので、枝はそこで終わる
                    if (job.step().take == .before and job.br_split >= 16) {
                        job.cut = .{ .fold = .branch_close };
                        h.stopJob(job);
                        break;
                    }
                    continue;
                };
                job.br_text.append(h.gpa, t) catch {
                    job.quiet = true;
                    h.stopJob(job);
                    break;
                };
                job.emitted.append(h.gpa, t) catch {
                    job.quiet = true;
                    h.stopJob(job);
                    break;
                };
                shown.append(h.gpa, t) catch {};
                job.think_n += 1;
                const cap: u64 = job.br_cap;
                const answer: u64 = job.br_text.items.len - job.br_split;
                const all: u64 = job.br_text.items.len;
                if ((answer >= cap and (isBreak(r.think_breaks, t) or answer - cap >= think_grace)) or all >= 2 * cap + think_grace) {
                    job.cut = .{ .fold = .branch_close };
                    h.stopJob(job);
                    break;
                }
            }
            if (shown.items.len > 0) emit(job, .{ .tokens = shown.items });
            h.lock();
            h.live_generated = @intCast(job.emitted.items.len);
            job.generated = job.emitted.items.len;
            h.unlock();
        }
    };

    fn serve(h: *Host, job: *Job) void {
        h.byteResume(job);
        const r = job.request;
        const seg = job.promptNow();
        if (job.segs == 0) job.began = h.now();
        job.context = h.eng.capacity;
        if (seg.len == 0 or seg.len > h.eng.promptLimit()) {
            noPrefill(job);
            return h.finish(job, .failed, .{}, "the prompt does not fit this server's context window");
        }
        const prompt = h.gpa.alloc(i64, seg.len) catch {
            noPrefill(job);
            return h.finish(job, .failed, .{}, "out of host memory");
        };
        defer h.gpa.free(prompt);
        for (prompt, seg) |*p, t| p.* = t;
        var out: Out = .{ .h = h, .job = job };
        // the request's EOS switch: every rank stops at the model's EOS ids (the server's set also holds the
        // tokenizer's; for GLM-5.3 they agree)
        // 要求の EOS 指定: 全ランクがモデルの EOS id で止まる (サーバーの集合はトークナイザーのものも含むが、GLM-5.3
        // では一致する)
        // drafts when the request allows them ("draft": false: the serial rounds): the MTP layer's, or DFlash2's
        // 要求が許せば draft する ("draft": false なら逐次ラウンド): MTP 層のもの、なければ DFlash2 のもの
        const mode: u8 = if (!r.drafts) 0 else if (h.eng.mtp != null) glm.lockstep.mode_mtp else if (h.eng.drafter != null) glm.lockstep.mode_dflash else 0;
        var header: glm.lockstep.Header = .{ .max_tokens = @intCast(@min(job.maxNow(), std.math.maxInt(i32))), .stop_eos = r.eos.len > 0, .mode = mode };
        // a later segment of the same reply continues its own slot (Header.own_slot)
        // 同じ応答の 2 つ目以降の区間は自分のスロットを続ける (Header.own_slot)
        if (job.segs > 0) header.own_slot = job.pool_x;
        // the server resolved the model's defaults and the prompt's seed
        // モデルの既定値と prompt の seed はサーバーが決めてある
        if (r.sampling) |s| if (s.temperature > 0) {
            header.seed = s.seed;
            header.temperature = s.temperature;
            header.top_k = @intCast(s.top_k);
            header.top_p = s.top_p;
            header.min_p = s.min_p;
        };
        // from here every rank runs the request: a failure leaves the others waiting in a collective
        // ここから全ランクが要求を実行する: 失敗すると他のランクは collective 通信で待ったままになる
        const sink: glm.engine.Sink = .{ .ctx = @ptrCast(&out), .tokens = Out.tokens, .prefilled = Out.prefilled, .chunk = Out.chunk, .round = Out.round };
        // TF_GLM53_MULTI: the followers run events; this request is one of them (single)
        // TF_GLM53_MULTI: フォロワーは出来事を実行する。この要求はその 1 つ (single)
        const st = (if (h.multi_n > 0) glm.lockstep.leadSingle(&h.ch, h.eng, header, prompt, sink, null) else glm.lockstep.lead(&h.ch, h.eng, header, prompt, sink, null)) catch |err| {
            // refused before it was shared: every rank goes on serving
            // 共有の前に断った: 全ランクがそのまま処理を続ける
            if (err == error.ShapeNotVerified) {
                noPrefill(job);
                return h.finish(job, .failed, .{}, "this prompt needs a kernel shape the server did not verify at start-up (see the server log)");
            }
            fatal("serving a shared request", err);
        };
        // the summary beside the reply ended with its segment (max_tokens or the context left): closed as at its EOS
        // 応答の横での要約が区間とともに終わった (max_tokens か残りの文脈): EOS で終わった時と同じに閉じる
        if (job.br_writing and job.cut == null and !job.quiet) job.cut = .{ .fold = .branch_close };
        // stopped for a notice or the budget's cut: the reply goes on from the queue's head
        // お知らせか予算の打ち切りのために止めた: 応答は待ち行列の先頭から続く
        if (job.cut != null and h.goOn(job, st, null)) return;
        // the cancel list under the lock: a cancel arriving now may grow it
        // cancel の一覧はロックの下で読む: いま届く cancel が一覧を伸ばしうる
        const cancelled = if (job.quiet) blk: {
            h.lock();
            defer h.unlock();
            break :blk std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null;
        } else false;
        const reason: api.Reason = if (job.quiet) (if (cancelled) .cancelled else .stop) else switch (st.finish) {
            .stop => .stop,
            .length => .length,
            .cancelled => .cancelled,
        };
        h.lock();
        const computed = job.done -| job.cached;
        h.unlock();
        h.finish(job, reason, statsOf(job, st, computed), "");
    }

    /// A finished reply's stats over its segments (one, unless notices or the budget's cut stopped its decode): the
    /// first segment's prefill, every later one's decode and re-admission in the decode.
    /// 終わった応答の統計を区間すべてについて (お知らせや予算の打ち切りで decode を止めていなければ 1 つ): prefill は最初の
    /// 区間のもの、それ以降の区間の decode と受け入れ直しは decode に入れる。
    fn statsOf(job: *const Job, st: glm.engine.Stats, computed: u64) api.Stats {
        const rounds = job.acc_rounds + (if (st.rounds > 0) st.rounds else st.generated -| 1);
        const drafted = job.acc_drafted + st.drafted;
        const prefill_ns = if (job.segs > 0) job.acc_prefill_ns else st.prefill_ns;
        const decode_ns = job.acc_decode_ns + st.decode_ns + (if (job.segs > 0) st.prefill_ns else 0);
        return .{
            .rounds = rounds, .drafted = drafted, .accepted = job.acc_accepted + st.accepted, .min_rows = if (drafted > 0) 2 else 1,
            .prefill_seconds = @as(f64, @floatFromInt(prefill_ns)) / 1e9,
            .decode_seconds = if (st.generated > 0 or job.segs > 0) @as(f64, @floatFromInt(decode_ns)) / 1e9 else null,
            .computed = computed,
            .queue_seconds = secs(job.began - job.queued),
        };
    }

    /// The context's think ids remade as the tokenizer cuts their text (Request.think_canon), so the text a client sends
    /// back renders to the same ids; `live` changes only between the common prefix and suffix of the two. `from`: only
    /// when the first difference lies at or past that index (an insertion's own boundary), not before it (the model's
    /// own ids: the next fold or the block's end takes them). Returns where `live` first changed, or null (the same
    /// ids, a difference left before `from`, or the hook declined).
    /// 文脈の思考の id を、トークナイザーが文を切るとおりに作り直す (Request.think_canon): クライアントが送り返す文が同じ id
    /// になる。`live` が変わるのは両者の共通の先頭と末尾の間だけ。`from`: 最初の違いがその添字以降 (差し込みのつなぎ目)
    /// の時だけ直し、それより前 (モデル自身の id) は次の畳み込みかブロックの終わりに任せる。`live` が最初に変わった添字を
    /// 返す (同じ id、`from` より前の違い、フックが断った時は null)。
    fn canonThink(h: *Host, job: *Job, from: ?usize, what: []const u8) ?usize {
        const hook = job.request.think_canon orelse return null;
        const l = job.live.items;
        const end = if (job.request.think_end) |e| std.mem.indexOfScalar(u32, l, e) orelse l.len else l.len;
        if (end == 0) return null;
        var arena: std.heap.ArenaAllocator = .init(h.gpa);
        defer arena.deinit();
        const t0 = h.now();
        const canon = hook.run(hook.ctx, arena.allocator(), l[0..end]) orelse {
            std.log.info("glm53: thinking: ids kept ({s}): the think text does not round-trip ({d} ids)", .{ what, end });
            return null;
        };
        const ms = @as(f64, @floatFromInt(h.now() - t0)) / 1e6;
        const p = std.mem.indexOfDiff(u32, l[0..end], canon) orelse return null;
        if (from) |f| if (p < f) {
            std.log.info("glm53: thinking: ids kept ({s}): they differ at {d} of {d}, before the insertion at {d}; the next fold or the close remakes them ({d:.1} ms)", .{ what, p, end, f, ms });
            return null;
        };
        var s: usize = 0;
        while (s < end - p and s < canon.len - p and l[end - 1 - s] == canon[canon.len - 1 - s]) s += 1;
        job.live.replaceRange(h.gpa, p, end - s - p, canon[p .. canon.len - s]) catch return null;
        job.canons += 1;
        std.log.info("glm53: thinking: ids remade ({s}): {d} -> {d} think ids, the first difference at {d} ({d:.1} ms)", .{ what, end, canon.len, p, ms });
        return p;
    }

    /// The model may cut a text differently from the tokenizer: the prompt then holds the same bytes as a slot under other
    /// ids (pool/byte_resume.zig). Once for each reply, before its first admission (rank 0 tells the other ranks the prompt).
    /// log: a line with how far the bytes agree. on: also the prompt with the slot's ids to there, so the cache resumes
    /// past the place. The line holds ids only, no text.
    /// モデルはトークナイザーと違う切り方をすることがあり、その時 prompt はスロットと同じバイトを別の id で持つ
    /// (pool/byte_resume.zig)。応答ごとに 1 回、最初の受け入れの前に (prompt は rank 0 が他のランクに伝える)。log はバイトが
    /// どこまで一致するかの 1 行。on はそこまでの prompt をスロットの id にし、キャッシュはその先から再開する。行には id
    /// だけで文は入れない。
    fn byteResume(h: *Host, job: *Job) void {
        if (h.byte_resume == .off or job.byte_tried or job.segs > 0) return;
        job.byte_tried = true;
        const hook = job.request.token_vocab orelse return;
        var arena: std.heap.ArenaAllocator = .init(h.gpa);
        defer arena.deinit();
        const Vocab = struct {
            hook: api.TokenVocab,
            a: Allocator,
            pub fn bytes(v: @This(), id: u32) ?[]const u8 {
                return v.hook.bytes(v.hook.ctx, v.a, id);
            }
            pub fn added(v: @This(), id: u32) bool {
                return v.hook.added(v.hook.ctx, id);
            }
        };
        const t0 = h.now();
        const prompt = job.request.prompt;
        const b = glm.pool_serve.byteProbe(h.eng, prompt, Vocab{ .hook = hook, .a = arena.allocator() }) orelse return;
        const x = b.x;
        // nothing past the match on one side, or a few tokens only: no line
        // どちらかに一致の先がない、または数トークンだけの時は出さない
        if (x.p >= prompt.len or x.p + 8 > b.seq.len) return;
        const ms = @as(f64, @floatFromInt(h.now() - t0)) / 1e6;
        const w = struct {
            fn seq(s: []const i64, at: u64) []const i64 {
                const c: usize = @intCast(@min(at, s.len));
                return s[c -| 2..@min(c + 6, s.len)];
            }
            fn pr(s: []const u32, at: u64) []const u32 {
                const c: usize = @intCast(@min(at, s.len));
                return s[c -| 2..@min(c + 6, s.len)];
            }
        };
        // on, and the bytes went past the first id: the prompt takes the slot's ids, unless that no longer fits
        // on でバイトが最初の id より先に進んだ: prompt をスロットの id にする。収まらなくなるならしない
        const used: []const u8 = blk: {
            if (h.byte_resume != .on or x.q <= x.p) break :blk "";
            if (x.q + prompt.len - x.r > h.eng.promptLimit()) break :blk "; too long, kept";
            job.matched = glm.pool_serve.byteMerged(h.gpa, b, prompt) catch break :blk "; no memory, kept";
            break :blk "; used";
        };
        std.log.info("glm53: byteresume: prompt {d}, slot {d} holds {d}: ids stop at {d}, bytes reach {d} (+{d}, the prompt at {d}), {d} drifts, stop {s} ({d:.1} ms); first slot {any} prompt {any}; stop slot {any} prompt {any}{s}", .{
            prompt.len, b.slot, b.seq.len, x.p, x.q, x.q - x.p, x.r, x.drifts, @tagName(x.stop), ms,
            w.seq(b.seq, x.p), w.pr(prompt, x.p), w.seq(b.seq, x.q), w.pr(prompt, x.r), used,
        });
    }

    /// The think block the model just closed: its ids remade (canonThink) when they differ within the last
    /// `canon_near` (TF_GLM53_CANON_CLOSE=near, the default; all: anywhere; off: never), so the answer is computed on
    /// the ids the next turn renders; the cut that re-admits the reply before the answer, or null to go on.
    /// モデルがいま閉じた思考ブロック: id が最後の `canon_near` の中で違う時に作り直し (canonThink。
    /// TF_GLM53_CANON_CLOSE=near が既定、all はどこでも、off は作り直さない)、次のターンが組む id の上で答えを計算する。
    /// 答えの前に応答を受け入れ直す打ち切り、または続けるなら null。
    fn closeCut(h: *Host, job: *Job) ?Cut {
        if (h.canon_close == .off or job.request.think_canon == null) return null;
        const end = job.live.items.len -| 1; // the close itself stays / 閉じるトークン自身は残す
        const from: ?usize = if (h.canon_close == .near) end -| h.canon_near else null;
        _ = h.canonThink(job, from, "close") orelse return null;
        return job.fits(.{ .fold = .canon });
    }

    /// The decode rank 0 stopped for a notice, the budget's cut or a fold (job.cut, at a line break): its text into the
    /// reply and the context, the context folded, then the job back to the head of the queue with its stream open, to
    /// go on from the prompt and the context (an admission the cache resumes up to where the context changed;
    /// TF_GLM53_MULTI: conversation `k` freed first). False: the client left or the server closes (the caller ends
    /// the reply).
    /// rank 0 がお知らせか予算の打ち切りのために止めた decode (job.cut、改行のところ): その文を応答に書き、job をストリームを
    /// 開いたまま待ち行列の先頭へ戻す。prompt・ここまでの応答・その文から続ける (キャッシュが文以外を再開する受け入れ。
    /// TF_GLM53_MULTI では先に会話 `k` を空ける)。false: クライアントが去った、またはサーバーが終わる (呼び出し側が応答を終える)。
    fn goOn(h: *Host, job: *Job, st: glm.engine.Stats, k: ?usize) bool {
        const c = job.cut.?;
        job.cut = null;
        const gone = blk: {
            h.lock();
            defer h.unlock();
            break :blk h.closing or std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null;
        };
        if (gone) return false;
        // the text into the reply and the context alike; then the fold, of the context only
        // 文は応答にも文脈にも入れる。そのあとの畳み込みは文脈だけ
        job.emitted.appendSlice(h.gpa, c.text) catch return false;
        job.live.appendSlice(h.gpa, c.text) catch return false;
        if (!job.think_shut) {
            job.think_n += c.text.len; // a notice is think text too / お知らせも思考の文
            job.fold_n += c.text.len;
        }
        const before = job.live.items.len;
        const ck_len = job.live.items.len -| job.ck_start;
        const br_len = job.br_text.items.len;
        const br_notes = job.br_split;
        const br_secs = @as(f64, @floatFromInt(h.now() - job.br_t0)) / 1e9;
        var br_kind: Job.BranchFold = .summary;
        switch (c.fold) {
            .none, .canon => {},
            .recent => job.foldRecent(),
            .checkpoint => job.foldCheckpoint(h.gpa) catch return false,
            // the summary beside the reply: its segment runs from the context with the request after it, and the
            // client gets the label now (the summary follows as the model writes it)
            // 応答の横での要約: その区間は文脈とその後の依頼から動き、クライアントにはいま見出しを送る (要約はモデルが
            // 書くとおりに続く)
            .branch_open => {
                job.br_writing = true;
                job.br_cap = job.stepCap();
                job.br_t0 = h.now();
                job.br_split = 0;
                job.br_text.clearRetainingCapacity();
                job.emitted.appendSlice(h.gpa, job.step().label) catch return false;
                job.emitted.appendSlice(h.gpa, job.step().heading) catch return false;
            },
            .branch_close => {
                br_kind = job.foldBranch(h.gpa) catch return false;
                job.br_room = null;
                // the budget's step leaves the block open, and the model closes it, or its memory filling does;
                // the close's step shuts it after what it wrote
                // 予算の手順はブロックを開いたままにし、モデルが閉じる。閉じなければメモリが埋まった時に閉じる。閉じる時の
                // 手順は書いたもののあとで閉じる
                if (job.br_step == .budget and br_kind == .summary and job.step().place != .show) {
                    job.went_on = true;
                    job.want = null;
                    job.want_wrap = false;
                    std.log.info("glm53: thinking: after the budget's summary the model goes on, until the thinking's memory of {d} tokens is full ({d} in it; prompt {d})", .{ job.request.think_budget, job.live.items.len, job.prompt0().len });
                } else if (job.br_step != .midway) {
                    job.live.appendSlice(h.gpa, job.request.think_close) catch return false;
                    job.emitted.appendSlice(h.gpa, job.request.think_close) catch return false;
                    job.think_shut = true;
                }
            },
        }
        const folded = c.fold == .recent or c.fold == .checkpoint or (c.fold == .branch_close and br_kind != .kept);
        if (folded) job.folds += 1;
        // a fold remakes the context's think ids as the tokenizer cuts their text, so a client's copy renders to them
        // next turn (its admission computes past `prompt + head` anyway); an inserted text needs none, for it starts
        // with no line break of its own (server.inserted), and a close's were remade by closeCut
        // 畳み込みは文脈の思考の id をトークナイザーが文を切るとおりに作り直す (次のターンにクライアントの写しが同じ id になる。
        // その受け入れはもともと `prompt + head` より後ろを計算する)。差し込む文は自分の改行で始まらないので要らない
        // (server.inserted)。閉じた時のものは closeCut が作り直した
        const was = job.live.items.len;
        if (folded) _ = h.canonThink(job, null, "fold");
        // the engine closed the block with a text (its last id the close): the same as the model's own close (closeCut),
        // or the line break it wrote beside the model's last id cuts otherwise next turn and the answer falls out of the
        // cache
        // エンジンが文でブロックを閉じた (最後の id が閉じるトークン): モデル自身が閉じた時 (closeCut) と同じ。さもないと、
        // モデルの最後の id の横に書いた改行が次のターンに別の切り方になり、答えがキャッシュから外れる
        if (c.fold == .none and job.think_shut and h.canon_close != .off and c.text.len > 0 and job.request.think_end != null and c.text[c.text.len - 1] == job.request.think_end.?) {
            const end = job.live.items.len -| 1;
            _ = h.canonThink(job, if (h.canon_close == .near) end -| h.canon_near else null, "close");
        }
        // remade ids may change the context's length: the next fold starts past the summary as it now stands
        // 作り直した id で文脈の長さが変わることがある: 次の畳み込みは、いまの要約の後ろから始める
        if (c.fold == .branch_close and br_kind == .summary and job.step().place != .show) job.fold_from = job.live.items.len;
        if (!job.think_shut) job.fold_n = @intCast(@max(0, @as(i64, @intCast(job.fold_n)) + @as(i64, @intCast(job.live.items.len)) - @as(i64, @intCast(was))));
        if (folded) job.fold_kept = job.live.items.len;
        const asked = job.ck == .opening;
        if (asked) {
            job.ck = .writing;
            job.ck_start = job.live.items.len;
            job.ck_n = 0;
        }
        const n0 = job.prompt0().len;
        const ask: []const u32 = if (c.fold == .branch_open) job.step().ask else &.{};
        const p = h.gpa.alloc(u32, n0 + job.live.items.len + ask.len) catch return false;
        @memcpy(p[0..n0], job.prompt0());
        @memcpy(p[n0..][0..job.live.items.len], job.live.items);
        @memcpy(p[n0 + job.live.items.len ..], ask);
        if (job.seg_prompt) |old| h.gpa.free(old);
        job.seg_prompt = p;
        if (c.text.len > 0) emit(job, .{ .tokens = c.text });
        if (c.fold == .branch_open) {
            if (job.step().label.len > 0) emit(job, .{ .tokens = job.step().label });
            if (job.step().heading.len > 0) emit(job, .{ .tokens = job.step().heading });
        }
        if (c.fold == .branch_close and br_kind == .summary and job.step().place != .show and job.step().after.len > 0) emit(job, .{ .tokens = job.step().after });
        if (c.fold == .branch_close and job.think_shut) emit(job, .{ .tokens = job.request.think_close });
        job.acc_rounds += if (st.rounds > 0) st.rounds else st.generated -| 1;
        job.acc_drafted += st.drafted;
        job.acc_accepted += st.accepted;
        if (job.segs == 0) job.acc_prefill_ns = st.prefill_ns else job.acc_decode_ns += st.prefill_ns;
        job.acc_decode_ns += st.decode_ns;
        job.segs += 1;
        switch (c.fold) {
            .recent => std.log.info("glm53: thinking: fold {d} (recent) at {d} think tokens: the context's reply {d} -> {d} tokens (head {d}, last {d}; prompt {d})", .{ job.folds, job.think_n, before, job.live.items.len, job.request.think_fold_head, job.fold_n, job.prompt0().len }),
            // the summary at the close shut the block (a fold's checkpoint never does; its stage texts may be absent)
            // 閉じる時の要約はブロックを閉じる (畳み込みのチェックポイントは閉じない。段階の文が無いこともある)
            .checkpoint => if (job.think_shut)
                std.log.info("glm53: thinking: fold {d} (summary at the close) at {d} think tokens: the context's reply {d} -> {d} tokens (summary {d}; prompt {d}), the answer goes on from it", .{ job.folds, job.think_n, before, job.live.items.len, ck_len, job.prompt0().len })
            else
                std.log.info("glm53: thinking: fold {d} (checkpoint, stage {d}{s}) at {d} think tokens: the context's reply {d} -> {d} tokens (checkpoint {d}; prompt {d})", .{ job.folds, job.ck_stage, if (c.text.ptr == job.request.think_fold_closes[job.ck_stage].ptr) ", ended at its cap" else "", job.think_n, before, job.live.items.len, ck_len, job.prompt0().len }),
            .canon => std.log.info("glm53: thinking: the think block closed after {d} reply tokens; the answer goes on from its remade ids", .{job.emitted.items.len}),
            .branch_open => std.log.info("glm53: thinking: a summary beside the reply asked for{s} at {d} think tokens ({d} in the context; prompt {d})", .{ stepName(job.br_step), job.think_n, job.fold_n, job.prompt0().len }),
            .branch_close => if (br_kind == .kept)
                std.log.info("glm53: thinking: the summary{s} came back with {d} tokens: the think block closed as it is (prompt {d})", .{ stepName(job.br_step), br_len, job.prompt0().len })
            else if (br_kind == .summary)
                std.log.info("glm53: thinking: fold {d} (branch{s}) at {d} think tokens: the context's reply {d} -> {d} tokens (summary {d} tokens{s}, {d} of its own thinking first, in {d:.1} s; prompt {d})", .{ job.folds, stepName(job.br_step), job.think_n, before, job.live.items.len, if (br_len - br_notes >= 16) br_len - br_notes else br_notes, if (br_len - br_notes >= 16) "" else " (its thinking: the answer was too short)", br_notes, br_secs, job.prompt0().len })
            else
                std.log.info("glm53: thinking: fold {d} (branch) at {d} think tokens: the summary came back with {d} tokens, folded as recent: the context's reply {d} -> {d} tokens (prompt {d})", .{ job.folds, job.think_n, br_len, before, job.live.items.len, job.prompt0().len }),
            .none => if (job.unclosed)
                std.log.info("glm53: thinking: the reply ended in its think block at {d} think tokens: the block closed, the answer goes on (prompt {d})", .{ job.think_n, job.prompt0().len })
            else if (job.think_shut and job.budget_near)
                std.log.info("glm53: thinking: the thinking budget left ({d} of {d}) holds no summary midway and the summary at the close: the think block closed at the fold after {d} reply tokens, the answer goes on", .{ job.request.think_budget -| job.think_n, job.request.think_budget, job.emitted.items.len })
            else if (asked and job.ck_close and job.want_wrap and job.budget_near)
                std.log.info("glm53: thinking: the thinking budget left ({d} of {d}) holds no summary midway and the summary at the close: the summary at the close asked for at the fold (prompt {d})", .{ job.request.think_budget -| job.think_n, job.request.think_budget, job.prompt0().len })
            else if (job.think_shut and job.room_full)
                std.log.info("glm53: thinking: the thinking filled its share of the context again with no fold left ({d} folds, {d} think tokens): the think block closed after {d} reply tokens, the answer goes on", .{ job.folds, job.think_n, job.emitted.items.len })
            else if (job.think_shut and job.went_on)
                std.log.info("glm53: thinking: past the budget's summary {s} (memory {d} tokens, {d} in the context): the think block closed after {d} reply tokens, the answer goes on", .{ if (job.live.items.len >= job.request.think_budget) "the thinking's memory is full" else "only the answer's part is left", job.request.think_budget, job.live.items.len, job.emitted.items.len })
            else if (job.think_shut)
                std.log.info("glm53: thinking: the budget of {d} think tokens reached: the think block closed after {d} reply tokens, the answer goes on", .{ job.request.think_budget, job.emitted.items.len })
            else if (asked and job.ck_close and job.want_wrap and job.room_full)
                std.log.info("glm53: thinking: the thinking filled its share of the context again with no fold left ({d} folds, {d} in the context since the last fold; prompt {d}): a summary asked for before the answer", .{ job.folds, job.fold_n, job.prompt0().len })
            else if (asked and job.ck_close and job.want_wrap)
                std.log.info("glm53: thinking: the budget of {d} think tokens reached ({d} in the context since the last fold; prompt {d}): a summary asked for before the answer", .{ job.request.think_budget, job.fold_n, job.prompt0().len })
            else if (asked and job.ck_close)
                std.log.info("glm53: thinking: the model closed the think block at {d} think tokens ({d} in the context since the last fold; prompt {d}): a summary asked for before the answer", .{ job.think_n, job.fold_n, job.prompt0().len })
            else if (asked)
                std.log.info("glm53: thinking: a checkpoint (stage {d}) asked for at {d} think tokens ({d} in the context since the last fold; prompt {d})", .{ job.ck_stage, job.think_n, job.fold_n, job.prompt0().len })
            else
                std.log.info("glm53: thinking: notice {d} written at {d} think tokens ({d} tokens), the reply goes on", .{ job.notice_k - 1, job.think_n - c.text.len, c.text.len }),
        }
        h.lock();
        job.generated = job.emitted.items.len;
        // the stop bit was the cut's, not a cancel's / 停止ビットは打ち切りのもので、取り消しではない
        job.stop.store(false, .monotonic);
        if (k) |kk| {
            h.act[kk] = null;
            // the scheduler holds it until it is queued again: a cancel meanwhile is recorded for it (requeue takes it)
            // 待ち行列に戻るまではスケジューラが持つ: その間の取り消しはこの job のものとして記録する (requeue が拾う)
            h.admitting = job;
            h.unlock();
            h.requeue(job);
            // this release is not the admission's: the scheduler's next turn decides as before
            // この返却は受け入れのものではない: スケジューラの次の番はいつもどおり決める
            h.requeued = false;
            return true;
        }
        // queued and taken off `running` under one lock: a cancel finds it in one or the other
        // 待ち行列に入れるのと `running` から外すのは同じロックの下で: 取り消しはどちらかで見つかる
        defer h.unlock();
        h.queued.insert(h.gpa, 0, job) catch return false;
        if (h.running == job) h.running = null;
        h.wake.signal(h.io);
        return true;
    }

    // ----------------------------------------------- TF_GLM53_MULTI: batched conversations / まとめた会話 ---

    fn sinkOf(out: *Out) glm.engine.Sink {
        return .{ .ctx = @ptrCast(out), .tokens = Out.tokens, .prefilled = Out.prefilled, .chunk = Out.chunk, .round = Out.round };
    }

    fn liveCount(h: *Host) usize {
        var n: usize = 0;
        for (h.act[0..h.multi_n]) |a| n += @intFromBool(a != null);
        return n;
    }

    /// Rank 0's scheduler with TF_GLM53_MULTI: admit the head of the queue into a free conversation (its prefill runs
    /// alone, the others wait), then one batched round over the live conversations, then release the ones done; a
    /// sampled request runs the old way once no conversation is live.
    /// TF_GLM53_MULTI での rank 0 のスケジューラ: 待ち行列の先頭を空いた会話に受け入れ (その prefill は単独で走り、他は
    /// 待つ)、使用中の会話をまとめたラウンドを 1 回、終わった会話を返す。サンプリングの要求は、使用中の会話が無くなって
    /// から従来の方法で実行する。
    fn runMulti(h: *Host) void {
        h.ctx.makeCurrent() catch |err| fatal("binding the CUDA context", err);
        while (true) {
            // the waits past TF_GLM53_MULTI_WAIT_S, before the lock (expireWaits takes it)
            // TF_GLM53_MULTI_WAIT_S を過ぎた待ち。鍵を取る前に (expireWaits が自分で取る)
            h.expireWaits();
            h.lock();
            while (h.queued.items.len == 0 and h.liveCount() == 0 and h.swap_want == null and !h.closing) {
                h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            }
            // a change of the bits goes before what came after it was asked (nothing was live or waiting then); the
            // pool is cold after it, so the queued jobs' estimates are made again
            // bit の変更は、求められた後に来たものより先 (そのとき使用中も待ちもなかった)。変更の後の
            // pool は空なので、待ち行列の job の見積もりは作り直す
            if (h.swap_want) |want| if (!h.closing and h.liveCount() == 0) {
                h.swap_want = null;
                h.swap.state = .running;
                h.pool_gen +%= 1;
                h.unlock();
                h.runSwap(want);
                h.memSnap();
                continue;
            };
            if (h.closing) {
                if (h.swap_want) |want| h.gpa.free(want);
                h.swap_want = null;
                const left = h.gpa.dupe(*Job, h.queued.items) catch &.{};
                h.queued.clearRetainingCapacity();
                for (h.act[0..h.multi_n]) |a| if (a) |job| job.stop.store(true, .monotonic);
                h.unlock();
                for (left) |job| h.finish(job, .cancelled, .{}, "");
                h.gpa.free(left);
                // the live conversations stop (one round of stops) and commit
                // 使用中の会話は止め (停止のラウンド 1 回)、確定する
                h.roundOnce();
                glm.lockstep.shutdownMulti(&h.ch) catch |err| std.log.err("glm53: telling the other ranks to stop failed: {s}", .{@errorName(err)});
                return;
            }
            // greedy and sampled requests alike (a sampled conversation draws its rows with its own keyed sampler)
            // greedy も sampled も同じく (sampled の会話は自分のキー付きの抽選で行を選ぶ)
            var pick: ?*Job = null;
            var at: usize = 0;
            var short = false;
            const share = h.share.load(.monotonic);
            const work = h.decoding();
            if (h.queued.items.len > 0 and h.eng.multiFree() != null) {
                pick = h.queued.items[0];
                // the first short prompt (at most TF_GLM53_MULTI_SHORT_PROMPT rows to compute past the cache) goes
                // before the longer ones waiting
                // 最初の短い prompt (キャッシュより後ろで計算する行が TF_GLM53_MULTI_SHORT_PROMPT 以下) は、待っている
                // 長いものより先に入る
                const limit = h.short.load(.monotonic);
                // a reply going on after a notice or the budget's cut (its text is all it computes) first of all
                // お知らせや予算の打ち切りのあとに続く応答 (計算するのはその文だけ) は何よりも先
                for (h.queued.items, 0..) |job, i| if (job.segs > 0 or (limit > 0 and h.freshOf(job) <= limit)) {
                    pick = job;
                    at = i;
                    short = true;
                    break;
                };
            }
            // the decode's share (glm.pace): a prompt that can be admitted goes first unless conversations decode and
            // are behind their share of the time (100: while any decodes; 0: never, so the queue is admitted back to
            // back). A short prompt is not held by it
            // decode の取り分 (glm.pace): 受け入れられる prompt が先。ただし会話が decode していて、時間の取り分に届いて
            // いなければ decode が先 (100 ならどれかが decode している間ずっと。0 なら決して。待ち行列は続けて受け入れる)。
            // 短い prompt はこれで止めない
            const held = work > 0 and h.pace.behind(share);
            const admit = pick != null and (short or !held);
            if (admit) {
                _ = h.queued.orderedRemove(at);
                h.admitting = pick;
                pick.?.short = short;
            }
            h.unlock();
            if (admit and short and pick.?.segs == 0 and (at > 0 or held))
                std.log.info("glm53: schedule: a short prompt ({d} rows to compute) goes first: past {d} waiting{s}", .{ pick.?.fresh, at, if (held) ", beside the decode's share" else "" });
            if (admit) {
                // the prefill beside decoding conversations is counted from here to its end (its pauses count their
                // own parts); with none decoding there is nothing to share
                // decode 中の会話の横の prefill は、ここから終わりまでを数える (止まる所は自分の分を数える)。decode 中の会話が
                // なければ分けるものはない
                if (work > 0) h.pace.begin(h.now(), 0) else h.pace.idle();
                h.requeued = false;
                h.admitJob(pick.?);
                h.pace.cut(h.share.load(.monotonic), h.now());
                // admitted or ended: the next turn decides again. Back in the queue (no room beside the live ones,
                // which only rounds free): a round first
                // 受け入れた、または終わった: 次の番でまた決める。待ち行列に戻った (使用中の会話の横に空きがなく、ラウンド
                // でしか空かない): 先にラウンド
                if (!h.requeued) continue;
            }
            const t0 = h.now();
            h.roundOnce();
            // a round while a prompt could have been admitted is the decode's part; with none waiting, nothing is shared
            // prompt を受け入れられたのに回したラウンドは decode の分。待つものがなければ分けるものはない
            if (pick != null and work > 0) h.pace.addDecode(share, h.now() - t0) else h.pace.idle();
        }
    }

    /// The live conversations not done (rank 0's scheduler thread).
    /// 未完了の使用中の会話の数 (rank 0 のスケジューラのスレッド)。
    fn decoding(h: *Host) usize {
        const m = h.eng.multi.?;
        var n: usize = 0;
        for (h.act[0..h.multi_n], 0..) |a, k| n += @intFromBool(a != null and !m.convs[k].done);
        return n;
    }

    /// A queued job's rows to compute past the cache if admitted now (rank 0's scheduler thread, under the lock): the
    /// estimate kept until the pool changes (glm.pool_serve.freshRows).
    /// 待ち行列の job を今受け入れたら、キャッシュより後ろで計算する行数 (rank 0 のスケジューラのスレッド、ロックの下で):
    /// 見積もりは pool が変わるまで使い回す (glm.pool_serve.freshRows)。
    fn freshOf(h: *Host, job: *Job) u64 {
        h.byteResume(job);
        if (job.fresh_gen != h.pool_gen) {
            job.fresh = glm.pool_serve.freshRows(h.eng, job.promptNow());
            job.fresh_gen = h.pool_gen;
        }
        return job.fresh;
    }

    /// The job into a free conversation: its plan beside the live ones, its prefill and first token. No room in the
    /// pool beside them: the job goes back to the head of the queue.
    /// job を空いた会話へ: 使用中の会話の横でのプラン、prefill、最初のトークン。その横に pool の余地がなければ、job は
    /// 待ち行列の先頭へ戻る。
    fn admitJob(h: *Host, job: *Job) void {
        // the pool's slots change (or may): the queued jobs' estimates are made again
        // pool のスロットが変わる (かもしれない): 待ち行列の job の見積もりは作り直す
        defer h.pool_gen +%= 1;
        h.byteResume(job);
        const r = job.request;
        const seg = job.promptNow();
        if (job.segs == 0) job.began = h.now();
        job.context = h.eng.capacity;
        if (seg.len == 0 or seg.len > h.eng.promptLimit()) {
            noPrefill(job);
            return h.finish(job, .failed, .{}, "the prompt does not fit this server's context window");
        }
        const prompt = h.gpa.alloc(i64, seg.len) catch {
            noPrefill(job);
            return h.finish(job, .failed, .{}, "out of host memory");
        };
        defer h.gpa.free(prompt);
        for (prompt, seg) |*pp, t| pp.* = t;
        const k = h.eng.multiFree() orelse return h.requeue(job);
        const mode: u8 = if (!r.drafts) 0 else if (h.eng.mtp != null) glm.lockstep.mode_mtp else if (h.eng.drafter != null) glm.lockstep.mode_dflash else 0;
        var header: glm.lockstep.Header = .{ .max_tokens = @intCast(@min(job.maxNow(), std.math.maxInt(i32))), .stop_eos = r.eos.len > 0, .mode = mode };
        // a later segment of the same reply continues its own slot (Header.own_slot)
        // 同じ応答の 2 つ目以降の区間は自分のスロットを続ける (Header.own_slot)
        if (job.segs > 0) header.own_slot = job.pool_x;
        // the server resolved the model's defaults and the prompt's seed (as serve)
        // モデルの既定値と prompt の seed はサーバーが決めてある (serve と同じ)
        if (r.sampling) |s| if (s.temperature > 0) {
            header.seed = s.seed;
            header.temperature = s.temperature;
            header.top_k = @intCast(s.top_k);
            header.top_p = s.top_p;
            header.min_p = s.min_p;
        };
        h.outs[k] = .{ .h = h, .job = job, .k = @intCast(k) };
        job.multi = true;
        job.yielding = false;
        job.adm_t0 = null;
        h.mid_paused_ns = 0;
        job.conv = @intCast(k);
        job.pool_id = null;
        job.stop.store(false, .monotonic);
        h.lock();
        h.admitting = null;
        h.prefilling = job;
        h.eng.stop_req.store(false, .monotonic);
        // a cancel that came after the dequeue and before this registration (cancelFn found it neither queued nor
        // prefilling): honoured now
        // 取り出した後、この登録より前に来た取り消し (cancelFn は待ち行列にも prefill 中にも見つけられなかった): ここで効かせる
        if (std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null) h.stopJob(job);
        h.unlock();
        // its plan changes the pool from here: the queued jobs' estimates read inside its prefill (yieldFor) are made
        // again
        // ここからそのプランが pool を変える: その prefill の中で読む待ち行列の見積もり (yieldFor) は作り直す
        h.pool_gen +%= 1;
        glm.lockstep.leadAdmit(&h.ch, h.eng, k, header, prompt, sinkOf(&h.outs[k]), .{ .ctx = @ptrCast(h), .f = liveFor, .release = releaseFor, .pace = &h.pace, .share = &h.share, .straight = job.short, .yield = yieldFor, .paused = &h.mid_paused_ns }) catch |err| {
            h.lock();
            h.prefilling = null;
            // back in the scheduler's hands (a cancel from now on is recorded for it)
            // スケジューラの手に戻る (これ以降の取り消しはこの job のものとして記録する)
            h.admitting = job;
            // a cancel while it was planned (its stop bit, or recorded): it ends here as cancelled, neither back in the
            // queue (the cancel came once) nor refused later as busy
            // 計画中に来た取り消し (停止ビット、または記録): ここで cancelled として終える。待ち行列には戻さない
            // (取り消しは 1 回しか来ない)、あとで busy として断ることもしない
            const cancelled = h.takeCancel(job);
            const live = h.liveCount();
            h.unlock();
            if (cancelled) {
                job.multi = false;
                return h.finish(job, .cancelled, .{}, "");
            }
            switch (err) {
                // refused before it was shared: every rank goes on serving
                // 共有の前に断った: 全ランクがそのまま処理を続ける
                error.NoRoom => {
                    // nothing else in flight: no wait frees room (the start's bound holds one request alone)
                    // 処理中のものが無い: 待っても空かない (起動時の下限は 1 要求だけなら収まる)
                    if (live == 0) {
                        noPrefill(job);
                        return h.finish(job, .failed, .{}, "the KV pool cannot hold this request's prompt and max_tokens");
                    }
                    const t = h.now();
                    const since = job.room_since orelse t;
                    job.room_since = since;
                    if (h.room_wait_ns > 0 and t - since >= h.room_wait_ns) return h.refuseBusy(job, t - since);
                    return h.requeue(job);
                },
                error.ShapeNotVerified => {
                    job.multi = false;
                    noPrefill(job);
                    return h.finish(job, .failed, .{}, "this prompt needs a kernel shape the server did not verify at start-up (see the server log)");
                },
                else => fatal("admitting a shared request", err),
            }
        };
        h.lock();
        h.prefilling = null;
        h.act[k] = job;
        h.unlock();
        // done at once (its first token ended it, or it stopped in the prompt)
        // すぐに終わった (最初のトークンで終わった、または prompt の途中で止まった)
        if (h.eng.multi.?.convs[k].done) h.releaseConv(k);
    }

    /// The live conversations not done but `skip` (the one being admitted), for the rounds inside its prefill.
    /// prefill 中の会話 `skip` を除く、未完了の使用中の会話 (その prefill の中のラウンド用)。
    fn liveFor(ctx: *anyopaque, skip: usize, ks: []u8, stops: []bool, sinks: []?glm.engine.Sink) usize {
        const h = self(ctx);
        h.expireWaits(); // a long prefill holds the scheduler: the waits are judged here too / 長い prefill の間もここで
        // closing: every conversation stopped at its next round, so a prefill held for them (decode share 100) ends
        // 終了中: どの会話も次のラウンドで止める。会話のために待っている prefill (decode の取り分 100) も終わる
        const closing = blk: {
            h.lock();
            defer h.unlock();
            break :blk h.closing;
        };
        const m = h.eng.multi.?;
        var n: usize = 0;
        for (h.act[0..h.multi_n], 0..) |a, k| if (a) |job| {
            if (k == skip or m.convs[k].done) continue;
            ks[n] = @intCast(k);
            stops[n] = closing or job.stop.load(.monotonic);
            sinks[n] = sinkOf(&h.outs[k]);
            n += 1;
        };
        return n;
    }

    fn releaseFor(ctx: *anyopaque, k: usize) void {
        self(ctx).releaseConv(k);
    }

    /// The prefill in flight, `left` rows from its end, gives way when TF_GLM53_MULTI_SHORT_PREEMPT is on, a short
    /// prompt waits and `left` is longer than a short prompt (then it is marked so; rank 0's scheduler thread, inside
    /// the prefill).
    /// 実行中の prefill (終わりまで `left` 行) は、TF_GLM53_MULTI_SHORT_PREEMPT がオンで、短い prompt が待っていて、
    /// `left` が短い prompt より長ければ譲る (そのとき譲るものとして印を付ける。rank 0 のスケジューラのスレッドが prefill の
    /// 中で呼ぶ)。
    fn yieldFor(ctx: *anyopaque, left: u64) bool {
        const h = self(ctx);
        if (!h.preempt.load(.monotonic)) return false;
        const limit = h.short.load(.monotonic);
        if (limit == 0 or left <= limit) return false;
        h.lock();
        defer h.unlock();
        const job = h.prefilling orelse return false;
        if (h.closing or job.yielding) return job.yielding;
        for (h.queued.items) |q| if (h.freshOf(q) <= limit) {
            job.yielding = true;
            job.yields += 1;
            job.yield_row = job.prompt0().len -| left;
            return true;
        };
        return false;
    }

    /// TF_GLM53_MULTI_WAIT_S passed for a request waiting for room: refused as busy (503, overloaded_error).
    /// 空きを待つ要求が TF_GLM53_MULTI_WAIT_S を過ぎた: busy として断る (503、overloaded_error)。
    fn refuseBusy(h: *Host, job: *Job, waited: i96) void {
        std.log.warn("glm53: a request of {d} prompt tokens waited {d} s for room in the KV pool beside {d} conversations: refused as busy", .{ job.prompt0().len, @divTrunc(waited, std.time.ns_per_s), h.liveCount() });
        noPrefill(job);
        h.finish(job, .busy, .{}, "no room in the KV pool beside the conversations in flight within the wait limit (TF_GLM53_MULTI_WAIT_S): retry later, or ask for fewer max_tokens");
    }

    /// The queued requests whose wait for room began and passed TF_GLM53_MULTI_WAIT_S, refused whether or not a
    /// conversation is free to try them again: every scheduler pass and every round inside a prefill. Takes the lock:
    /// never call it holding it.
    /// 空きの待ちが始まって TF_GLM53_MULTI_WAIT_S を過ぎた待ち行列の要求を、試し直せる会話の空きがなくても断る:
    /// スケジューラの各周回と、prefill の中の各ラウンドで。
    fn expireWaits(h: *Host) void {
        if (h.room_wait_ns == 0) return;
        const t = h.now();
        var gone: [32]*Job = undefined;
        var waited: [32]i96 = undefined;
        var n: usize = 0;
        h.lock();
        var i: usize = 0;
        while (i < h.queued.items.len and n < gone.len) {
            const job = h.queued.items[i];
            if (job.room_since) |since| if (t - since >= h.room_wait_ns) {
                _ = h.queued.orderedRemove(i);
                gone[n] = job;
                waited[n] = t - since;
                n += 1;
                continue;
            };
            i += 1;
        }
        h.unlock();
        for (gone[0..n], waited[0..n]) |job, w| h.refuseBusy(job, w);
    }

    /// The job back to the head of the queue (no free conversation, or no room beside the live ones), unless a cancel
    /// came for it while the scheduler held it: then it ends here as cancelled.
    /// job を待ち行列の先頭へ戻す (空いた会話がない、または使用中の会話の横に空きがない)。スケジューラが持っている間に
    /// 取り消しが来ていれば、ここで cancelled として終える。
    fn requeue(h: *Host, job: *Job) void {
        job.multi = false;
        h.requeued = true;
        h.lock();
        if (h.admitting == job) h.admitting = null;
        if (h.takeCancel(job)) {
            h.unlock();
            return h.finish(job, .cancelled, .{}, "");
        }
        h.queued.insert(h.gpa, 0, job) catch {
            h.unlock();
            return h.finish(job, .failed, .{}, "out of host memory");
        };
        h.unlock();
    }

    /// One batched round over the live conversations not done (rank 0's stop bit for each), then the done ones
    /// released.
    /// 未完了の使用中の会話をまとめたラウンド 1 回 (それぞれに rank 0 の停止ビット)、続いて終わった会話を返す。
    fn roundOnce(h: *Host) void {
        const m = h.eng.multi.?;
        var ks: [glm.engine.multi_max]u8 = undefined;
        var stops: [glm.engine.multi_max]bool = undefined;
        var sinks: [glm.engine.multi_max]?glm.engine.Sink = undefined;
        var n: usize = 0;
        for (h.act[0..h.multi_n], 0..) |a, k| if (a) |job| {
            if (m.convs[k].done) continue;
            ks[n] = @intCast(k);
            stops[n] = job.stop.load(.monotonic);
            sinks[n] = sinkOf(&h.outs[k]);
            n += 1;
        };
        if (n > 0) glm.lockstep.leadRound(&h.ch, h.eng, ks[0..n], stops[0..n], sinks[0..n]) catch |err| fatal("a batched round", err);
        for (h.act[0..h.multi_n], 0..) |a, k| if (a != null and m.convs[k].done) h.releaseConv(k);
    }

    /// Conversation k done: its commit on every rank, the job finished with its stats.
    /// 会話 k が終わった: 全ランクで確定し、統計を付けて job を終える。
    fn releaseConv(h: *Host, k: usize) void {
        const job = h.act[k].?;
        const st = glm.lockstep.leadRelease(&h.ch, h.eng, k) catch |err| fatal("releasing a conversation", err);
        h.pool_gen +%= 1;
        // a prefill that gave way to a short prompt: its chunks done are committed (as a stop in the prompt); the job
        // goes back to the head of the queue with its stream open, to resume from them, and no round comes first
        // 短い prompt に譲った prefill: 終わったチャンクは確定した (prompt の途中の停止と同じ)。job はストリームを開いた
        // まま待ち行列の先頭へ戻り、そこから再開する。先にラウンドは回さない
        if (job.yielding) {
            job.yielding = false;
            h.lock();
            h.act[k] = null;
            // the chunks it computed count when it resumes (the one it gave up at a pause is not in job.done either)
            // 計算したチャンクの時間は再開のときに引き継ぐ (止まる所で手放したチャンクは job.done にも入っていない)
            job.prefill_base_ns = job.prefill_ns;
            job.adm_t0 = null;
            h.unlock();
            std.log.info("glm53: schedule: a prompt of {d} tokens gives way to a short one at row {d} ({d} time{s}); it resumes from there", .{ job.prompt0().len, job.yield_row, job.yields, if (job.yields == 1) "" else "s" });
            h.requeue(job);
            h.requeued = false;
            return;
        }
        // the summary beside the reply ended with its segment (max_tokens or the context left): closed as at its EOS
        // 応答の横での要約が区間とともに終わった (max_tokens か残りの文脈): EOS で終わった時と同じに閉じる
        if (job.br_writing and job.cut == null and !job.quiet) job.cut = .{ .fold = .branch_close };
        // stopped for a notice or the budget's cut: it goes on from the queue's head (its progress stays out of the
        // totals until its last segment ends)
        // お知らせか予算の打ち切りのために止めた: 待ち行列の先頭から続ける (進み具合は最後の区間が終わるまで合計に入れない)
        if (job.cut != null and h.goOn(job, st, k)) return;
        h.lock();
        h.act[k] = null;
        // its progress joins the totals as it leaves the live ones, under the same lock: status reads it in one or
        // the other, so the totals never fall back
        // 進み具合は使用中から外すのと同じロックの下で合計へ移す: status はどちらか一方で読むので、合計は後戻りしない
        if (job.prefill_at) |start| {
            h.addFinished(&h.totals, job, start, h.now());
            h.totals.requests_total += 1;
        }
        const cancelled = job.quiet and std.mem.indexOfScalar(api.Id, h.cancels.items, job.id) != null;
        // its cancels are spent (run() clears the list a request; the batched scheduler drops a request's as it ends)
        // その取り消しは使い終わった (run() は要求ごとに消す。まとめるスケジューラは終わった要求の分を外す)
        h.dropCancels(job.id);
        const computed = job.done -| job.cached;
        h.unlock();
        const reason: api.Reason = if (job.quiet) (if (cancelled) .cancelled else .stop) else switch (st.finish) {
            .stop => .stop,
            .length => .length,
            .cancelled => .cancelled,
        };
        h.finish(job, reason, statsOf(job, st, computed), "");
    }

    fn followMulti(h: *Host) void {
        h.ctx.makeCurrent() catch |err| fatal("binding the CUDA context", err);
        glm.lockstep.followMulti(&h.ch, h.eng) catch |err| fatal("following rank 0's events", err);
        std.log.info("glm53: rank {d} stopped following (rank 0 shut down)", .{h.group.rank});
    }

    fn follow(h: *Host) void {
        h.ctx.makeCurrent() catch |err| fatal("binding the CUDA context", err);
        glm.lockstep.follow(&h.ch, h.eng, null, null) catch |err| fatal("following rank 0", err);
        std.log.info("glm53: rank {d} stopped following (rank 0 shut down)", .{h.group.rank});
    }
};

/// The routed experts' device bytes on this rank (slabs and scales).
/// このランクの routed expert のデバイスのバイト (置き場とスケール)。
fn expertBytes(e: *const glm.engine.Engine) u64 {
    var ex: u64 = 0;
    for (e.layers) |*L| if (L.experts) |*x| {
        for (x.trellis) |b| ex += b.len;
        for ([_]cuda.DeviceBuffer{ x.suh_g, x.suh_u, x.svh_g, x.svh_u, x.suh_d, x.svh_d }) |b| ex += b.len;
    };
    return ex;
}

/// A failure once a request is shared ends this rank at once with exit code 70; the launcher's watch stops the rest
/// (they wait in a collective).
/// 要求の共有後の失敗では、このランクを直ちに終了コード 70 で終える。残りのランク (collective 通信で待っている)
/// はランチャーの監視が止める。
fn fatal(what: []const u8, err: anyerror) noreturn {
    std.log.err("glm53: {s} failed ({s}); exiting with 70 so the watch stops every rank", .{ what, @errorName(err) });
    std.process.exit(70);
}

/// The build identity every rank must share: sha256 of this executable and of the engine settings (every
/// TF_GLM53_* / TF_EXL3_* variable but TF_GLM53_RANK, sorted).
/// 全ランクで同じでなければならないビルド ID: この実行ファイルとエンジン設定 (TF_GLM53_RANK を除く
/// TF_GLM53_* / TF_EXL3_* の全変数を整列したもの) の sha256。
fn buildIdentity(gpa: std.mem.Allocator, io: std.Io) ![]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    const exe = try std.Io.Dir.cwd().readFileAlloc(io, "/proc/self/exe", gpa, .limited(1 << 30));
    h.update(exe);
    gpa.free(exe);
    var vars: std.ArrayList([]const u8) = .empty;
    defer vars.deinit(gpa);
    var i: usize = 0;
    while (std.c.environ[i]) |e| : (i += 1) {
        const kv = std.mem.span(e);
        if (!(std.mem.startsWith(u8, kv, "TF_GLM53_") or std.mem.startsWith(u8, kv, "TF_EXL3_"))) continue;
        if (std.mem.startsWith(u8, kv, "TF_GLM53_RANK=")) continue;
        try vars.append(gpa, kv);
    }
    std.mem.sort([]const u8, vars.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);
    for (vars.items) |v| {
        h.update(v);
        h.update("\n");
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.allocPrint(gpa, "{x}", .{d});
}

fn envOr(name: [:0]const u8, default: []const u8) []const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else default;
}

fn envNum(name: [:0]const u8, default: usize) !usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10);
}

/// TF_GLM53_MULTI=n from the environment alone, before the load: the pool, DCP 2, DFlash2 with its saves, no index FP8
/// and not on three ranks (DCP 1 and TP3 are not offered with it for now).
/// TF_GLM53_MULTI=n を読み込みの前に環境だけで確かめる: pool、DCP 2、DFlash2 なら写し、index FP8 なし、3 ランクでない
/// (DCP 1 と TP3 は今は同時推論では選べない)。
fn multiPrecheck(world: u32, n: usize) !void {
    const Need = struct {
        fn no(m: usize, what: []const u8, err: anyerror) anyerror {
            std.log.err("glm53: TF_GLM53_MULTI={d} needs {s}", .{ m, what });
            return err;
        }
    };
    if (n > glm.engine.multi_max) return Need.no(n, "at most 4 conversations", error.MultiCount);
    if (!std.mem.eql(u8, envOr("TF_GLM53_PREFIX", ""), "pool")) return Need.no(n, "the KV pool (TF_GLM53_PREFIX=pool)", error.MultiNeedsPool);
    if (world == 3) return Need.no(n, "other than three ranks (TP3 is not offered with it)", error.MultiWithThreeRanks);
    if (try envNum("TF_GLM53_DCP", 1) != 2) return Need.no(n, "DCP 2 (DCP 1 is not offered with it)", error.MultiNeedsDcp2);
    if (std.mem.eql(u8, envOr("TF_GLM53_INDEX_FP8", "0"), "1")) return Need.no(n, "TF_GLM53_INDEX_FP8=0", error.MultiWithIndexFp8);
    if (std.c.getenv("TF_GLM53_DFLASH") != null and try envNum("TF_GLM53_POOL_DRAFT_SAVES", 0) == 0)
        return Need.no(n, "DFlash2's saves (TF_GLM53_POOL_DRAFT_SAVES of 1 or more)", error.MultiNeedsDraftSaves);
    // its own settings' form (engine.multiSetup checks their ranges on what was built): whole numbers, a mode, a step
    // その設定の形 (範囲は読み込みの後に engine.multiSetup が、できたもので確かめる): 整数、モード、刻み
    _ = envNum("TF_GLM53_MULTI_WAIT_S", 300) catch return Need.no(n, "TF_GLM53_MULTI_WAIT_S in whole seconds", error.MultiWait);
    _ = envNum("TF_GLM53_MULTI_DRAFT_ROWS", 1) catch return Need.no(n, "TF_GLM53_MULTI_DRAFT_ROWS a whole number of rows", error.MultiDraftRows);
    const how = envOr("TF_GLM53_MULTI_MTP", "batch");
    if (!std.mem.eql(u8, how, "batch") and !std.mem.eql(u8, how, "serial")) return Need.no(n, "TF_GLM53_MULTI_MTP=batch or serial", error.MultiMtpMode);
    const bucket = envNum("TF_GLM53_MULTI_BUCKET", 0) catch return Need.no(n, "TF_GLM53_MULTI_BUCKET a whole number of rows", error.MultiBucket);
    if (bucket % 1024 != 0) return Need.no(n, "TF_GLM53_MULTI_BUCKET a multiple of 1,024 rows", error.MultiBucket);
    if (std.meta.stringToEnum(glm.engine.ConfPad, envOr("TF_GLM53_MTP_CONF_PAD", "none")) == null) return Need.no(n, "TF_GLM53_MTP_CONF_PAD=none, max or full", error.MtpConfPad);
    const share = envNum("TF_GLM53_MULTI_DECODE_SHARE", 30) catch return Need.no(n, "TF_GLM53_MULTI_DECODE_SHARE a whole percent", error.MultiDecodeShare);
    if (share > 100) return Need.no(n, "TF_GLM53_MULTI_DECODE_SHARE 0 .. 100", error.MultiDecodeShare);
    const short = envNum("TF_GLM53_MULTI_SHORT_PROMPT", 8192) catch return Need.no(n, "TF_GLM53_MULTI_SHORT_PROMPT a whole number of rows", error.MultiShortPrompt);
    if (short > api.short_prompt_max) return Need.no(n, "TF_GLM53_MULTI_SHORT_PROMPT 0 .. 1,048,576 rows", error.MultiShortPrompt);
    const preempt = envNum("TF_GLM53_MULTI_SHORT_PREEMPT", 0) catch return Need.no(n, "TF_GLM53_MULTI_SHORT_PREEMPT 0 or 1", error.MultiShortPreempt);
    if (preempt > 1) return Need.no(n, "TF_GLM53_MULTI_SHORT_PREEMPT 0 or 1", error.MultiShortPreempt);
}

/// This process's rank of the replica: loads it, checks with the others that every rank loaded the same, and starts
/// rank 0's engine thread or a follower. `window`: positions a request may reach (prompt + reply).
/// TF_GLM53_MASTER: rank 0's HOST:PORT for the NCCL id; TF_GLM53_STORED_WORLD: the experts' split on disk.
/// このプロセスが受け持つレプリカのランク: 読み込み、全ランクが同じものを読み込んだかを他と確かめ、rank 0 なら
/// エンジンのスレッドを、他ならフォロワーを起動する。`window`: 要求が届きうる位置の数 (prompt + 応答)。
/// TF_GLM53_MASTER: NCCL id 用の rank 0 の HOST:PORT。TF_GLM53_STORED_WORLD: ディスク上の expert の分割数。
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, window: usize) !*Host {
    const rank: u32 = @intCast(try envNum("TF_GLM53_RANK", 0));
    const world: u32 = @intCast(try envNum("TF_GLM53_WORLD", 1));
    const master = std.c.getenv("TF_GLM53_MASTER") orelse return error.NoMaster;
    const queue_max = try envNum("TF_GLM53_QUEUE", 32);
    if (queue_max == 0) return error.QueueZero;
    // TF_GLM53_EXPERT_BITS (a saved set, applied before the first request) with TF_GLM53_MULTI, whose scheduler changes
    // no bits (not supported for now): refused before anything loads, on every rank alike (the set would wait for ever
    // while the old bits served)
    // TF_GLM53_EXPERT_BITS (保存したセット。最初の要求の前に当てる) と、bit を変えないスケジューラの TF_GLM53_MULTI の
    // 組み合わせ (今は対応しない): 何も読む前に、全ランクで同じく断る
    if (std.c.getenv("TF_GLM53_EXPERT_BITS") != null and try envNum("TF_GLM53_MULTI", 0) > 0) {
        std.log.err("glm53: TF_GLM53_EXPERT_BITS is not supported with TF_GLM53_MULTI (its scheduler changes no bits): start with those bits as the expert plan (the launcher's allocation) instead", .{});
        return error.ExpertBitsWithMulti;
    }
    // TF_GLM53_MULTI refused before anything loads when the start's settings cannot run it (engine.multiSetup checks
    // again after the load, on what was built), on every rank alike
    // TF_GLM53_MULTI を動かせない起動の設定は、何も読む前に全ランクで同じく断る (読み込みの後に engine.multiSetup が
    // できたもので再び確かめる)
    const multi_env = try envNum("TF_GLM53_MULTI", 0);
    if (multi_env > 0) try multiPrecheck(world, multi_env);
    // the engine and the channel keep pointers to the driver, context and group: they live in h from the start
    // エンジンとチャネルは driver・context・group へのポインタを持つので、それらは最初から h に置く
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{
        .gpa = gpa, .io = io, .driver = undefined, .ctx = undefined, .nccl = undefined, .group = undefined,
        .eng = undefined, .ch = undefined, .info_ = .{ .name = "glm53-zig", .lanes = 1, .context_window = @intCast(window), .cuda_fields = true },
        .queue_max = queue_max,
    };
    // TF_GLM53_TOPOLOGY=ring4 refused before any rendezvous when the ring cannot hold it (the engine checks again):
    // four ranks, DCP 1 or 2, PREFILL_REDUCE=ring
    // ring4 を満たさない設定は待ち合わせの前に断る (エンジンも再び確かめる)
    const topology = envOr("TF_GLM53_TOPOLOGY", "full");
    if (!std.mem.eql(u8, topology, "full") and !std.mem.eql(u8, topology, "ring4")) return error.BadTopology;
    if (std.mem.eql(u8, topology, "ring4")) {
        if (world != 4) return error.Ring4World;
        if (try envNum("TF_GLM53_DCP", 1) > 2) return error.Ring4Dcp;
        if (!std.mem.eql(u8, envOr("TF_GLM53_PREFILL_REDUCE", "ring"), "ring")) return error.Ring4PrefillReduce;
    }
    h.driver = try cuda.Driver.open();
    h.ctx = try cuda.Context.init(&h.driver, @intCast(try envNum("TF_CUDA_DEVICE", 0)));
    h.nccl = try cuda.nccl.Library.openEnv();
    h.group = try cuda.tp.Group.init(&h.nccl, rank, world, try cuda.tp.parseMaster(std.mem.span(master)));
    // TF_GLM53_TOPOLOGY=ring4: four nodes cabled as a ring, NODES in ring order (glm53/R3RT-102.md); the engine refuses
    // what would reach the opposite rank directly
    // TF_GLM53_TOPOLOGY=ring4: リングに結線した 4 台、NODES はリングの順。向かいのランクに直接届くものはエンジンが断る
    if (std.mem.eql(u8, topology, "ring4")) h.group.topology = .ring4;
    const t0 = std.Io.Clock.awake.now(io);
    h.eng = try glm.engine.Engine.create(gpa, io, &h.driver, &h.ctx, &h.group, glm.engine.withEnv(.{
        .ckpt = dir,
        .stored_world = try envNum("TF_GLM53_STORED_WORLD", 8),
        .layers = if (std.c.getenv("TF_GLM53_LAYERS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else null,
        .capacity = window,
        .graphs = !std.mem.eql(u8, envOr("TF_GLM53_GRAPHS", "1"), "0"),
        .drafter = if (std.c.getenv("TF_GLM53_DFLASH")) |v| std.mem.span(v) else null,
        .kv_fp8 = std.mem.eql(u8, envOr("TF_GLM53_KV", "bf16"), "fp8") or std.mem.eql(u8, envOr("TF_GLM53_KV", "bf16"), "fp4"),
        .kv_fp4 = std.mem.eql(u8, envOr("TF_GLM53_KV", "bf16"), "fp4"),
        .dcp = @intCast(try envNum("TF_GLM53_DCP", 1)),
    }));
    // every rank's checkpoint, executable and engine env compared before the prewarm, whose collectives would hang
    // on ranks set up differently
    // 全ランクのチェックポイント・実行ファイル・エンジンの env を prewarm の前に比べる。設定の違うランクがあると
    // prewarm の collective 通信が止まるため
    h.ch = glm.lockstep.Channel.init(gpa, h.eng);
    const ident0 = try buildIdentity(gpa, io);
    defer gpa.free(ident0);
    // the expert plan's contents count too, not only its path (its banks' manifests were checked against it)
    // エキスパートの計画はパスだけでなく中身も比べる (バンクの manifest は計画と照らし済み)
    const ident = if (h.eng.plan) |p| try std.fmt.allocPrint(gpa, "{s} plan {x}", .{ ident0, p.digest }) else try gpa.dupe(u8, ident0);
    defer gpa.free(ident);
    const fp = try glm.lockstep.Fingerprint.of(io, gpa, dir, ident, world, rank);
    fp.agree(&h.ch) catch |err| {
        std.log.err("glm53: the ranks loaded different checkpoints, executables, settings or ranks ({s}); refusing to start", .{@errorName(err)});
        return err;
    };
    var mt: usize = 0; // TF_GLM53_MEMTRACE
    glm.engine.memMark(&h.ctx, rank, "-", &mt);
    if (h.eng.load_progress) |*p| p.setPhase(.prewarm);
    {
        const tw = std.Io.Clock.awake.now(io);
        const n = try h.eng.prewarm();
        std.log.info("glm53: rank {d}: {d} decode graphs captured up front in {d:.1} s", .{ rank, n, @as(f64, @floatFromInt(tw.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds())) / 1000.0 });
    }
    // the slot cache, when TF_GLM53_PREFIX=slots
    // スロットキャッシュ (TF_GLM53_PREFIX=slots のとき)
    glm.engine.memMark(&h.ctx, rank, "decode graphs (prewarm)", &mt);
    try glm.slots_serve.setup(&h.ch, h.eng, &h.ctx, try glm.slots.Env.fromEnv(), &.{ dir, envOr("TF_GLM53_EXPERTS_DIR", ""), envOr("TF_GLM53_DFLASH", "") });
    try glm.pool_serve.setup(&h.ch, h.eng); // TF_GLM53_PREFIX=pool
    h.canon_close = std.meta.stringToEnum(@TypeOf(h.canon_close), envOr("TF_GLM53_CANON_CLOSE", "near")) orelse return error.BadCanonClose;
    h.canon_near = try envNum("TF_GLM53_CANON_NEAR", 4096);
    h.byte_resume = std.meta.stringToEnum(@TypeOf(h.byte_resume), envOr("TF_GLM53_BYTE_RESUME", "on")) orelse return error.BadByteResume;
    // TF_GLM53_MULTI=n: up to n conversations decoded together (needs the pool; engine.multiSetup)
    // TF_GLM53_MULTI=n: 最大 n 個の会話を一緒に decode する (pool が要る。engine.multiSetup)
    const multi_n = try envNum("TF_GLM53_MULTI", 0);
    if (multi_n > 0) {
        // every rank's setup agreed before any rank serves: one rank short of memory would otherwise leave the others
        // waiting in the first collective
        // 全ランクの準備を、どのランクも処理を始める前に揃える: 1 ランクだけメモリが足りないと、他は最初の collective
        // 通信で待ち続ける
        const set = if (h.eng.multiSetup(multi_n)) |_| true else |err| blk: {
            std.log.err("glm53: rank {d}: setting up TF_GLM53_MULTI={d} failed: {s}", .{ rank, multi_n, @errorName(err) });
            break :blk false;
        };
        if (!try h.ch.allOk(set)) {
            std.log.err("glm53: rank {d}: TF_GLM53_MULTI={d} could not be set up on every rank; refusing to start", .{ rank, multi_n });
            return error.MultiSetupFailed;
        }
        h.multi_n = multi_n;
        h.info_.lanes = @intCast(multi_n);
        h.room_wait_ns = @as(i96, @intCast(try envNum("TF_GLM53_MULTI_WAIT_S", 300))) * std.time.ns_per_s;
        // TF_GLM53_MULTI_DECODE_SHARE (0 .. 100, default 30): the decode's share at start (rank 0's counts; POST
        // /v1/schedule changes it while the engine runs)
        // TF_GLM53_MULTI_DECODE_SHARE (0 .. 100、既定 30): 起動時の decode の取り分 (使うのは rank 0 のもの。動作中は
        // POST /v1/schedule で変わる)
        const share = try envNum("TF_GLM53_MULTI_DECODE_SHARE", 30);
        if (share > 100) return error.MultiDecodeShare;
        h.share.store(@intCast(share), .monotonic);
        if (rank == 0) std.log.info("glm53: schedule: the decode's share {d}% while a prompt waits or is computed", .{share});
        // TF_GLM53_MULTI_SHORT_PROMPT (rows, default 8,192; 0: none): the short prompt's limit at start (POST
        // /v1/schedule changes it while the engine runs)
        // TF_GLM53_MULTI_SHORT_PROMPT (行数、既定 8,192、0 はなし): 起動時の短い prompt の上限 (動作中は POST
        // /v1/schedule で変わる)
        const short = try envNum("TF_GLM53_MULTI_SHORT_PROMPT", 8192);
        if (short > api.short_prompt_max) return error.MultiShortPrompt;
        h.short.store(@intCast(short), .monotonic);
        if (rank == 0) Host.logShort(@intCast(short));
        // TF_GLM53_MULTI_SHORT_PREEMPT (0 / 1, default 0): whether a prefill running gives way to a short prompt at
        // start (POST /v1/schedule changes it while the engine runs)
        // TF_GLM53_MULTI_SHORT_PREEMPT (0 / 1、既定 0): 起動時に、実行中の prefill が短い prompt に譲るか (動作中は
        // POST /v1/schedule で変わる)
        const preempt = try envNum("TF_GLM53_MULTI_SHORT_PREEMPT", 0);
        if (preempt > 1) return error.MultiShortPreempt;
        h.preempt.store(preempt == 1, .monotonic);
        if (rank == 0) Host.logPreempt(preempt == 1);
        h.sched_start = .{ .decode_share = @intCast(share), .short_prompt = @intCast(short), .short_preempt = preempt == 1 };
    }
    glm.engine.memMark(&h.ctx, rank, "slot cache store", &mt);
    // /health shows the store's positions: 0 when a rank could not allocate it
    // /health にストアの位置数を出す: どこかのランクで確保できなかったときは 0
    if (h.eng.slots) |s| h.info_.slot_positions = @intCast(s.S);
    if (h.eng.pool) |p| h.info_.slot_positions = @intCast(p.env.positions);
    // /metrics' engine_info: the settings a monitor shows beside the numbers
    // /metrics の engine_info: モニタが数値と並べて見せる設定
    const kv = envOr("TF_GLM53_KV", "bf16");
    const kv_name = if (std.mem.eql(u8, kv, "fp4")) "fp4" else if (std.mem.eql(u8, kv, "fp8")) "fp8" else "bf16";
    const cache_name = if (h.eng.pool != null) "pool" else if (h.eng.slots != null) "slots" else "off";
    const drafter_name = if (h.eng.mtp != null) "mtp" else if (h.eng.drafter != null) "dflash2" else "none";
    h.labels = try std.fmt.allocPrint(gpa, "tp=\"{d}\",dcp=\"{d}\",kv=\"{s}\",cache=\"{s}\",drafter=\"{s}\",context=\"{d}\",queue=\"{d}\"", .{ world, h.eng.dcp, kv_name, cache_name, drafter_name, window, queue_max });
    h.info_.labels = h.labels;
    std.log.info("glm53: rank {d}/{d} up in {d:.1} s ({d} layers, window {d})", .{ rank, world, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds())) / 1000.0, h.eng.layers.len, window });
    if (h.eng.load_progress) |*p| p.setPhase(.ready);
    // TF_GLM53_EXPERT_LIVE: the warm-up's rows are not usage; requests start from zero
    // TF_GLM53_EXPERT_LIVE: 暖機の行は使用ではない。要求は 0 から数える
    if (h.eng.live) |lv| lv.reset();
    // rank 0: the memory the control panel shows: the start-up steps, the routed experts' bytes, a first snapshot
    // rank 0: 制御パネルが出すメモリ: 起動の段階、routed expert のバイト、最初の写し
    if (rank == 0) {
        h.mem = try glm.memory_live.MemLive.init(gpa, io, rank, glm.engine.mem_steps[0..glm.engine.mem_n], expertBytes(h.eng));
        h.memSnap();
        h.ckpt = try gpa.dupe(u8, dir);
        // TF_GLM53_EXPERT_BITS=<set>.json (saved by POST /v1/experts/save): the experts changed to it first, before
        // any request
        // TF_GLM53_EXPERT_BITS=<set>.json (POST /v1/experts/save で保存したもの): どの要求よりも先に、expert をそれに変える
        if (std.c.getenv("TF_GLM53_EXPERT_BITS")) |v| {
            h.swap_want = try glm.expert_swap.readSet(gpa, io, std.mem.span(v), h.eng);
            h.swap = .{ .seq = 1, .state = .waiting };
            std.log.info("glm53: expert bits from {s} (the first job)", .{std.mem.span(v)});
        }
    }
    const spawn: std.Thread.SpawnConfig = .{ .stack_size = 16 << 20 };
    h.thread = if (rank == 0)
        (if (h.multi_n > 0) try std.Thread.spawn(spawn, Host.runMulti, .{h}) else try std.Thread.spawn(spawn, Host.run, .{h}))
    else if (h.multi_n > 0) try std.Thread.spawn(spawn, Host.followMulti, .{h}) else try std.Thread.spawn(spawn, Host.follow, .{h});
    return h;
}

/// Rank 0: stops admitting, tells the followers to stop, joins. Others: the follower thread ends when rank 0's stop
/// arrives; closing before that would leave rank 0 waiting, so a follower's server waits for it.
/// rank 0: 受け付けをやめ、フォロワーに停止を伝えて join する。他: rank 0 の停止が届くとフォロワーのスレッドが
/// 終わる。その前に閉じると rank 0 が待ち続けるので、フォロワーのサーバーはそれを待つ。
pub fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    if (h.group.rank == 0) {
        h.lock();
        h.closing = true;
        // TF_GLM53_MULTI: a prefill in flight stops at its next chunk (its done chunks kept), so the close does not wait
        // for a long prompt, nor for one held behind decoding conversations (decode share 100)
        // TF_GLM53_MULTI: 実行中の prefill は次のチャンクで止める (終わったチャンクは残る)。長い prompt も、decode 中の
        // 会話の後ろで待っている prompt (decode の取り分 100) も、終了を待たせない
        if (h.multi_n > 0 and h.prefilling != null) h.eng.stop_req.store(true, .monotonic);
        h.wake.broadcast(h.io);
        h.unlock();
    }
    if (h.thread) |t| t.join();
    h.ch.deinit();
    h.eng.destroy();
    h.group.deinit();
    h.nccl.close();
    h.queued.deinit(h.gpa);
    h.cancels.deinit(h.gpa);
    h.decoded.deinit(h.gpa);
    h.gpa.free(h.labels);
    h.gpa.destroy(h);
}
