English | [日本語](PANEL-GUIDE.ja.md) | [简体中文](PANEL-GUIDE.zh-CN.md)

# Expert Atlas: the control panel

The launcher serves the panel at `http://<first node>:<launcher port>/panel`, on every address by default and with no
key: anyone who can reach the panel can start, stop and change the engine, so keep the nodes on a closed network (or set
`LAUNCHER_HOST=127.0.0.1` and open it from another machine through an ssh tunnel, `ssh -L 8010:127.0.0.1:8010
<first node>`, then `http://127.0.0.1:8010/panel` there). The page has two
screens: the launch screen while the engine is stopped, and the live panel while it runs. It switches between them by
itself. This release's panel runs on four or eight nodes (the count is picked in Nodes on the launch screen).

The circle shows the 19,200 routed experts: layers 3 to 77 run from the inside out, 256 experts a ring.

## Node check

While no nodes are registered, the panel opens here, and it reads how the head is cabled as it opens (on a switch, the list of nodes to choose from). **Check the nodes** does the rest: it finds how the nodes are cabled, checks
them (copying the package where it is missing), registers them and goes on to the launch screen; anything wrong stops it there
and says why. A ring also measures its lanes' bandwidth. A switch takes the lane's registered workers, else the largest node
count the free nodes fill. To use others, change the node count or the ticks and press again.

**A switch with other people's nodes:** the button can take only nodes your ssh key reaches, but choose the node count and the
ticks first, then press. For `docker run ... IMAGE up`, give your own to the start line (`-e NODES=...`, `QUICKSTART.md`).

## Launch screen (engine stopped)

- **Plan** ([ BIT ALLOC ]): the bits each expert loads at. The screen opens with a new plan at the last run's shares
  (before any run: all 3 bit); Open picks the last run's allocation as it was, or a saved one. The head of the card names the plan: the last run, a saved allocation's name, or
  New · unsaved.
  - **Shares**: type each width's share of the experts in the table (5, 4 and 3 bit; the narrowest width with data
    takes the rest, so they add up to 100 %), or drag the knobs on the curve. The scale counts from the 5 bit end,
    which is 0 %: 5 bit 10 % is the range 0–10 %. In a mix the most used experts get the widest bits, ranked by
    prefill 0.3 / decode 0.7 (an expert's decode share weighed 0.7, its prefill share 0.3; `ATLAS-TESTS.md`).
  - **Count** (decode, prefill, prefill 0.3 / decode 0.7): for reading only. The curve, the knobs' readouts (what share
    of the picks the experts up to a point carry), the circle's order and Use follow the counts picked; the plan's bits
    do not change.
  - **A new plan**: changing the shares (typed or with the knobs) makes a new plan (New · unsaved), the experts ranked by
    the plan's usage record (below).
  - **Open**: the last run's allocation and the saved ones, with their shares, mean bits, when and how each was made.
    One opened becomes the plan bit for bit, and the start uses that allocation. Its usage record comes with it: Count
    and the curve show it, a plan whose shares are changed from it is ranked by it, and a run started from either adds
    to it. Its bits stay as saved until the shares are changed. An allocation that needs a width without data, or a bank this
    lane does not have, cannot be opened; the reason is shown under it. Rename and Delete (press it again to confirm)
    work on each saved one; a deleted one goes to the launcher's trash.
  - **Saving**: nothing is saved by itself. A plan is saved when its run stops (Save and stop, below; the live panel's
    Save at any time), with its usage record.
- **Usage records**: each saved allocation keeps the usage of the runs it was saved from (each expert's picks, decode
  and prefill), so its data travels with it. There is no common pool of all runs.
  - New, then a run, then Save: the allocation keeps that run's usage only.
  - Opened (or its shares changed), then a run, then Save: the allocation keeps its usage so far plus that run's.
  - A plan with no usage record (the screen's first new plan, the last run's allocation) is ranked by the shipped
    counts: the 12 domains measured for this release, shipped in the page.

  The 12 domains: Coding, Agent, Japanese, English, Agent A, Agent B, Coding (ja), Coding (en), Linux ops (ja),
  Linux ops (en), Security (ja), Security (en). Each weighs the same: an expert ranks by the largest share it has of its
  layer in any domain.
- **[ WEIGHTS ]**: where the checkpoint and the expert banks are, the same path on every node. By default the checkpoint is
  `~/models/GLM-5.3-EXL3-Base` (Hugging Face `fiore-lab/GLM-5.3-EXL3-Base`) and each bank `~/models/GLM-5.3-EXL3-Experts-<N>bit`
  (`QUICKSTART.md`, section 2). Weights somewhere else: write the directory in the row, or use Browse, then press Check and save.
  Every node is looked at when the screen opens, when a row changes, and every 15 seconds while something required is missing;
  this only reads, nothing is saved. The checkpoint and the 3-bit bank are required: without them the row is red, names the
  nodes and the repository, and Start stays off. Start also stays off while a row shows a directory that is not saved. The other
  widths are optional: a row left empty, or a bank not found, only leaves that width out.
- **Banks and No data**: the experts at each width come from Hugging Face `fiore-lab/GLM-5.3-EXL3-Experts-<N>bit`
  (N = 2, 3, 4, 5), each in its own directory on every node, by default `~/models/GLM-5.3-EXL3-Experts-<N>bit`, with
  its manifest. A width whose bank is missing on any node shows No data with the node and what it lacks. It cannot be
  planned: its row in the table shows No data and takes no share.
- **Memory** ([ NODE ]): each node's memory for the plan (expert, Other weights, KV, Work, free) and the verdict:
  Can start, Can start (little room) or Not enough memory. KV FP4 / FP8 shows the KV size each format would allocate (tokens, and GiB a node); press
  one to choose it. By default the KV takes all the room the plan leaves: Context length for one conversation, the rest for
  earlier conversations.
- **Context length**: the longest request (prompt and reply) the start allows, in tokens: 131K, 262K, 524K, glm53.conf's
  CONTEXT_LENGTH, or Max (the most the plan leaves room for in the chosen KV format, less what the engine needs beyond it: it
  is never the whole KV size, the pool cache keeping at least 512 tokens). A size that does not fit is greyed; a
  context the launcher had to cut is shown under the row. Max is the default until the panel has written one: every choice on this screen goes into glm53.conf as it is made (Max as its number, and again when that moves; not pressed, at the start), and the next page opens on it. The launcher finds Max from the memory a
  start with that drafter took on this lane (measured each start, a drafter at a time); until then it plans on built-in
  sizes with 2 GiB more kept, so Max is shorter (est.); the KV rows say est. or Measured to the left of their size. The measurement is taken just after the start, with nothing
  running, so at Max a node can dip a little under what it keeps for the OS while requests run. The i beside the name
  says this and whether the drafter picked is measured.
- **Keep for the OS**: what each node keeps free after the start, in whole GiB from 2 to 32 (glm53.conf's RESERVE_GB, 4 by
  default). The verdict and the engine's own admission use it; more kept leaves less for the KV, whose size shows in the box below.
- **KV size**: the tokens the KV holds in all. Type a number to set it yourself, at least Context length + 512: Keep for the OS
  then shows what would be left, the least of the nodes, marked Expected, and another KV format keeps the tokens. If the memory
  does not hold it, the verdict is Not enough memory and the context stays as it is. Typing Keep for the OS again gives the KV all
  the room left. Both go into glm53.conf as they change (RESERVE_GB, KV_SIZE).
- **Drafter**: MTP, DFlash2 or None, chosen under the KV format; the memory is planned for it. DFlash2 is not shipped
  (its licence): download it and set DRAFTER in glm53.conf. Until every node has it, it is greyed with the reason.
- **Nodes** (in the card's head, while the engine is stopped): the head (rank 0) is the node the launcher runs on.
  Workers are registered once, whatever the node count (an address, Add), and each count's workers are ticked from
  the list (Use 3 / 3 at 4 nodes, 7 / 7 at 8); the ranks follow the list's order. The list and each count's ticks are
  kept in the lane's panel settings, so they come back on any browser. Delete takes a worker off the list (and off
  every count's ticks); the workers the lane uses now are always there and are deleted once unticked and saved. Node count (4 or 8, when the launcher offers both) picks the count. Check copies the package
  to the head and the ticked workers (tf-glm53 deploy), then runs tf-glm53 check on them (ssh, network, memory, NCCL,
  the files; ✓ or ✗ a node, with the reason); it takes a while. Save, once every node of that list passed, writes
  the nodes to glm53.nodes.conf for the next start, and leaves glm53.conf as it is (a copy of the old node file kept beside it); saving another count moves the lane
  to it. Beside a greyed Save the editor says why. The addresses are the ConnectX-7 ports' on a switch (the ones NCCL uses) and the LAN's on a ring, and
  every bank must be on every node. Until a start on a new count has been measured, the plans are estimates.
- **Start**: press it twice (Start, then Confirm within 8 s). The circle lights each layer as it loads (real: the engine's load
  progress), then the page turns into the live panel.
- **Startup flow**: a screen opened from the header's button (shown while the step bar is, the screens before the
  engine runs): 0 start (`./tf-glm53 up` on the head node) › 1 › 2 › 3 › the live panel, on one page, numbered as the
  step bar, each step tagged with the step bar's own button. Close or Escape goes back.
- **Optimised allocations**: a start loads the table it is given and never re-sorts, so the launch screen has no
  Optimise experts. Optimise while the engine runs (below): the result is what the engine holds at the stop, so the
  launch screen opens with it as the last run; Save keeps it for later starts, and Open brings it back.

## Live panel (engine running)

- **Circle**: what is measured and what is only a look.
  - Decode: the experts the model picks for each token light up and fade (real: the engine's log of the picks).
  - Prefill: the cells' twinkling is only a look. The ring that widens from the inside out marks the layer being
    computed now, a sweep a prompt chunk (estimated from the prompt's progress). The experts a chunk used are not
    drawn: a chunk of thousands of tokens uses nearly every expert a little, and lighting them looked flat.
  - Start: the circle filling as the experts load is real (see Start above).

  While a prompt is prefilled the experts are in number order; once decoding starts they turn to usage order, the
  most used first, from 9 o'clock.
- **Rim**: the prefill's progress while a prompt is computed (real); while decoding, the reply's text runs round the
  rim.
- **Centre and below**: in the centre, bpw (the mean bits of the experts picked) and each width's share of the picks;
  below the circle, prefill and decode tok/s (decode: every request's tokens together, decode total when several run at
  once; prefill when several run: the request the circle follows, over the time its prompt computed, the waits for the
  decode left out, — from an engine that does not give it), and bpw over the last 3 minutes.
- **[ NODE ]**: Health check sends the running engine one request of about 1,000 tokens and judges the reply: Inference OK over the circle's centre, for a few seconds, or Inference error, which stays with its reason under the button until the next press or the engine stops (under the button too while the circle is hidden). Behind a long inference it waits its turn, up to 600 s. Stop, and the servers: each named with head or worker beside it, with its free memory and its GPU and CPU
  temperatures (the CPU's is the hottest of the node's thermal zones: on GB10 the CPU and GPU share one chip, which runs
  6-10 °C above the GPU's own reading under load), read by the launcher every 5 s; "—" where the launcher gives none.
  RDMA (an engine that says where its RoCE proxy threads run): each node's two proxy threads' CPUs, between the nodes
  and the DCP pair (e.g. 0·10), with their kind, A725 or X925, in the cell's title; the line under the table says the
  start setting (ROCE_PROXY_CORES: kept on A725, the default, or left to the OS), and the line below it where the engine
  itself runs (ENGINE_CORES: A725, or OS, the default), for an engine that says so. The head's are where they ran last; the
  other nodes' where they were kept at the start, "—" when left to the OS. Without the launcher the same is a card of
  its own, [ RDMA ].
  Concurrent: the engine's conversations at once and, at 2 or more, the requests in flight.
- **GPU split** (an engine that runs several conversations at once and serves its schedule; otherwise not shown): how
  the GPU's time is shared while a prompt is computed beside conversations that are decoding. The slider goes from
  prefill 100% (the prompts first: the decode stops while a prefill runs) to decode 100% (while a conversation is
  generating, new prompts wait and a running prefill stops), in steps of 5; between, that percentage of the time goes
  to the decode. When only one of them has work, it uses all of the GPU. The value is sent when the slider is let go
  (Applying…, then Applied) and holds for the running requests too; a refused change goes back to the value in use, the
  reason in red. The dot under the slider is the value the engine started with, when the engine gives it. Below, at decode 100%: the
  prompts waiting and the longest wait. Each row's explanation opens from the ⓘ beside its name (as Context length's)
  and shuts with it or Escape; a refusal's reason shows in red without it.
  Priority prompt (short_prompt in the API): a prompt with at most this many new tokens to compute (past the cache, not its length) goes before the
  longer ones whatever the split, and its prefill takes all of the GPU (the decode stops for the time it takes; the
  hint gives no seconds). − / + step through 0, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K; a value can
  be typed (0 to 1,048,576, sent on Enter or leaving the field); 0 turns it off. Overtake a running prefill too (off by
  default): a long prefill running gives way to a priority prompt and goes on from the chunks it finished (it is up to a
  chunk later). Every value follows the engine every second, so a change made elsewhere shows here. An engine started
  without the launcher (no [ NODE ] card) shows the same in a card of its own, [ GPU SPLIT ], with Concurrent in its
  head.
- **[ REASONING CONTROL ]** (under [ NODE ], as the GPU split; the thinking's settings that change while the engine runs; the API and the settings keep their names, `/v1/thinking` and `THINKING_*`):
  Thinking limit, a slider from 5% to 100% (No limit) of max_tokens: at it the engine has the model close
  its thinking and answer in the rest, so a short reply (a harness's max_tokens) keeps room for its answer too; the
  start value shows as a dot. Organize midway Yes (checkpoint) / No;
  Organize every, a slider from 10% to 100% of max_tokens (1,024 tokens at least; the right end 100%, the default: no
  organizing until the limit), greyed without organizing midway and otherwise below the thinking limit; Organize at the close Yes (the share of the
  thinking limit it had, else glm53.conf's, else 100%) / No. Each takes the requests that start after it (Applying…, then Applied;
  refused: back, the reason in red) and is written to glm53.conf by the launcher, so the next start keeps it; a
  request's own thinking budget or summaries come first. On the launch screen (the engine stopped), or with an engine
  that cannot change them while running, the same card saves the values for the next start at once, in glm53.conf (From the next
  start). The engine settings screen sets the same four; either shows the other's change.
- **[ BIT ALLOC ]**: the bits loaded, with the allocation the start used in the head (New · unsaved for a new plan).
  Save keeps them under a name.
  For each width: Experts (their number), Share (of the experts), Use (their share
  of the picks) and GiB. Also Mean bits, Expert size, KV size (tokens and the format, e.g. 837K FP4), the Drafter, MemAvailable and
  Allocated (rank 0).
  The live panel counts this run's picks only; the shipped counts are not shown here.
- **Stop**: Save and stop saves the bits loaded and the run's usage record (the record the run started with plus this
  run's) under the Allocation name, then stops; Stop without saving saves nothing. The name offered is the allocation
  the run started from, saved over (its old version goes to the launcher's trash); another name can be given, and an
  existing one asks for a second press before it is overwritten.

## Optimise experts while running

- Optimise experts in [ BIT ALLOC ]'s menu re-sorts the loaded experts by the run's usage record (the record it started
  with plus this run's picks); each width keeps its count. The most-picked experts get the widest bits, so the mean bits of the picks rise at the same memory. Its panel
  has its own Count (decode, prefill, prefill 0.3 / decode 0.7): each press starts at prefill 0.3 / decode 0.7, and
  another pick estimates again. Before this run has prefill counts, prefill alone is greyed and the default counts
  decode only.
- The estimate gives the experts that change, how long inference stops, and that the conversations' caches are dropped.
  While it shows, it is asked again every 30 s, and Apply sorts by the counts at the press.
  Press Apply: the panel waits for the running request to end, then swaps layer by layer. The result is not saved by
  itself; Save keeps it. A full change (9,756 experts over 75 layers) took 83 to 90 s on four nodes.
- With concurrent inference (2 or more) Apply works as it does without it: it waits until no request runs or waits.

## Engine settings

Hide, right of the switch at the top right, takes the circle off the inference and launch screens and stops drawing it, and the cards stand in columns: beside what was under it, the nodes, then what changes while it runs and the plan (on the launch screen the nodes, the plan, then the rest). Show brings it back, and the lane remembers. The switch at the top right, Inference (Launch while stopped) | Engine settings, opens the settings that need a restart,
in glm53.conf (a launcher without them shows no switch). The settings that change while running stay on the inference
screen: the GPU split, [ REASONING CONTROL ] (its changes written to glm53.conf as they apply) and Optimise experts.

- Each row has its name, an (i) that opens what it acts on, what is set, why, an example and what it goes with (and the
  default), Changed while it differs from glm53.conf, and the key. A value against its rule (the same checks as
  tf-glm53) is red and Save is greyed; a choice that cannot be taken now is greyed and says why when pointed at.
- **KV and context length**: Context length (CONTEXT_LENGTH: one request's longest) and KV size (KV_SIZE: the whole
  KV cache, one pool every conversation shares; a start from the panel takes it once it is set here or typed on the launch
  screen; Keep for the OS, typed there, gives the KV all the room left again) and the KV format.
- **Thinking and answer sizes**: max_tokens, one reply's limit (MAX_TOKENS_SHARE, a slider: a share of the context length, 70% by default,
  for a request that names no max_tokens; the client's own max_tokens comes first, and an older glm53.conf's MAX_TOKENS
  until the slider is saved, which clears it) and the
  thinking's limit as a share of it (THINKING_SHARE; saving it clears an older THINKING_BUDGET). The thinking's limit,
  Organize midway, Organize every (THINKING_FOLD_SHARE) and organizing at the close are tagged Applied at once: while the
  engine runs a save applies them at once (the message says which), and the inference screen's [ REASONING CONTROL ] sets the
  same values. With organizing midway the limits are tagged Virtual: they count the
  tokens written, and the thinking that takes out of the KV still counts, so they may pass the context length. The (i) beside the group's name draws a reply's split from the values set, as shares: the sizes change with each
  prompt. Organize midway (THINKING_FOLD, and the size target as a share of the thinking it organizes, 70% by default, midway and at the close) and organizing at the close (Yes or No, No by default; with Yes, organize after a thinking of N% of the thinking limit or more, counted since the last organizing, 100% to start from: THINKING_FOLD_AT_CLOSE_SHARE, 0 for No) are their own groups. Notice of what is left is Yes or No too (with Yes, every how many thinking tokens, 5,000 to start from: THINKING_NOTICE, 0 for No).
- While the engine runs, the drafter, concurrent inference, KV format, context length, KV size and memory kept show what it took; a row saved since the start has Next start: X beside it. Only a row you change is saved; while stopped they show glm53.conf.
- The weights, the DFlash2 model and the nodes are shown here and changed where they are checked: the launch screen's
  [ WEIGHTS ] (Check and save: saved once every node passes; a failed row says why and nothing is saved) and the node check. The API port changes only while the engine is stopped; the panel's port once the
  launcher starts again. Advanced (development) holds the prompt chunk, SP, the token fixes and
  EXTRA_ENV.
- A change is written as it is made: a choice at once, a box when you leave it or press Enter, a slider when you let go
  (in glm53.conf, with a copy beside it as glm53.conf.bak-<time> at most every ten minutes). A value a rule refuses is not written and says why. Rows a rule joins
  (concurrent inference with DFlash2 and its saved states, for one) are written together once they fit; Revert takes back what is waiting. While the engine runs it
  keeps its values: the rows say Saved · next start. **Restart to apply** asks once, then the stop's own question
  (save this run's usage under a name or not), and the launcher stops the engine and starts it again with the saved
  settings and the same bits.

## Display

The drawer at the bottom (Display):

- **General**: Language (English or Japanese; by default the browser's), Data (Live or Simulation, and Pause), and the
  Memory bar's look.
- **Circle**: Order (Auto by default, from 9 o'clock), Afterglow (Short by default), Unpicked and Picked brightness (6 and
  19 of 24 by default), and Light colour (By bit, or One colour: blue by default).
- **Text and centre**: the rim's Text (shown, still, by default), the Digits (7-segment by default), and the Centre's
  cells.
- **Colours**: the presets (Green · one hue by default; Random picks another).

The settings are kept in one file, `launch/panel-settings.json` in the package's directory (`TF_PANEL_SETTINGS` in
`EXTRA_ENV` moves it), so every browser opens the same panel. Back to defaults (press it twice) drops the display
settings from that file, so every browser of the lane gets the defaults; the language and the nodes stay.
