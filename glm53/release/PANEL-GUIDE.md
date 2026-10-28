English | [日本語](PANEL-GUIDE.ja.md)

# Expert Atlas: the control panel

The launcher serves the panel at `http://<first node>:<launcher port>/panel`, on every address by default and with no
key: anyone who can reach the panel can start, stop and change the engine, so keep the nodes on a closed network (or set
`LAUNCHER_HOST=127.0.0.1` and open it from another machine through an ssh tunnel, `ssh -L 8010:127.0.0.1:8010
<first node>`, then `http://127.0.0.1:8010/panel` there). The page has two
screens: the launch screen while the engine is stopped, and the live panel while it runs. It switches between them by
itself. This release's panel runs on four or eight nodes (the count is picked in Nodes on the launch screen).

The circle shows the 19,200 routed experts: layers 3 to 77 run from the inside out, 256 experts a ring.

## Launch screen (engine stopped)

- **Plan** ([ BIT ALLOC ]): the bits each expert loads at. The screen opens with a new plan at the last run's shares
  (before any run: all 3 bit); Open picks the last run's allocation as it was, or a saved one. The head of the card names the plan: the last run, a saved allocation's name, or
  New · unsaved.
  - **Shares**: type each width's share of the experts in the table (5, 4 and 3 bit; the narrowest width with data
    takes the rest, so they add up to 100 %), or drag the knobs on the curve. The scale counts from the 5 bit end,
    which is 100 %: 5 bit 10 % is the range 100–90 %. In a mix the most used experts get the widest bits, ranked by
    prefill 0.3 / decode 0.7 (an expert's decode share weighed 0.7, its prefill share 0.3; `ATLAS-TESTS.md`).
  - **Count** (decode, prefill, prefill 0.3 / decode 0.7): for reading only. The curve, the knobs' readouts (what share
    of the picks the experts up to a point carry), the circle's order and Use follow the counts picked; the plan's bits
    do not change. When the plan's bits follow other counts than the ones shown (an opened allocation, or Count other
    than prefill 0.3 / decode 0.7), the regions are faded and marked Regions: a guide; the strip under the curve shows
    the real bits.
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
- **Banks and No data**: the experts at each width come from Hugging Face `fiore-lab/GLM-5.3-EXL3-Experts-<N>bit`
  (N = 2, 3, 4, 5), each in its own directory on every node, by default `~/models/GLM-5.3-EXL3-Experts-<N>bit`, with
  its manifest. A width whose bank is missing on any node shows No data with the node and what it lacks. It cannot be
  planned: its row in the table shows No data and takes no share.
- **Memory** ([ LAUNCH ]): each node's memory for the plan (expert, Other weights, KV, Work, Margin) and the verdict:
  Fits, Tight or Does not fit. KV FP4 / FP8 shows the KV size each format would allocate (tokens, and GiB a node); press
  one to choose it. The KV always takes all the room the plan leaves: Max model len for one conversation, the rest for
  earlier conversations. Keep for the OS sets how much each node keeps free.
- **Max model len**: the longest request (prompt and reply) the start allows, in tokens: 131K, 262K, 524K, glm53.conf's
  CONTEXT, or Max (the most the plan leaves room for in the chosen KV format, less what the engine needs beyond it: it
  is never the whole KV size, the pool cache keeping at least 512 tokens). A size that does not fit is greyed; a
  context the launcher had to cut is shown under the row. Max is the default. The launcher finds Max from the memory a
  start with that drafter took on this lane (measured each start, a drafter at a time); until then it plans on built-in
  sizes with 2 GiB more kept, so Max is shorter (est.); the KV rows say est. or Measured to the left of their size. The measurement is taken just after the start, with nothing
  running, so at Max a node can dip a little under what it keeps for the OS while requests run. The i beside the name
  says this and whether the drafter picked is measured.
- **Keep for the OS**: what each node keeps free after the start, 4, 6, 8 or 12 GiB (glm53.conf's RESERVE_GB, 4 by
  default). The verdict and the engine's own admission use it; more kept leaves less for the KV.
- **Drafter**: MTP, DFlash2 or None, chosen under the KV format; the memory is planned for it. DFlash2 is not shipped
  (its licence): download it and set DRAFTER in glm53.conf. Until every node has it, it is greyed with the reason.
- **Nodes** (in the card's head, while the engine is stopped): the head (rank 0) is the node the launcher runs on.
  Workers are registered once, whatever the node count (an address, Add), and each count's workers are ticked from
  the list (Use 3 / 3 at 4 nodes, 7 / 7 at 8); the ranks follow the list's order. The list and each count's ticks are
  kept in the lane's panel settings, so they come back on any browser. Delete takes a worker off the list (and off
  every count's ticks); the workers the lane uses now are always there and are deleted once unticked and saved. Node count (4 or 8, when the launcher offers both) picks the count. Check copies the package
  to the head and the ticked workers (tf-glm53 deploy), then runs tf-glm53 check on them (ssh, network, memory, NCCL,
  the files; ✓ or ✗ a node, with the reason); it takes a while. Save, once every node of that list passed, writes
  NODES to glm53.conf for the next start (a copy of the old file kept beside it); saving another count moves the lane
  to it. Beside a greyed Save the editor says why. The addresses are the ConnectX-7 ports' (the ones NCCL uses), and
  every bank must be on every node. Until a start on a new count has been measured, the plans are estimates.
- **Start**: press it twice (Start, then Confirm within 8 s). The circle lights each layer as it loads (real: the engine's load
  progress), then the page turns into the live panel.
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
  below the circle, prefill and decode tok/s, and bpw over the last 3 minutes.
- **[ LAUNCH ]**: Stop, and the servers: head (rank 0) and the workers, each with its free memory and its GPU and CPU
  temperatures (the CPU's is the hottest of the node's thermal zones: on GB10 the CPU and GPU share one chip, which runs
  6-10 °C above the GPU's own reading under load), read by the launcher every 5 s; "—" where the launcher gives none.
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
- An engine run with `TF_GLM53_MULTI` changes no bits while it runs. There Optimise experts gives the estimate with
  Save for a start in place of Apply: the estimate's bits are saved under a name (`opt-<time>`), inference going on.
  Stop the engine, Open that allocation and Start.

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
