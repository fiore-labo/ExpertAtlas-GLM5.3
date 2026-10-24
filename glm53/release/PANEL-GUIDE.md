English | [日本語](PANEL-GUIDE.ja.md)

# Expert Atlas: the control panel

The launcher serves the panel at `http://<first node>:<launcher port>/panel`. The page needs no key. When the server has
API keys, open `/panel#key=<key>` once; the page keeps the key for the tab and sends it to the APIs. The page has two
screens: the launch screen while the engine is stopped, and the live panel while it runs. It switches between them by
itself. This release's panel runs on four nodes.

The circle shows the 19,200 routed experts: layers 3 to 77 run from the inside out, 256 experts a ring.

## Launch screen (engine stopped)

- **Plan** ([ BIT ALLOC ]): the bits each expert loads at. Pick a PRESET (Last run, once there is one = the last run's
  allocation as it was; Accuracy, Balanced, Memory, 4 bit, 3 bit), or drag the knobs on the curve to set the ranges of
  5 / 4 / 3 / 2 bit. Before any run the plan is all 3 bit. In a mix the most used experts get the widest bits; Count
  picks the counts to rank by: decode, prefill or Both.
- **Ranked by**: which counts ranked the plan.
  - Shipped (12 domains): the usage measured for this release, shipped in the page. The plan stands on them until this
    install has saved a run.
  - Saved runs: the runs this install saved at a stop. From the first saved run the plan stands on them alone; the
    shipped counts are no longer used.
  - None: no counts at all. A mix then gives every layer the same share of each width.

  The 12 domains: Coding, Agent, Japanese, English, Agent A, Agent B, Coding (ja), Coding (en), Linux ops (ja),
  Linux ops (en), Security (ja), Security (en). Each weighs the same: an expert ranks by the largest share it has of its
  layer in any domain.
- **Banks and No data**: the experts at each width come from Hugging Face `fiore-lab/GLM-5.3-EXL3-Experts-<N>bit`
  (N = 2, 3, 4, 5), each in its own directory on every node, by default `~/models/GLM-5.3-EXL3-Experts-<N>bit`, with
  its manifest. A width whose bank is missing on any node shows No data with the node and what it lacks. It cannot be
  planned, and the presets that need it are greyed out.
- **Memory** ([ LAUNCH ]): each node's memory for the plan (expert, Other weights, KV, Work, Margin) and the verdict:
  Fits, Tight or Does not fit. KV FP4 / FP8 shows the KV size (tokens) each format would allocate; press one to choose it.
- **Context size**: the longest request (prompt and reply) the start allows, in tokens: 131K, 262K, 524K, glm53.conf's
  CONTEXT, or Max (the most the plan leaves room for in the chosen KV format). A size that does not fit is greyed; a
  context the launcher had to cut is shown under the row.
- **Store size**: the tokens kept beyond the context for earlier conversations: glm53.conf's STORE, 100K, 500K, or Max
  (all the KV room the context leaves; while the plan is an estimate, the launcher keeps 2 GiB more). The context comes
  first in the KV room; a store that does not fit is greyed.
- **Keep for the OS**: what each node keeps free after the start, 4, 6, 8 or 12 GiB (glm53.conf's RESERVE_GB, 4 by
  default). The verdict and the engine's own admission use it; more kept leaves less for the KV.
- **Drafter**: MTP, DFlash2 or None, chosen under the KV format; the memory is planned for it. DFlash2 is not shipped
  (its licence): download it and set DRAFTER in glm53.conf. Until every node has it, it is greyed with the reason.
- **Nodes** (in the card's head): the nodes in rank order. The head (rank 0) is the node the launcher runs on. The
  workers' addresses can be changed while the engine is stopped. Check copies the package to the listed nodes
  (tf-glm53 deploy), then runs tf-glm53 check on them (ssh, network, memory, NCCL, the files; ✓ or ✗ a node, with the
  reason); it takes a while. Save, once every node of that list passed, writes NODES to glm53.conf for the next start.
- **Start**: press it twice (Start, then Confirm). The circle lights each layer as it loads, then the page turns into
  the live panel.
- **Optimise experts** (under Allocation): re-sorts the last run's allocation by that run's picks, keeping each width's
  count. The estimate shows the experts that change on the circle, in their new colours. Press Apply to save the
  result as `sort-YYYYMMDD-HHMMSS` and make it the plan for the next start. It needs a last run: on a new install the
  button stays off until a run is saved.

## Live panel (engine running)

- **Circle**: the experts each token picks light up. While a prompt is prefilled the experts are in number order; once
  decoding starts they turn to usage order, the most used first, from 9 o'clock.
- **Rim**: the prefill's progress while a prompt is computed; while decoding, the reply's text runs round the rim.
- **Centre and below**: in the centre, bpw (the mean bits of the experts picked) and each width's share of the picks;
  below the circle, prefill and decode tok/s, and bpw over the last 3 minutes.
- **[ BIT ALLOC ]**: the bits loaded. For each width: Experts (their number), Share (of the experts), Use (their share
  of the picks) and GiB. Also Mean bits, Expert size, KV size (GiB and tokens), the Drafter, MemAvailable and
  Allocated (rank 0).
  The live panel counts this run's picks only; the shipped counts are not shown here.
- **Stop**: asks whether to add this run's usage to the learning data: Save and stop, or Stop without saving.

## Optimise experts while running

- Allocation > Optimise experts re-sorts the loaded experts by this run's picks; each width keeps its count. The
  most-picked experts get the widest bits, so the mean bits of the picks rise at the same memory.
- The estimate gives the experts that change, how long inference stops, and that the conversations' caches are dropped.
  Press Apply: the panel waits for the running request to end, then swaps layer by layer and saves the allocation.
  A full change (9,756 experts over 75 layers) took 83 to 90 s on four nodes.
- An engine run with `TF_GLM53_MULTI` changes no bits while it runs. It saves the allocation instead; stop the engine
  and run Optimise experts on the launch screen, which gives the same result for the next start.

## Display

The drawer at the bottom (Display):

- **General**: Language (English or Japanese; by default the browser's), Data (Live or Simulation, and Pause), and the
  Memory bar's look.
- **Circle**: Order (Auto by default, from 9 o'clock), Afterglow, and Light colour (By bit, or One colour).
- **Text and centre**: the rim's Text, the Digits, and the Centre's cells.
- **Colours**: the presets (Green · one hue by default; Random picks another).

The settings are kept in one file, `launch/panel-settings.json` in the package's directory (`TF_PANEL_SETTINGS` in
`EXTRA_ENV` moves it), so every browser opens the same panel.
