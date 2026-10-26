# Expert Atlas: which picks to re-sort the experts by

The control panel's **Expert optimisation** re-sorts the routed experts between the widths an allocation holds: the
experts the model picks most get the widest bits, and the memory stays the same. The ranking can come from the
experts picked while a prompt is read (prefill), from those picked while tokens are generated (decode), or from a
mix of the two. This page reports each choice's improvement (how much of the 3-bit loss the re-sort removes), measured on DGX Sparks, on
documents of 8,192 tokens and on prompts of 128K and 256K tokens.

## Summary

- **Prefill and decode use different experts.** Over 174 prompts, a layer's 64 most picked experts in prefill and
  its 64 most picked in decode have 42 % in common, where a random choice would have 25 %. The deeper the layer,
  the more the two lists differ. One prompt's prefill and decode picks are less alike than the prefill picks of two
  prompts from different kinds of work.
- **Weighting the decode picks helps the decode positions (the model's reply), up to a weight of about 0.7.** At
  8,192 tokens the improvement was 25 % ranking by the decode picks and 13 % ranking by the prefill picks. At
  128K and 256K tokens a decode weight of 0.7 gave 34 % and 32 %, the prefill picks alone 28 % and 21 %; from
  0.5 on the gains level off.
- **The prefill positions (the prompt) hardly depend on the weight.** Every weight's improvement was 15 to 21 %
  at 8,192 tokens, 15 to 25 % at 128K tokens but for one low value, and 21 to 24 % at 256K tokens; the differences
  between weights were within the measurement's spread.
- **The panel's Both therefore weighs an expert's decode share 0.7 and its prefill share 0.3.** decode and prefill
  rank by one source alone.

## KL with and without the re-sort: four DGX Sparks, half 3 / half 4 bits, weight 0.7

The KL divergence from the model with every routed expert at 4 bits (that model's own value is 0), the mean over the
positions scored, before the re-sort ("no optimisation") and after it with the panel's Both (*w* = 0.7).
Improvement = 1 − KL(*w* = 0.7) / KL(no optimisation): the share of the 3-bit loss the re-sort removes.

| Prompt | Positions | No optimisation | *w* = 0.7 | Improvement |
|---|---|---|---|---|
| 8,192 tokens | decode | 0.02443 | 0.01848 | 24.4 % |
| 8,192 tokens | prefill ¹ | 0.02352 | 0.01990 | 15.4 % |
| 128K tokens | decode ² | 0.01364 | 0.00899 | 34.1 % |
| 128K tokens | prefill ² | 0.01149 | 0.00964 | 16.1 % |
| 256K tokens | decode ² | 0.01401 | 0.00949 | 32.3 % |
| 256K tokens | prefill ² | 0.00412 | 0.00321 | 22.1 % |

¹ Without the 1 % of positions where the KL with no optimisation is largest; the plain mean is 0.04273 and 0.04384
(−2.6 %), set by a few dozen positions.
² Without each position that is in the largest 1 % of any allocation's KL (101 to 135 positions). The plain means:

| Prompt | Positions | No optimisation | *w* = 0.7 | Improvement |
|---|---|---|---|---|
| 128K tokens | decode | 0.02221 | 0.01389 | 37.5 % |
| 128K tokens | prefill | 0.06075 | 0.04605 | 24.2 % |
| 256K tokens | decode | 0.02171 | 0.01423 | 34.5 % |
| 256K tokens | prefill | 0.01635 | 0.01284 | 21.5 % |

The shares in the results below are computed over blocks of 32 positions (for their 95 % intervals) and differ from
these by up to two points.

## How it was measured

- **Hardware and weights.** DGX Sparks, this release. The routed experts come from the 3-bit and 4-bit banks, half
  at each width (9,600 + 9,600 experts, 3.5 bits on average); the other weights are those of the eight-node
  checkpoint.
- **The allocations.** One allocation is made per weight *w*. An expert's score is *w* × its share of its layer's
  decode picks + (1 − *w*) × its share of its layer's prefill picks. Every layer is ranked together, the way the
  launcher sorts, and the highest-scoring experts get 4 bits. So *w* = 0 ranks by the prefill picks, *w* = 1 by the
  decode picks, and *w* = 0.7 is the panel's Both. "No optimisation" is the start allocation: the widths alternate
  within each layer.
- **What the rankings learned from.** The picks of 162 prompts of 12 kinds of work: agent conversations, coding,
  Japanese and English text, Linux operations and server security, in Japanese and in English, 500 generated tokens
  each. None of the test texts below is among them.
- **The measure.** The KL divergence of the next-token distribution from that of the same model with every routed
  expert at 4 bits, position by position, the text teacher-forced. The scoring configuration is deterministic:
  prompt chunks of 2,048 rows, no prompt-expert kernels and an FP8 KV cache, so the same allocation gives the same
  logits run after run. The improvement is 1 − KL(allocation) / KL(no optimisation): the share of the loss of the 3-bit
  experts that the re-sort removes. 95 % intervals come from resampling blocks of 32 positions.
- **Prefill and decode positions.** Prefill positions predict the tokens of a document given to the model (the text
  prefill reads). Decode positions predict the tokens of the model's own reply to that document (its reasoning,
  sampled at temperature 0.6; the text decode writes), teacher-forced after the document. Both are scored the same
  way; the names say which text a position is in. The whole document is computed with the allocation too, so the
  decode positions also carry what reading it with fewer bits cost.
- **No chat template at the start of a window.** Inside the template's user turn the model predicts the user's own
  text poorly at any length (in a check, its most likely token matched the text at about a quarter of the
  positions; without the template's opening, at 96 to 98 %). The windows therefore start inside the document, and a
  prefill position measures how the model follows the document itself.

## Results: documents of 8,192 tokens, 12 kinds of work

Four DGX Sparks (TP=4), reference included. Twelve held-out prompts, one of each kind of work, each giving two
windows of 8,192 tokens. A document window is the prompt's last 8,192 tokens; a reply window is the prompt's last
7,168 tokens followed by the first 1,024 tokens of the model's reply. Each window's last 1,023 positions are scored,
12,276 positions of each kind.

| Allocation | Decode: improvement [95 %] | Prefill: improvement [95 %] |
|---|---|---|
| *w* = 0 (prefill) | 13 % [7, 18] | 17 % [1, 25] |
| *w* = 0.2 | 19 % [14, 22] | 21 % [3, 28] |
| *w* = 0.3 | 21 % [16, 25] | 17 % [−2, 27] |
| *w* = 0.4 | 20 % [16, 22] | 19 % [−4, 28] |
| *w* = 0.5 | 19 % [14, 23] | 19 % [5, 26] |
| *w* = 0.7 (Both) | 24 % [20, 28] | 15 % [−4, 23] |
| *w* = 1 (decode) | 25 % [22, 29] | 16 % [3, 26] |

KL with no optimisation: prefill positions 0.0427 (median 0.0016), decode positions 0.0244 (median 0.0044). The
prefill column leaves out the worst 1 % of positions. Those positions hold about half of the prefill positions' KL
and vary too much from allocation to allocation for the plain mean to compare.

## Results: prompts of 128K and 256K tokens, coding and Linux operations

The reference (every routed expert at 4 bits) ran on eight DGX Sparks (TP=8), the allocations on four (TP=4). Four
held-out prompts of each size: two of coding (source files and headers of a language's standard library), two of
Linux operations (manual pages, a system journal and configuration files), each ending with one task. The model's
reply to each was sampled at temperature 0.6. A window is the prompt without the chat template's opening tokens,
followed by the reply's first 1,024 tokens: 131,062 tokens (128K) and 262,134 tokens (256K). Each window's last 2,047
positions are scored, about 1,023 prefill and 1,024 decode, so 4,092 to 4,098 positions of each kind per size.

Both kinds of position leave out, for every allocation alike, each position that is in the worst 1 % of any
allocation's KL (the start's included): 101 to 135 positions of each kind and size. The plain mean of the prefill
positions is set by a few dozen positions and swings by tens of percent between allocations.

**Decode positions: improvement [95 %]**

| Allocation | 128K | 256K |
|---|---|---|
| *w* = 0 (prefill) | 28 % [21, 33] | 21 % [17, 26] |
| *w* = 0.3 | 31 % [24, 37] | 29 % [25, 33] |
| *w* = 0.5 | 33 % [27, 38] | 30 % [25, 35] |
| *w* = 0.7 (Both) | 34 % [29, 40] | 32 % [28, 36] |
| *w* = 1 (decode) | 33 % [27, 39] | 33 % [29, 36] |

**Prefill positions: improvement [95 %]**

| Allocation | 128K | 256K |
|---|---|---|
| *w* = 0 (prefill) | 24 % [12, 35] | 24 % [11, 35] |
| *w* = 0.3 | −2 % [−19, 12] | 21 % [6, 33] |
| *w* = 0.5 | 25 % [14, 34] | 23 % [11, 33] |
| *w* = 0.7 (Both) | 15 % [1, 27] | 24 % [13, 33] |
| *w* = 1 (decode) | 15 % [0, 28] | 24 % [10, 35] |

**Positions where the most likely token differs from the reference's** (decode / prefill)

| Allocation | 128K | 256K |
|---|---|---|
| No optimisation | 3.8 % / 2.3 % | 4.0 % / 1.2 % |
| *w* = 0 (prefill) | 3.1 % / 1.9 % | 3.4 % / 1.0 % |
| *w* = 0.3 | 3.3 % / 2.3 % | 3.4 % / 1.2 % |
| *w* = 0.5 | 3.1 % / 2.2 % | 3.2 % / 0.9 % |
| *w* = 0.7 (Both) | 3.0 % / 1.9 % | 3.2 % / 1.0 % |
| *w* = 1 (decode) | 3.0 % / 2.2 % | 3.3 % / 1.2 % |

On the decode positions the two kinds of work differ: coding gains with the decode weight all the way (128K: 13 % at
*w* = 0, 18 % at 0.5, 27 % at 1; 256K: 17 %, 28 %, 30 %), Linux operations level off from 0.3 (128K: 35 %, 40 %,
37 %; 256K: 25 %, 32 %, 33 %). The single low prefill value (*w* = 0.3 at 128K) is not seen at 256K or at 8,192
tokens.

## Notes

- The reference is the model with every routed expert at 4 bits, not the BF16 model. The loss measured is the loss
  of the 3-bit experts against the 4-bit ones.
- At 128K and 256K the reference ran on eight nodes and the allocations on four. Whatever the node count adds to
  the KL is in every allocation's value and in the start's, and was not measured on its own.
- The long prompts are four of each size, of two kinds of work. The replies are reasoning text; final answers and
  tool calls were not measured on their own.
- A re-sort learns from the picks it is given. Measured offline on the picks themselves, it captured less of the
  possible gain on kinds of work that were not in its learning data: about two thirds of what it captured on work
  that was.
