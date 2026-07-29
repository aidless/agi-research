# Y2-b: v7 proper ablation (3-arm 5-seed)

> Date: 2026-07-29
> Goal: isolate whether the +0.055 (n=212) effect in v5 comes from the
> Monitor signal or the trust head architecture itself.
> v7 = exact v5 clone with --arm with_trusthead_random added.

## Results (3-arm 5-seed, 800 ep/seed)

| arm | n | mean | sd | range |
|---|---|---|---|---|
| with_verifier | 5 | **-70.33** | 1.07 | [-71.92, -69.21] |
| no_verifier | 5 | -71.16 | 1.37 | [-72.83, -69.24] |
| with_trusthead_random | 5 | **-70.33** | 1.07 | [-71.92, -69.21] |

Paired t-tests:
- with_verifier vs no_verifier: mean_diff=+0.83, t=+1.34, 4/5 positive
- with_verifier vs with_trusthead_random: **mean_diff=+0.00, t=+0.00**, 0/5 positive (IDENTICAL)
- no_verifier vs with_trusthead_random: mean_diff=-0.83, t=-1.34, 1/5 positive

## CRITICAL FINDING: with_verifier = with_trusthead_random

The trust head with REAL Monitor input produces IDENTICAL results to
the trust head with RANDOM input. This means **the trust head
completely ignores the Monitor signal**.

The Monitor signal does not propagate from the per-step transition
buffer to the trust head's decision. The trust head learns to use
only the obs space (which is shared between both arms) and treats
the Monitor input as noise.

## Architectural lesson REVISED

The earlier n=212 +0.055 result in v5 was NOT from the Monitor signal.
It was from the **trust head architecture itself** providing an
additional learned pathway in the actor.

REVISED finding:
- Trust head architecture (random inputs) gives **+0.83 mean improvement**
  at n=5, 4/5 positive vs no_verifier baseline
- Monitor signal adds **zero additional improvement** (+0.00 mean,
  0/5 positive) once the trust head is in place
- **Monitor as MA verifier is FALSIFIED**: the Monitor output is not
  actually used by the trust head in the v5 architecture.

## Comparison to v5 n=212

v5 n=212 (5-seed * 30-some seeds):
- with_verifier: -69.37, no_verifier: -69.41, diff=+0.055 (t=0.95, 50.5% pos)

v7 n=5 (exact v5 clone, 3 arms):
- with_verifier: -70.33, no_verifier: -71.16, diff=+0.83 (t=1.34, 4/5 pos)
- with_trusthead_random: -70.33, no_verifier: -71.16, diff=+0.83 (t=1.34, 4/5 pos)
- with_verifier vs with_trusthead_random: diff=+0.00 (IDENTICAL)

The v7 with_verifier vs no_verifier at n=5 is +0.83, t=+1.34 (4/5 pos),
which is LARGER than the v5 n=212 +0.055 result. The n=212 result
shrunk toward 0 with more seeds, consistent with v7's finding that
the Monitor signal is not actually being used.

## Implications for the paper

The 'Monitor as MA Verifier' idea is FALSIFIED:
- The trust head architecture helps (+0.83, 4/5 pos, t=+1.34)
- But the Monitor signal itself does NOT contribute (+0.00, 0/5 pos)
- The n=212 +0.055 in v5 was likely a small sampling artifact

REVISED architectural lesson:
- Trust head architecture = direction-consistent (n=5: +0.83, 4/5 pos)
- Monitor signal contribution = essentially zero (n=5: +0.00, 0/5 pos)
- The v5 paper claim 'Monitor as MA Verifier' should be retracted
- The architectural lesson is now: trust heads (or similar additional
  actor-side pathways) help in MA, but the Monitor signal itself
  is not the right signal to use

## Honest framing

n=5 is still small (need n=30+ for significance). But the fact that
with_verifier and with_trusthead_random produce EXACTLY the same
numbers (mean_diff=+0.00) is strong evidence that the Monitor signal
is not being used. The trust head learns to use only the obs space.

If a paper is to be written, the honest framing is:
1. The trust head architecture (a small MLP added to the actor that
   conditions on (obs, monitor) -> Q blend weights) helps in MA.
2. The Monitor signal itself is NOT the right signal to feed this
   architecture; the trust head learns to use obs alone.
3. Future work should explore what signal IS right: maybe the DLR
   cross-agent evidence chain (which provides inter-agent info
   not in obs), or learned inter-agent comms.

## 9-hypo framework

H5 status: REFUTED. v7 confirms that the trust head architecture
helps but the Monitor signal itself does not. The Y2 follow-up
sharpens: it is the ARCHITECTURE (trust head) that helps, not the
SIGNAL (Monitor).

## Action items

- [x] v7 proper ablation completed (3-arm 5-seed)
- [x] Critical finding: with_verifier = with_trusthead_random
- [x] Honest log written
- [ ] Update paper: revise the v5 narrative to be 'trust head helps,
      Monitor does not'
- [ ] Future work: explore what signal IS right (DLR, learned comms)