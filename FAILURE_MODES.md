# FAILURE_MODES

What this page does not prove.

The README and `STATUS.md` are a reading guide for a notebook. They are not a result, a submission, or a certificate that the training code still runs.

## Not proved

- **Not a product and not a completed system.** There is no deployed monitor, no API, and no claim that a general learner has been built.
- **Not peer review.** Cover letters, venue names, and "camera-ready" lines are drafts. Reviewer-simulator files are not referee reports. `PUBLICATION_HOLD.md` disagrees with itself about whether a hold is active. Neither section is an acceptance or a rejection from a venue.
- **Not an independent replication.** Every verified row was recomputed from JSON the same project committed. Recomputation checks arithmetic. It does not re-run PPO, MADDPG, or the language-model pilot.
- **Not a significant reversal on H10.** H10 stays falsified because the success rule was missed. The committed intervals include zero on the frozen-minus-joint contrast. A null interval is not evidence that joint training is better.
- **Not a transfer result for the monitor.** The multi-agent monitor pathways stay falsified in the lab note. That note is not re-verified here, and its opening paragraphs disagree about the DLR-only arm. The DLR-only n = 100 figure is a markdown aggregation without per-seed JSON, on one PettingZoo task, and the committed three-seed file is not significant.
- **Not the single-agent headline in the drafts.** t = 6.76 and p < 0.001 are not in the committed seed file. The pre-registered monitor-versus-random test was not supported (Welch t = 0.781). A gain of reward shaping over an unshaped PPO baseline is a different statement, and the Welch test against that baseline is t = 1.94 on unequal sample sizes (15 versus 5).
- **Not a multi-environment AUROC result.** The frozen-versus-joint table is one environment, five seeds, and the checkpoint files are absent. One frozen seed in that table is below chance (0.21).
- **Not the H3 accuracy table.** `h3_500k_summary.json` is a reward comparison at 500k steps in which the real monitor is behind the random control. It does not confirm a 97% predicate-accuracy claim.
- **Not a green test suite for the science.** `experiments_log/_data_audit.py` prints which of 12 documents contain selected strings and exits 0 if a string is missing. `tests/test_front_door_honesty.py` checks this page and the JSON arithmetic (18 checks). It does not train a policy. The four paper PDF jobs are not on the push path.

## How a number on this page can still be wrong

- The JSON can be a summary written by the same process that wrote the paper. Per-seed training logs for the H10 and Y1.3 files were not re-derived from raw rollout archives in this pass.
- Seeds can be dependent, environments can be single, and a bootstrap p-value in a summary file was not recomputed from the raw pairs. Cohen's d was recomputed from the stored mean and standard deviation of the paired differences only.
- Later notes can contradict earlier ones. When they do, the later pre-registered verdict is the one this page uses, and the contradiction is listed rather than averaged away.
