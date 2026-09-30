# Why We Re-ran the Statistics with a "Mixed-Effects Model"

## The reviewer's concern, in one sentence

We have data from ~19 mice, but thousands of individual brain cells (neurons) recorded across them — and our original statistics treated every single neuron as if it were its own completely independent piece of evidence, which overstates how certain we can be.

## Why a normal regression can be misleading here

A standard linear regression (or a correlation, which is closely related) assumes every data point is independent — like asking a different, unrelated person the same question and averaging their answers. The more independent people you ask, the more confident you can be in the average.

Our data doesn't work like that. Every neuron in a given mouse shares that mouse's specific brain state, behavior, and imaging session that day. Measuring 500 neurons from one mouse is much closer to asking *the same person* the same question 500 times than to asking 500 different people. The 500 answers will look consistent with each other, but they don't give you 500 independent opinions — they give you one opinion, sampled 500 times.

When you feed that into an ordinary regression anyway, the math has no way of knowing some of those data points are "repeats" from the same source. It counts every neuron as a fresh, independent vote of confidence — which makes the result look far more statistically certain than it really is (this is why our original p-value was an almost impossibly tiny number).

## What a "linear mixed-effects model" changes

It's the same regression as before, plus one addition: instead of forcing every neuron in the dataset to share one single starting point, we let **each mouse have its own personal baseline** — its own "starting line" — before measuring the slope of interest (how participation rate relates to LMI, or how it changes across days).

That per-mouse starting-line adjustment is what's meant by adding "mouse as a random effect" (or "random intercept for mouse"). Concretely:

- **Fixed effect** (what we actually care about and report): the one overall slope — does participation rate go up with LMI, does it rise across days — that we believe applies across mice in general.
- **Random effect** (the correction): each mouse is allowed to sit a bit higher or lower than average, and that mouse-specific quirk gets absorbed separately, instead of being mistaken for the effect we're testing.

You could, in principle, add "mouse" to a normal regression as an ordinary predictor too — but that treats each of our 19 mice as a fixed, specific thing we're studying for its own sake, which isn't what we want. We want to say something about the *general* relationship across mice, while still being honest that mice differ somewhat from each other. Treating mouse as a random effect is the standard way to say "some of the spread in our data is just mouse-to-mouse variability, not signal" without trying to draw a separate conclusion about each individual animal.

## Why this directly addresses the reviewer's concern

The reviewer's worry was, essentially: "you're analyzing this as if you had thousands of independent samples, when you really only have ~19 independent animals." Giving each mouse its own baseline is exactly how the model is told not to treat neurons from the same mouse as being as informative as neurons from different mice.

Once that correction is in place:

| | Original approach | Mixed-effects model |
|---|---|---|
| Treats each neuron as... | An independent, unrelated observation | Part of a cluster that shares its mouse's baseline |
| "Real" sample size the math trusts | Thousands of neurons | ~19 mice |
| Result | Overconfident — artificially tiny p-values | Honest — appropriately larger p-values and wider uncertainty |
| Underlying effect / direction | Same data, same relationship | Same data, same relationship |

## Bottom line

We're not changing what we found — the underlying relationship in the data is the same. We're changing *how confident we're allowed to claim to be about it*, so that the confidence we report is no longer inflated by silently counting the same mouse's neurons over and over as if they were independent evidence. This is precisely the correction the reviewer asked for.
