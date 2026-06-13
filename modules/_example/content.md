---
module: _example
title: Example module — KNN snippet
version: 0.1
sources:
  - (none — toolchain demo)
outcomes:
  - Verify the make pipeline end to end
---

## 1. K-Nearest Neighbours {#sec-knn}

To classify a new point, look at the $k$ closest labelled points and take a
majority vote. Distance does all the work here: "closest" presumes a meaningful
metric on the features.

::: example
A bank wants to flag risky loans. A new applicant is compared to the $k=5$
most similar past applicants; if 4 of 5 defaulted, the application is flagged.
:::

::: student
**Why scaling matters.** With Euclidean distance
$d(x, x') = \sqrt{\sum_j (x_j - x'_j)^2}$, a feature measured in grams instead
of kilograms contributes $10^6 \times$ more to the sum, so it silently
dominates the vote. Standardize features before using distances.
:::

::: deep
**For the interested reader.** In high dimensions, distances concentrate: the
ratio of nearest to farthest neighbour distance tends to 1, one face of the
curse of dimensionality.
:::

::: widget
id: knn-decision-boundary
goal: feel how k changes the decision boundary
controls: k (slider 1–50), dataset (two-moons | blobs)
shows: scatter plot with shaded decision regions, live
:::

::: animation
id: knn-voting
goal: animate the k nearest neighbours being found and voting
style: manim, 20–30 seconds
:::

Choosing $k$ trades off noise against blur: small $k$ memorizes, large $k$
oversmooths.

::: build
id: knn-from-scratch
task: implement a KNN predict() with NumPy broadcasting; no sklearn.
given: train/test split helper and a distance function signature.
check: accuracy within 1% of the sklearn reference on the same split.
:::
