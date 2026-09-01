# Combined-source sensitivity audit

Status: post-hoc diagnostic, specified after the 11-source packing result.

The combined source analysis treats dataset identity as the unit and reports
equal-source means.  This audit asks whether its three favorable averages are
carried by one unusually helpful source.  For each comparison, report:

- the minimum and maximum individual-source percentage reduction;
- the median and 20% trimmed equal-source mean;
- all eleven leave-one-source-out equal-source means, their range, and the
  source whose deletion produces the smallest remaining mean.

There is no additional hypothesis-test or performance gate.  The diagnostic
passes only in the descriptive sense if the minimum individual effect, trimmed
mean, and every leave-one-source-out mean remain strictly positive.  Because
this audit was conceived after seeing the combined result, it cannot strengthen
the confirmatory evidence grade; it can only expose source concentration.
