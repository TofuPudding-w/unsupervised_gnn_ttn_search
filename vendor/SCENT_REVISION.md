# Upstream provenance

`treetensorcomp/` is the unmodified GitLab archive of revision
`f05c28fbf959ecc023fd4ed89d1e00691d5ca564`.
Source: https://gitlab.com/gohmat/treetensorcomp
License: AGPL-3.0-only; see treetensorcomp/LICENSE.txt.

Stage A calls its original recursive spectral partition function with an exact
small-state mutual-information affinity calculated by our adapter. This is the
SCENT mutual-information variant with exact affinity, not a reproduction of the
paper's Monte Carlo/projector-entropy defaults or approximate compiler.
Upstream TCI is not executed in this environment; its older SciPy requirement
has not been validated here. Do not infer TCI compatibility from stage A tests.
