# Data provenance and limits

The included CSV files are experiment metrics and selected result tables, not
the MNIST images or trained model checkpoints.

The compressed `all_runs.csv.gz` was produced from the three completed
controlled-ridge seed archives using
`experiments/analysis/analyze_24gpu_archives.py`. The original archives contain
per-job configurations, histories, and logs and are not included in this
compact Git repository. The table has 174,960 successful rows, covering the
square, sine, tanh, and SiLU target profiles at dimensions 4, 16, and 64.
`run_all.py` consumes this table directly, without recomputing training
results. The archive analyzer is included for users who possess the archives.
The uncompressed CSV has SHA-256
`543333c58ff08356056a62cc369197d4804b6b24b33153c8f4a46c281f1a21c8`;
the committed gzip file has SHA-256
`4a04bd602511c9cd3ada8897f5a2bf32e1bb23261939bf820c4c8c066c9d9584`.

The MNIST tables under `experiments/data/raw/mnist_n60000` and
`experiments/data/raw/mnist_n6000` contain the recorded normal-label and
random-label scans. The sine-profile CSVs under
`experiments/data/raw/sine_ridge_v2` are the older three-seed subset used for
the appendix activation and training-probe diagnostics. The four constructive
CSV files under `experiments/diagnostics/constructive_example2/results` are
inputs to the main-text scaling panel.

Rebuilding a figure from these released tables verifies the analysis and
plotting stage. Reproducing the training stage additionally requires the
specified software versions, compute resources, and input data; stochastic
runs can differ across hardware and software stacks. The release validation
script checks selected invariants, not every scientific claim in the paper.

The historical MNIST runner snapshots used on the servers differed in some
implementation and optimizer-selection details from the maintained runners
included here. Do not treat a fresh run of the maintained scripts as a
byte-for-byte reproduction of those archived results. The committed result
tables are the provenance for the plotted paper values.

The authors release their original code and generated metric tables here under
the repository's MIT license. This does not change the licenses of PyKAN,
PyTorch, MNIST, or any other upstream work.
