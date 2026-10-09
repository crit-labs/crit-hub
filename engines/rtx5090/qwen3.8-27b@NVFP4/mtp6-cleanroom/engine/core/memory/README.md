# Resource accounting

[resources.cpp](resources.cpp) implements ResourceBudget and reservation
accounting shared by runtime and backends. It defines resource ownership rather
than a model's tensor layout. Public declarations are in
[core.h](../include/crit/core.h); allocation services are in
[the CUDA backend](../backends/cuda/README.md).
