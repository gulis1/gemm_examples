# gemm_examples

Varios códigos para multiplicar matrices, usando cada una diferentes formatos de precisión.

## Compilación

Antes de usar *make*, conviene especificar la arquitectura de la GPU que tengamos.

|  Fermi | Kepler | Maxwell | Pascal | Volta | Turing | Ampere |
|--------|--------|---------|--------|-------|--------|--------|
| sm_20  | sm_30  | sm_50   | sm_60  | sm_70 | sm_75  | sm_80  |
|        | sm_35  | sm_52   | sm_61  | sm_72 |        | sm_86  |
|        | sm_37  | sm_53   | sm_62  |       |        | sm_87  |

Los tensor cores solo están dispoibles en las arquitecutras Volta y superires.
Para más info: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
