# MedErr-CT question–answer files

| file | task | items |
|---|---|---|
| classification.json | error classification | 15,094 |
| detection.json | error detection | 13,276 |
| correction.json | error correction | 11,105 |
| exclusion_list.csv | distorted reports removed by the single-edit check | 609 |

Each JSON file is a list of items with these fields:

| field | content |
|---|---|
| id | item identifier |
| volume | CT-RATE volume file name (download the volume from CT-RATE) |
| patient | CT-RATE patient identifier (items of one patient share it) |
| task | classification, detection or correction |
| error_type | insertion, omission, direction, size, typo, unit, or none for an error-free report |
| group | nodule, effusion or control (group of the scan) |
| question | instruction given to the model |
| report | lung section of the report shown to the model |
| answer | reference answer |

The prompt used in the paper is `question + "\nHere is the medical report: '''" + report + "'''"`.

Licence: CC BY-NC-SA 4.0 (same as CT-RATE).
