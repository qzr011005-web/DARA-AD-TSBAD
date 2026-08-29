# DARA-AD: A Decomposition-Aware Reconstruction Architecture for Multivariate Time-Series Anomaly Detection

This repository provides the code for the final DARA-AD configuration reported in the paper:

**Decomposition-Aware Residual-Gated Attention for Multivariate Time-Series Anomaly Detection**

## Final Configuration

The released version corresponds to the final single-stage DARA-AD model:

- single-stage Robust Adaptive Time-Frequency Decomposition (RATFD)
- Role-Specialized Dual-Attention Reconstruction (RSDA-R)
- Whitened Residual-Gated Adapter (WRGA)
- Clean-Residual Joint Anomaly Scoring (CRJAS)

The final released implementation forces the RATFD module to use single-stage seasonal extraction. The second-stage seasonal extraction is disabled in the final configuration.

## Dataset

The experiments are conducted on the public TSB-AD multivariate evaluation subset, TSB-AD-M-Eva. The raw dataset is not redistributed in this repository.

## Run

```bash
bash scripts/run_final_dara_ad.sh
```

## Final Reported Result

| Metric | Result |
|---|---:|
| AUC-PR | 0.3895 |
| AUC-ROC | 0.7428 |
| VUS-PR | 0.3720 |
| VUS-ROC | 0.7631 |
| Standard-F1 | 0.4254 |
| PA-F1 | 0.7908 |
| Event-F1 | 0.6160 |
| R-F1 | 0.4125 |
| Affiliation-F | 0.8579 |

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Benchmark Notice

The dataset and benchmark protocol are from the public TSB-AD project. Users should follow the license and usage requirements of the original TSB-AD benchmark.
