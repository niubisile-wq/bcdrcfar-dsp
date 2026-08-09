# BCDRCFAR pressure matrix

This matrix separates true pass/fail evidence from boundary-only evidence.

| axis | artifact | gate | pass_signal | evidence | boundary | next_step |
|---|---|---|---|---|---|---|
| domain_reliability | IPIX | ACCEPT | ACCEPT | coverage=0.995, auroc=0.595 | post-P4 exploratory mechanism and method development; requires fresh domain-level confirmation | freeze this gate and validate on an entirely new radar domain |
| domain_reliability | IPIX_269_high_sea | ACCEPT | ACCEPT | coverage=1.000, auroc=0.475 | post-P4 exploratory mechanism and method development; requires fresh domain-level confirmation | freeze this gate and validate on an entirely new radar domain |
| domain_reliability | IPIX_287_low_sea | ACCEPT | ACCEPT | coverage=0.994, auroc=0.797 | post-P4 exploratory mechanism and method development; requires fresh domain-level confirmation | freeze this gate and validate on an entirely new radar domain |
| domain_reliability | St_Andrews_24GHz | ABSTAIN | ABSTAIN | coverage=0.373, auroc=0.674 | post-P4 exploratory mechanism and method development; requires fresh domain-level confirmation | freeze this gate and validate on an entirely new radar domain |
| domain_reliability | St_Andrews_94GHz | ABSTAIN | ABSTAIN | coverage=0.202, auroc=0.632 | post-P4 exploratory mechanism and method development; requires fresh domain-level confirmation | freeze this gate and validate on an entirely new radar domain |
| confirmatory_real | P4_real_confirmatory | NO_GO | NO_GO | gain_over_ood=0.039, gain_ci_low=0.007, positive_acq=7, rate_ratio=1.965 | partial external transfer, not a full P4 pass | Treat as partial external transfer only; do not retune on the confirmatory set. |
| scan_domain | IPIX scan domain | NO_GO | NO_GO | observed_action=ACCEPT, expected_action=ABSTAIN, coverage=0.879 | domain-validity confirmation only; scanning land/sea scene is not a sea-clutter CFAR performance test | Develop explicit acquisition-context and observation-semantic evidence using development domains, then freeze a new rule for a fresh sea-clutter domain. |
| st_andrews_holdout | St Andrews holdout | CLOSED | CLOSED | 24GHz direct_risk_auroc=0.350, 94GHz direct_risk_auroc=0.180, ratio_ci_24=[0.5194131090305047, 0.6327942540011804], ratio_ci_94=[0.7400251792797128, 0.7861046917003922] | exploratory untouched-row holdout from already downloaded prefixes; not an independent acquisition confirmation and cannot alone open the LLM gate | Treat as exploratory untouched-prefix holdout only; it does not open the gate. |
| negative_control | NEXRAD negative control | ABSTAIN | ABSTAIN | failed_criteria=complex_iq_available,slow_time_contiguous,target_exclusion_plan,minimum_sweeps,stare_mode_supported, observed_action=True | This negative control tests whether the semantic gate rejects a real but unsupported scanning radar product. It is not sea-clutter CFAR-risk confirmation and cannot open the LLM experiment gate. | Use as contract-proof that unsupported radar products must still ABSTAIN. |

## Reading

- `ACCEPT` here means the domain-reliability gate accepted the domain, not that CFAR is solved.
- `NO_GO` and `ABSTAIN` are preserved as boundaries, not rewritten as weaker positives.
- The St Andrews and NEXRAD rows are pressure tests: useful for delimiting scope, not for opening the confirmatory gate.
