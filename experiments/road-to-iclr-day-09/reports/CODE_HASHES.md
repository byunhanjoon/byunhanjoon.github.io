# Final executable hashes

The Day-09 directory is currently untracked, so the parent repository commit recorded
in run metadata does not by itself identify these files. Final SHA-256 hashes for the
shared runner and newest confirmation/audit code are:

| File | SHA-256 |
|---|---|
| `scripts/run_openml_breadth_competence.py` | `905b2b0b1aa4795a266b5af4b7767c3f8a0976eda31564ae10ec4f54c16bc7c2` |
| `scripts/analyze_classification_shrinkage_confirmation.py` | `7b2ff80712643194c58acce15977430b2b30e3f6ee071ba8de3930d689afb1f1` |
| `scripts/analyze_context_rescaled_confirmation.py` | `733c085d44d411325194099af88f049863580e4a509c9b8ebd6853887114e1bc` |
| `scripts/audit_manifest.py` | `7be9d83451b396d869f82a0f3324e926830da1eabcbf6923c5531d0e707c251d` |
| `scripts/audit_claim_state.py` | `31484b6de881aa8417f317ac92989d330f197fb0407d71dccf4caf68226b3220` |
| `configs/classification_shrinkage_confirmation.yaml` | `c41a629643fa84f2c1ce17c001db552a10b3457e8d1ee7b670fde857675c7e98` |
| `configs/context_rescaled_confirmation.yaml` | `c83c219597104e3ee599b8a42b74feea39920df43504b0c11fdb733fd708f2df` |

These hashes describe the final rerunnable state. The context-rescaling branch is inert
unless `episode_rescale: true`, so the shared runner remains behavior-compatible with
the earlier frozen configs.
