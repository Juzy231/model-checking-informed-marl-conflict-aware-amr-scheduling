# Manuscript-referenced model and simulator files

This small artifact set contains the source files explicitly identified in the manuscript text, plus the directly associated PRISM query and run context needed to interpret the reported formal result. It is not a copy of the full submission supplement.

## Files

- `formal/prism/amr_shared_priority_mdp.prism`: PRISM MDP source named in the model-checking description.
- `formal/prism/amr_shared_priority_mdp.props`: bounded reachability properties evaluated on that model.
- `formal/prism/shared_scenarios.csv`, `formal/run_shared_prism_scenarios.sh`: parameter cases and command wrapper for the reported PRISM scenarios.
- `formal/prism_shared_scenarios_results.csv`, `formal/logs/prism_shared_scenarios/`: summarized results and native checker outputs.
- `simulator/environment.py`: simulator environment source identified in the manuscript's implementation description.

## PRISM run

The formal analysis was run with PRISM 4.11. Install PRISM and set `PRISM_BIN` to its executable, then run from this directory:

```sh
PRISM_BIN=/path/to/prism ./formal/run_shared_prism_scenarios.sh
```

The MDP and simulator have distinct state-transition and event semantics; see the manuscript's model-scope discussion when interpreting their outputs.
