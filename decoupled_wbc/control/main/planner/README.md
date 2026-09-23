In three different terminals, run:

``` bash
source .venv_sim/bin/activate
python decoupled_wbc/control/main/planner/run_g1_control_loop.py
python decoupled_wbc/control/main/planner/run_planner_server.py
python decoupled_wbc/control/main/planner/example_planner_request.py
```

For reference-guided (demonstration-following) planning, see
[SIMILARITY_PLANNER.md](SIMILARITY_PLANNER.md).
