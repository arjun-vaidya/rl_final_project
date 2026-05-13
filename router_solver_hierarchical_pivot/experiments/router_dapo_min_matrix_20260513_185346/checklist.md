# Minimal Router-Solver DAPO Finalization Checklist

## Step 3. Clean router-solver DAPO-port comparison
Run:
- patched router-solver
- `execution_branch=soft`
- `G=5`
- `informative_group_sampling=on`
- `outcome_credit_mode=all`
- same 10Q train slice
- same 10Q eval slice
- corrected eval path using true `G=5`

Compare against:
- streamed baseline: majority relaxed `0.40`
- streamed `all_steps`: majority relaxed `0.50`
- promoted soft inference branch: old 10Q headline `~0.90` but not like-for-like

Success criterion:
- if informative-group + `all_steps` beats plain streamed `all_steps`, keep it as the new training baseline
- otherwise inspect and do not bless the DAPO port yet

## Step 4. Selective credit comparison on top of the stronger trainer
Compare:
- A. `informative_group_sampling + all_steps`
- B. `informative_group_sampling + answer_bearing_final`

Use:
- same checkpoint start
- same train slice
- same eval slice
- same `G=5`
- same train token cap
- same eval token cap

Success criterion:
- `answer_bearing_final` must beat `all_steps` on question-majority relaxed accuracy
- and ideally reduce `correct_number_in_trace_wrong_final`
- otherwise stop tuning credit modes

## Step 5. Minimum diagnostics for every DAPO-style router-solver run
Record:
- kept informative groups
- dropped all-correct groups
- dropped all-wrong groups
- rollout relaxed accuracy
- question-majority relaxed accuracy
- question-any relaxed accuracy
- taxonomy counts:
  - `correct_number_in_trace_wrong_final`
  - `copied_intermediate_as_final`
  - `wrong_numeric_final`

## Step 6. Freeze the steady-state soft trainer
Candidates:
- streamed + checkpointing + informative-group sampling + `all_steps`
- streamed + checkpointing + informative-group sampling + `answer_bearing_final`

Decision rule:
- choose the one with the best question-majority relaxed accuracy on the clean `G=5` slice
- if tied, choose the one with fewer `correct_number_in_trace_wrong_final` failures
- if still tied, choose the simpler one: `all_steps`

## Step 7. Branch separation only after freezing the trainer
Once the DAPO-integrated `soft` trainer is selected, rerun:
- `easy-only`
- `soft-only` with the frozen DAPO-integrated trainer
- oracle-routed `easy -> easy`, harder -> `soft`

Use one clean `G=5` eval slice.
