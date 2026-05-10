# Failure Taxonomy

- source checkpoint: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/slim_g6_20260508_153045/phase4_final.pt`
- target failed rollouts: `12`
- questions processed: `4`
- valid rollouts observed: `16`
- relaxed-correct rollouts observed: `8`
- failed rollouts captured: `12`

## Category Counts
- wrong_computation_or_wrong_target: `4`
- plan_parse_failed: `4`
- plan_endpoint_mismatch: `3`
- correct_number_in_reasoning_wrong_final: `1`

## Representative Failures
### wrong_computation_or_wrong_target
- question: `Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?`
- ground_truth: `72`
- last_subgoal: `Step 2: Add the number of clips sold in April and May together.`
- final_answer: `2`
- last_reasoning_excerpt: ` To solve this problem, we need to follow the given steps:

1. **Find out how many clips Natalia sold in May**:
   - According to the question, Natalia sold half as many clips in May as she did in April.
   - Number of clips sold in April = 48
   - Therefore, the number of clips sold in May is calculated as follows:
     \[
     \text{Clips sold in May} = \frac{\text{Clips sold in April}}{2}
     `

### plan_parse_failed
- question: `Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her par`
- ground_truth: `5`
- last_subgoal: `None`
- final_answer: `None`
- last_reasoning_excerpt: `None`

### plan_endpoint_mismatch
- question: `Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?`
- ground_truth: `10`
- last_subgoal: `Multiply the number of hours by the hourly rate.`
- final_answer: `9.996`
- last_reasoning_excerpt: ` To solve this step:

Step 2: Multiply the number of hours by the hourly rate.
\[ \text{Earnings} = \text{Number of hours} \times \text{Hourly rate} \]
\[ \text{Earnings} = 0.833 \text{ hours} \times \$12/\text{hour} \]

To find out how much she earned, multiply the converted time (in hours) by her hourly rate ($12 per hour).

\[ \text{Earnings} = 0.833 \times 12 \]

Let's calculate the earnings:
`

### correct_number_in_reasoning_wrong_final
- question: `Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?`
- ground_truth: `10`
- last_subgoal: `Multiply the number of hours by the hourly rate.`
- final_answer: `6`
- last_reasoning_excerpt: ` Let's continue solving it step-by-step:

To solve the problem, we need to follow these steps:

**Step 1: Convert the time from minutes to hours.**
- We know that there are 60 minutes in one hour.
- Weng worked for 50 minutes yesterday.

Now, let's convert 50 minutes into hours:

\[ \text{Number of hours} = \frac{\text{Total minutes}}{\text{Minutes per hour}} \]
\[ \text{Number of hours} = \frac{5`
