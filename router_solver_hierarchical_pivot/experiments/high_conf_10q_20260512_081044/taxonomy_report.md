# Failure Taxonomy Report

## Summary
- trace_path: `/home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/high_conf_10q_20260512_081044/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2000`
- relaxed_numeric_accuracy: `0.2000`
- analyzed_failures: `48`
- invalid_counts: `{}`
- final_answer_source_counts: `{"heuristic_selector_refined": 35, "synthesis_self_consistency": 25}`

## Primary Categories
- correct_number_in_trace_wrong_final: `35`
- wrong_numeric_final: `11`
- copied_intermediate_as_final: `1`
- plan_endpoint_mismatch: `1`

## Flags
- correct_number_appears_in_trace: `35`
- last_subgoal_not_answer_like: `23`
- copied_intermediate_as_final: `3`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "198", "last_subgoal": "Step 3: Add the regular and overtime pay to get the total amount earned.", "step_answers": ["198", "-3", "198"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine Tina's total earnings for working 10 hours each day over 5 days, we need to follow these steps:\n\nStep 1: Calculate the regular pay for the first 8 hours.\n\nTina's regular pay rate is $1..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "100", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To find out how many clips Natalia sold in May, we need to divide the number of clips sold in April by 2 because she sold half as many in May compared to April.\n\nCalculation for May sales:\n\\[\n\\text..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "1", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem step by step:\n\nStep 1: We know that Natalia sold 48 clips in April.\nStep 2: In May, she sold half as many clips as she did in April. Therefore, we need to find half of 48.\n\\[ \\..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "13", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["9", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n### Step-by-Step Solution:\n\n1. Determine the number of clips sold in May:\n   - Natalia sold clips to 48 friends in April.\n   - In May, she sol..."}`

### plan_endpoint_mismatch
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "-6", "last_subgoal": "Learn More", "step_answers": ["2", "12", "4", ""], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "In his weekly schedule, James sends out two letters to his friends each time, which means he writes a total of 2 letters * 2 times = 4 letters every week. There are 52 weeks in a year so over the c..."}`

### wrong_numeric_final
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "49", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["18", "28", "41", "49"], "flags": [], "last_reasoning": "To find the total number of flowers in Mark's garden, we need to follow these steps:\n\nStep 1: Calculate the number of purple flowers.\nWe know that there are 10 yellow flowers and there are 80% more..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "0", "last_subgoal": "Calculate the total number of yellow and purple flowers: Now that we know the number of purple flowers, we can simply...", "step_answers": ["", "28"], "flags": [], "last_reasoning": "### Work:\n\nGiven:\n- Number of yellow flowers = 10\n- Percentage increase in purple flowers over yellow flowers = 80%\n\nLet's first determine the number of purple flowers.\n1. Calculate 80% of the numb..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "153", "last_subgoal": "Step 2: Subtract the remaining money from her budget to find out how much she paid for the shoes.", "step_answers": ["200", "153"], "flags": [], "last_reasoning": "Remember that Alexis' budget was $200. The $16 is what she has left. To solve this problem, we need to follow these steps:\n\nStep 1: Calculate the total amount Alexis spent on the items she remember..."}`
