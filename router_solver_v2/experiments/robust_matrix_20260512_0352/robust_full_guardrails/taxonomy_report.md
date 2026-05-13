# Failure Taxonomy Report

## Summary
- trace_path: `/home/machina/pvd2112/rl_final_project/router_solver_v2/experiments/robust_matrix_20260512_0352/robust_full_guardrails/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `40`
- valid_rollouts: `40`
- exact_accuracy: `0.1000`
- relaxed_numeric_accuracy: `0.1000`
- analyzed_failures: `36`
- invalid_counts: `{}`
- final_answer_source_counts: `{"candidate_rerank": 40}`

## Primary Categories
- correct_number_in_trace_wrong_final: `30`
- copied_intermediate_as_final: `3`
- wrong_numeric_final: `3`

## Flags
- correct_number_appears_in_trace: `30`
- last_subgoal_not_answer_like: `18`
- copied_intermediate_as_final: `9`

## Example Failures
### copied_intermediate_as_final
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "52", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["52", "936", "3120"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Final answer: 87480 To solve the problem:\n\n1. Calculate the number of letters written per week: \n   - James writes a 3-page letter to 2 different friends twice a week.\n   - Number of pages per lett..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "312", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["2340", "312", "31680"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Final answer: 71928 Final answer: 71928\n\nExplanation:\n\nWe have already calculated that James writes a total of 2340 pages per week (from Step 1). There are approximately 52 weeks in a year, so we m..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "756", "last_subgoal": "Step 3: Add the regular and overtime pay to find the total amount of money made.", "step_answers": ["756", "198", "198"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Let's solve it:\n\nTo calculate Tina's total earnings, we need to follow these steps:\n\n1. Calculate the regular pay for the first 8 hours:\n   - Tina earns $18.00 per hour.\n   - She works 8 hours at h..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "48", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem systematically:\n\nGiven information:\n- In April, Natalia sold 48 clips.\n- In May, she sold half as many clips as in April.\n\nTo find the total number of clips sold in April and ..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "5", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\nStep 1: Determine how many clips Natalia sold in May. According to the previous information provided, Natalia sold half as many clips in May a..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "150", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "150"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Let's solve the problem step by step:\n\n### Step-by-Step Solution\n\nStep 1: Find out how many clips Natalia sold in May.\nNatalia sold 48 clips in April. According to the plan, she sold half as many c..."}`

### wrong_numeric_final
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "10", "last_subgoal": "Step 3: The result will be the cost of the shoes.", "step_answers": ["57", "184", "184", "184"], "flags": [], "last_reasoning": "To solve the problem, let's follow the plan step by step:\n\n1. Solution: We need to determine the total amount spent excluding the shoes and then subtract this from the initial budget to find the co..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "252", "last_subgoal": "Step 3: Add the regular and overtime pay to get the total amount earned.", "step_answers": ["198", "8", "198"], "flags": [], "last_reasoning": "To solve this problem, let's break it down into the three steps mentioned:\n\n### Step 1: Calculate the regular pay for the first 8 hours.\nTina earns $18.00 per hour. She works up to 8 hours without ..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "27", "last_subgoal": "Step 3: Add the regular and overtime pay to find the total amount earned.", "step_answers": ["198", "198", "1"], "flags": [], "last_reasoning": "The answer should be unambiguous and easily understood from just that line alone. Do not include units of measure or dollar signs in your answer unless explicitly requested. To solve the problem st..."}`
