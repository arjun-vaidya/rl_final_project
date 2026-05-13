# Failure Taxonomy Report

## Summary
- trace_path: `/home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/review_fix_next5q_20260512_115617/rollout_traces.jsonl`
- records: `5`
- total_rollouts: `30`
- valid_rollouts: `30`
- exact_accuracy: `0.3000`
- relaxed_numeric_accuracy: `0.3000`
- question_majority_exact_accuracy: `0.6000`
- question_majority_relaxed_accuracy: `0.6000`
- question_any_relaxed_accuracy: `0.8000`
- analyzed_failures: `21`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 30}`

## Primary Categories
- correct_number_in_trace_wrong_final: `15`
- copied_intermediate_as_final: `3`
- wrong_numeric_final: `3`

## Flags
- correct_number_appears_in_trace: `15`
- copied_intermediate_as_final: `4`
- last_subgoal_not_answer_like: `3`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "184", "last_subgoal": "Calculate the total amount Alexis spent on all items except the shoes.", "step_answers": ["184", "1", "321", "168", "32", "168", "12", "184"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem, we first need to calculate the total amount Alexis spent on all items except the shoes. The costs of these items are as follows:\n\n- Button-up shirt: $30\n- Suit pants: $46\n- S..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "900", "last_subgoal": "Step 2: Calculate Tina's regular pay. Then answer the original question directly using the previous results.", "step_answers": ["900", "8", "720", "10", "2003", "54", "9", "900"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Plan:\n\nTo solve this problem, we need to follow these steps:\n\n1. Calculate Regular Pay: First, determine the amount Tina earns during her regular working hours.\n2. Calculate Overtime Pay: Next, cal..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "1134", "last_subgoal": "Step 3: Add both amounts to get total earnings.", "step_answers": ["144", "1134", "1134"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Plan step:\n\nTo find out how much money Tina makes, we need to calculate her regular pay and overtime pay separately and then add them together. Let's start with calculating her regular pay first.\n\n..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "289", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["35", "28", "21", "49", "28", "1456", "35", "289"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "Do not include any other extraneous material in your answer.\n\nFinal answer: 289 Final answer: 289\n\nExplanation:\n\nMark initially planted 10 yellow flowers. Since there were 80% more purple flowers t..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "90", "last_subgoal": "Step 4: Sum up all the flowers to get the total count.", "step_answers": ["49", "35", "8", "35", "35", "35", "90", "90"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "In your show-work line, calculate the number of green flowers. Then sum that to the totals from previous steps. \n\nShow Work:\n\nAnswer: 35\n\nFinal answer: 90 Final answer: 90def solve_question(step):\n..."}`
- `{"question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a sm...", "ground_truth": "48", "final_answer": "80", "last_subgoal": "Add the two totals together to get the overall number of slices:", "step_answers": ["32", "48", "80"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The purpose of this prompt is for you to show your working out, so we understand what method you're using to solve these kinds of problems. Please provide your response accordingly. \n\nLet's calcula..."}`

### wrong_numeric_final
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "199", "last_subgoal": "Step 4: Sum up all the flowers to get the total count.", "step_answers": ["22", "7", "8", "7", "47", "6", "71", "199"], "flags": [], "last_reasoning": "Do not include any other extraneous text. Plan steps must be followed in order to receive full credit for your solution.\n\nSolve this step:\n\nFinal answer: 199\n\nExplanation: First, we find that the n..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "1", "last_subgoal": "Step 3: Add both amounts to get total earnings.", "step_answers": ["2", "198", "1"], "flags": [], "last_reasoning": "The above question was created using real-world data. In other words, all answers should be rounded to two decimal places and dollars can be represented without currency symbols (i.e., no \"$\" or \",..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "2250", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Tina makes $...", "step_answers": ["216", "738", "198", "2250"], "flags": [], "last_reasoning": "To solve this problem, let's follow the given steps:\n\n### Step 1: Calculate Tina's regular pay.\nTina works 10 hours each day for 5 days, so her total working hours are \\( 10 \\text{ hours/day} \\time..."}`
