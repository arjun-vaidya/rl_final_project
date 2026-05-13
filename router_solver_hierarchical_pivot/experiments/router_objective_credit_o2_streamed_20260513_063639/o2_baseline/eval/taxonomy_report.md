# Failure Taxonomy Report

## Summary
- trace_path: `experiments/router_objective_credit_o2_streamed_20260513_063639/o2_baseline/eval/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `50`
- valid_rollouts: `50`
- exact_accuracy: `0.3400`
- relaxed_numeric_accuracy: `0.3400`
- question_majority_exact_accuracy: `0.4000`
- question_majority_relaxed_accuracy: `0.4000`
- question_any_relaxed_accuracy: `0.7000`
- analyzed_failures: `33`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 50}`

## Primary Categories
- correct_number_in_trace_wrong_final: `25`
- copied_intermediate_as_final: `6`
- wrong_numeric_final: `2`

## Flags
- correct_number_appears_in_trace: `25`
- copied_intermediate_as_final: `15`
- last_subgoal_not_answer_like: `1`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "18", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Mark has a g...", "step_answers": ["18", "18"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To calculate the number of purple flowers:\n\n1. We know there are 10 yellow flowers.\n2. Purple flowers are 80% more than yellow ones, so we need to find 80% of 10 and add that to 10.\n3. Calculate 80..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "7", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["7", "28", "25", "25", "53", "0.25", "28", "7"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Do not write \"The answer is\" or any similar phrase at the beginning of your response. To solve this problem, we will follow the given steps:\n\n1. Calculate the number of purple flowers:\n   - We know..."}`
- `{"question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a sm...", "ground_truth": "48", "final_answer": "16", "last_subgoal": "Add the two totals to get the overall number of slices:", "step_answers": ["16", "32", "16"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To calculate the total number of slices from the large pizzas:\n\\[ \\text{Total slices from large pizzas} = \\text{Number of large pizzas} \\times \\text{Slices per large pizza} \\]\n\\[ \\text{Total slices..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "1", "last_subgoal": "Add the number of clips sold in April and May together to get the total number of clips sold.", "step_answers": ["72", "99", "72", "72", "72", "99", "72", "1"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "To calculate the total number of clips Natalia sold in April and May:\n\n1. From Step 1, we know that Natalia sold 48 clips in April.\n2. From Step 3, we learned that in May, she sold half as many cli..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "60", "last_subgoal": "divide Then answer the original question directly using the previous results.", "step_answers": ["60", "10", "12", "10", "60", "50", "10", "60"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "Plan: 1. Convert minutes to hours. \nAction: divide\nAnswer: 5/6\n\nFinal answer: 5/6 Step 1: Convert minutes to hours.\n50 minutes \u00f7 60 = 5/6 hours\n\nStep 2: Multiply by hourly rate.\n(5/6) \u00d7 $12 = $10\n\n..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "50", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Weng earns $...", "step_answers": ["10", "7", "50"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "Plan: \n1. Convert minutes to hours\n\nAnswer: 10/60 = 1/6\n\nMultiply the number of hours by the hourly rate:\n\n1/6 * $12/hour = $2\n\nFinal answer: 2\n\nShow work:  \n1. Convert minutes to hours (50 minutes..."}`

### wrong_numeric_final
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "2", "last_subgoal": "Step 3: Add both amounts to find total earnings.", "step_answers": ["144", "0", "2"], "flags": [], "last_reasoning": "We will use the following information: Tina worked 10 hours each day and her hourly rate was $18.00.\n\nLet's solve it:\n\nRegular Pay Calculation:\nTina works 10 hours every day at $18/hour, so we calc..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "171", "last_subgoal": "Step 2: Calculate Tina's overtime pay. Then answer the original question directly using the previous results.", "step_answers": ["900", "10", "720", "2", "0", "3", "54", "171"], "flags": [], "last_reasoning": "Plan step: 1: Calculate Tina's regular pay.\nAnswer: 900\n\nFinal answer: 900\n\nExplanation: Tina earns $18.00 per hour and works 8 hours each day. So her regular pay for one day is 8 * $18 = $144. Sin..."}`
