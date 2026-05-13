# Failure Taxonomy Report

## Summary
- trace_path: `experiments/falsify_local_final_10q_20260512/rollout_traces_reconstructed_second.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.4000`
- relaxed_numeric_accuracy: `0.4000`
- analyzed_failures: `36`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 60}`

## Primary Categories
- correct_number_in_trace_wrong_final: `24`
- wrong_numeric_final: `7`
- copied_intermediate_as_final: `3`
- plan_endpoint_mismatch: `2`

## Flags
- correct_number_appears_in_trace: `24`
- last_subgoal_not_answer_like: `15`
- copied_intermediate_as_final: `4`

## Example Failures
### copied_intermediate_as_final
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "1248", "last_subgoal": "width=device-width, initial-scale=1.0", "step_answers": ["156", "208", "1248", "1", "1248"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, let's break it down into smaller steps and perform calculations accordingly:\n\n1. Calculate the number of letters written per week:\n   - James writes a 3-page letter to 2 diff..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "28", "last_subgoal": "Calculate the total number of yellow and purple flowers: Now that we know the number of yellow and purple flowers, we...", "step_answers": ["28", "28"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Let's solve this problem step by step.\n\nStep 1: Calculate the number of purple flowers\n\nGiven:\n- Number of yellow flowers = 10\n- Purple flowers are 80% more than yellow flowers\n\nFirst, calculate 80..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "198", "last_subgoal": "Step 3: Add the regular and overtime pay to get the total amount earned.", "step_answers": ["198", "198", "198"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much money Tina makes, we need to follow these steps:\n\nStep 1: Calculate the regular pay for the first 8 hours.\n\n\\[ \\text{Regular Pay} = \\text{Hourly Wage} \\times \\text{Number of R..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "2"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Remember to show your work for each step. Let's solve it:\n\n1. First, we know Natalia sold 48 clips in April.\n2. Then, we find out how many she sold in May by dividing that amount by two because she..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "2"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To find out how many clips Natalia sold in May, we need to calculate half of the number of clips she sold in April. \n\nWe know that she sold 48 clips in April and half as many in May.\n\nLet's calcula..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "1", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["9.96", "1"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this step, follow these detailed instructions:\n\nGiven that Weng earned $12 per hour and she spent 50 minutes on babysitting yesterday, we need to determine how much she earned.\n\n### Step-b..."}`

### plan_endpoint_mismatch
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "31", "last_subgoal": "Commission: This is earned based on the number of hours worked beyond the first 40 hours in a week.", "step_answers": ["63", "31"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "In previous versions, I have seen students write an explanation or sentence instead of the final answer line.\nTo solve this problem, we need to determine how much Alexis paid for the shoes given he..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "67", "last_subgoal": "An additional payment of $2 for each hour worked beyond the first 40 hours.", "step_answers": ["127", "67"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "It may be helpful to draw diagrams or make tables as needed to illustrate your solution strategy. To solve the problem, let's break down Alexis' expenses and income step by step:\n\n1. Fixed Salary: ..."}`

### wrong_numeric_final
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "208", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["2", "4", "208"], "flags": [], "last_reasoning": "To solve the problem, we need to break it down into smaller steps as described:\n\n1. Calculate the number of letters written per week:\n   - James writes 2 letters (to 2 different friends) twice a we..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "4", "last_subgoal": "Step 3: Add the remaining money ($16) to the cost of the shoes to determine their price.", "step_answers": ["200", "31", "4"], "flags": [], "last_reasoning": "Let's solve it step by step as requested:\n\n1. Step 1: Calculate the total amount Alexis spent on the items she remembers.\n   - Button-up shirt: $30\n   - Suit pants: $46\n   - Suit coat: $38\n   - Soc..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "61", "last_subgoal": "Step 2: Subtract the remaining money from the budget to find out how much she paid for the shoes.", "step_answers": ["59", "61"], "flags": [], "last_reasoning": "The shoes cost Alexis $\\boxed{61}$ dollars. \n\nLet's solve this step-by-step:\n\nStep 1: We already calculated the total amount Alexis spent on all items except the shoes, which was $59 (shirt + pants..."}`
