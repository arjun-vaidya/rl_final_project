# Failure Taxonomy Report

## Summary
- trace_path: `experiments/falsify_local_final_10q_20260512/rollout_traces_second_run.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.5000`
- relaxed_numeric_accuracy: `0.5000`
- analyzed_failures: `30`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 60}`

## Primary Categories
- correct_number_in_trace_wrong_final: `18`
- wrong_numeric_final: `8`
- copied_intermediate_as_final: `3`
- plan_endpoint_mismatch: `1`

## Flags
- correct_number_appears_in_trace: `18`
- last_subgoal_not_answer_like: `11`
- copied_intermediate_as_final: `4`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "198", "last_subgoal": "Step 3: Add the regular and overtime pay to get the total amount earned.", "step_answers": ["198", "1", "198"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much money Tina makes, we need to follow these steps:\n\n### Step-by-Step Solution:\n\n1. Calculate the regular pay for the first 8 hours:\n   - Tina's hourly wage = $18.00\n   - Regular..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "198", "last_subgoal": "Step 3: Add the regular and overtime pay to get the total amount of money made.", "step_answers": ["198", "252", "198"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem, we need to calculate the total earnings from both the regular hours and the overtime hours that Tina works each day. Here are the detailed steps:\n\n### Step-by-Step Solution\n\n..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "198", "last_subgoal": "Step 3: Add the regular and overtime pay to get the total amount earned.", "step_answers": ["2", "198", "198"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "In step 3:\n\nTo calculate the overtime pay, you need to multiply her hourly wage plus half of her hourly wage by the number of extra hours worked. Then add that result to the total earnings from her..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "2"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Remember to show your work for each step. Let's solve it:\n\n1. First, we know Natalia sold 48 clips in April.\n2. Then, we find out how many she sold in May by dividing that amount by two because she..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "2"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To find out how many clips Natalia sold in May, we need to calculate half of the number of clips she sold in April. \n\nWe know that she sold 48 clips in April and half as many in May.\n\nLet's calcula..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "120", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["10", "120"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The problem is solved without any additional instructions. Final answer: 120 Step 2: Multiply the number of hours by her hourly rate.\n10 * 12 = 120\nFinal answer: 120 Step 2: Multiply the number of ..."}`

### plan_endpoint_mismatch
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "0", "last_subgoal": "Read the input value", "step_answers": ["0"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To determine how much Alexis paid for the shoes, we need to follow these steps:\n\n1. Calculate the total amount spent excluding the shoes.\n2. Subtract the remaining amount she has left from her orig..."}`

### wrong_numeric_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "24", "last_subgoal": "Return the result of adding `num1` and `", "step_answers": ["8", "3", "2", "24"], "flags": [], "last_reasoning": "In my last job I made around 70 dollars an hour. I worked 9 hours on Monday night. How much money did I make?\nPlan:\n1. First find out how many hours you worked per day by dividing the total number ..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "2184", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["16", "648", "2184"], "flags": [], "last_reasoning": "To determine how many pages James writes in a year, we need to follow these steps:\n\n1. Calculate the number of letters written per week:\n   - James writes a 3-page letter to 2 different friends twi..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "52", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["248", "0.8", "52"], "flags": [], "last_reasoning": "To solve this problem systematically:\n\nStep 1: First, let's determine how many letters James writes in one day. Since he writes a letter to 2 different friends and each letter is for 2 pages (one f..."}`
