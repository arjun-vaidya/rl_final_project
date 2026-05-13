# Failure Taxonomy Report

## Summary
- trace_path: `experiments/falsify_local_final_10q_20260512/rollout_traces.jsonl`
- records: `20`
- total_rollouts: `120`
- valid_rollouts: `120`
- exact_accuracy: `0.4333`
- relaxed_numeric_accuracy: `0.4333`
- analyzed_failures: `68`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 120}`

## Primary Categories
- correct_number_in_trace_wrong_final: `44`
- wrong_numeric_final: `15`
- copied_intermediate_as_final: `6`
- plan_endpoint_mismatch: `3`

## Flags
- correct_number_appears_in_trace: `44`
- last_subgoal_not_answer_like: `28`
- copied_intermediate_as_final: `8`

## Example Failures
### copied_intermediate_as_final
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "1248", "last_subgoal": "width=device-width, initial-scale=1.0", "step_answers": ["156", "208", "1248", "1", "1248"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, let's break it down into smaller steps and perform calculations accordingly:\n\n1. Calculate the number of letters written per week:\n   - James writes a 3-page letter to 2 diff..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "28", "last_subgoal": "Calculate the total number of yellow and purple flowers: Now that we know the number of yellow and purple flowers, we...", "step_answers": ["28", "28"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Let's solve this problem step by step.\n\nStep 1: Calculate the number of purple flowers\n\nGiven:\n- Number of yellow flowers = 10\n- Purple flowers are 80% more than yellow flowers\n\nFirst, calculate 80..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "198", "last_subgoal": "Step 3: Add the regular and overtime pay to get the total amount earned.", "step_answers": ["198", "198", "198"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much money Tina makes, we need to follow these steps:\n\nStep 1: Calculate the regular pay for the first 8 hours.\n\n\\[ \\text{Regular Pay} = \\text{Hourly Wage} \\times \\text{Number of R..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "56", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "56"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The problem has been solved by showing all the necessary steps.\nTo find out how many clips Natalia sold in May, we need to calculate half of the number of clips she sold in April. Since she sold 48..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "18", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["4", "18"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Let's solve it:\n\nTo find out how many clips Natalia sold in May, we need to calculate half of the amount she sold in April. \n\nApril sales = 48 clips\nMay sales = 1/2 * April sales\n\nNow let's perform..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "0", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "0"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To find out how many clips Natalia sold in May, we need to divide the number of clips sold in April by two because she sold half as many clips in May as she did in April. \n\nSo, if Natalia sold 48 c..."}`

### plan_endpoint_mismatch
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "31", "last_subgoal": "Commission: This is earned based on the number of hours worked beyond the first 40 hours in a week.", "step_answers": ["63", "31"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "In previous versions, I have seen students write an explanation or sentence instead of the final answer line.\nTo solve this problem, we need to determine how much Alexis paid for the shoes given he..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "67", "last_subgoal": "An additional payment of $2 for each hour worked beyond the first 40 hours.", "step_answers": ["127", "67"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "It may be helpful to draw diagrams or make tables as needed to illustrate your solution strategy. To solve the problem, let's break down Alexis' expenses and income step by step:\n\n1. Fixed Salary: ..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "0", "last_subgoal": "Read the input value", "step_answers": ["0"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To determine how much Alexis paid for the shoes, we need to follow these steps:\n\n1. Calculate the total amount spent excluding the shoes.\n2. Subtract the remaining amount she has left from her orig..."}`

### wrong_numeric_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "24", "last_subgoal": "Return the result of adding `num1` and `", "step_answers": ["8", "3", "2", "24"], "flags": [], "last_reasoning": "In my last job I made around 70 dollars an hour. I worked 9 hours on Monday night. How much money did I make?\nPlan:\n1. First find out how many hours you worked per day by dividing the total number ..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "208", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["2", "4", "208"], "flags": [], "last_reasoning": "To solve the problem, we need to break it down into smaller steps as described:\n\n1. Calculate the number of letters written per week:\n   - James writes 2 letters (to 2 different friends) twice a we..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "2184", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["16", "648", "2184"], "flags": [], "last_reasoning": "To determine how many pages James writes in a year, we need to follow these steps:\n\n1. Calculate the number of letters written per week:\n   - James writes a 3-page letter to 2 different friends twi..."}`
