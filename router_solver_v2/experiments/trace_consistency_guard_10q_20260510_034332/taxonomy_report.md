# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/trace_consistency_guard_10q_20260510_034332/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2000`
- relaxed_numeric_accuracy: `0.2000`
- analyzed_failures: `48`
- invalid_counts: `{}`
- final_answer_source_counts: `{"consistency_fallback": 3, "synthesis": 57}`

## Primary Categories
- wrong_numeric_final: `17`
- copied_intermediate_as_final: `16`
- correct_number_in_trace_wrong_final: `12`
- plan_endpoint_mismatch: `3`

## Flags
- copied_intermediate_as_final: `22`
- correct_number_appears_in_trace: `12`
- last_subgoal_not_answer_like: `12`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow the steps outlined:\n\nStep 1: Find out how many clips Natalia sold in May.\n\nWe are given that Natalia sold half as many clips in May as she did in April. Sin..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "55", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["66.67", "30", "55", "65"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve the problem, let's follow the steps:\n\nStep 1: Calculate how much money Betty already has.\n\nBetty needs $100 and she currently has half of it. So,\n\\[\n\\text{Money Betty has} = \\frac{100}{2} ..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "84", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["2", "84", "36", "84"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To determine how many pages Julie should read tomorrow, we need to follow these steps:\n\n1. Calculate the number of pages Julie read today:  \n   Since she read twice as many pages today as she did y..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "800", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["2", "10"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To convert 50 minutes to hours, we use the conversion factor that there are 60 minutes in one hour. Therefore, we divide the number of minutes by 60:\n\n\\[ \\text{Hours} = \\frac{\\text{Minutes}}{60} = ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "95", "last_subgoal": "Add up all the money Betty", "step_answers": ["30", "30", "95", "30"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "Do not write in the final step.\n\nFinal answer: 95 Let's solve it step by step:\n\nStep 1: Calculate how much money Betty already has.\nSince Betty has half of the money she needs ($100) and she has di..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["5", "95", "50", "65"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "The problem cannot be solved without knowing what number appears on each line of the first set of steps (which do not appear here). To solve the problem, we need to follow these steps:\n\n### Step-by..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "9.96", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["2", "9"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To convert the time from minutes to hours, we use the conversion factor that 60 minutes is equal to 1 hour. So, we divide the number of minutes by 60:\n\n$$ \\text{Time in hours} = \\frac{\\text{Time in..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "5", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["60", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "If necessary, use scientific notation or decimal points as appropriate. To solve the problem of how much Weng earned for 50 minutes of babysitting at a rate of $12 per hour:\n\n1. Convert the time fr..."}`
- `{"question": "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into...", "ground_truth": "16", "final_answer": "1", "last_subgoal": "Step 5: Double the weight by adding more gummy worms.", "step_answers": ["3", "8", "8", "54", "8", "2"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "Let's go through each step carefully:\n\n1. Initial Weight: The initial weight is the empty box (step 1). We need to determine this based on the information provided that Ken filled the box with enou..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "90", "last_subgoal": "Add up all the money Betty now has: We'll sum up the amounts from each step above to get", "step_answers": ["50", "65", "30", "3"], "flags": [], "last_reasoning": "To find out how much more money Betty needs to buy the wallet, let's follow these steps:\n\n1. Calculate how much money Betty currently has: \n   - Betty has $50 because she has half of the cost of th..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "432", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["60", "6", "7"], "flags": [], "last_reasoning": "To determine how many pages James writes in a year, we need to follow these steps:\n\n1. Calculate the number of letters written per week:\n   - James writes a 3-page letter to 2 different friends twi..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "3", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["24", "208", "52"], "flags": [], "last_reasoning": "To solve the problem:\n\n1. Calculate the number of letters written per week:  \n   James writes a 3-page letter to 2 different friends twice a week. Therefore, the number of letters written per week ..."}`
