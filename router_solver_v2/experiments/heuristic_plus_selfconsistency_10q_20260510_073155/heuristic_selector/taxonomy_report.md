# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/heuristic_plus_selfconsistency_10q_20260510_073155/heuristic_selector/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2500`
- relaxed_numeric_accuracy: `0.2500`
- analyzed_failures: `45`
- invalid_counts: `{}`
- final_answer_source_counts: `{"heuristic_selector": 60}`

## Primary Categories
- wrong_numeric_final: `21`
- correct_number_in_trace_wrong_final: `12`
- plan_endpoint_mismatch: `7`
- copied_intermediate_as_final: `5`

## Flags
- correct_number_appears_in_trace: `12`
- last_subgoal_not_answer_like: `12`
- copied_intermediate_as_final: `8`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "36", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["2", "14", "36", "36"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "Remember to show your work by writing legibly wherever you show computation. To determine how many pages Julie needs to read tomorrow, we can follow these steps:\n\n1. Calculate the number of pages J..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "18", "last_subgoal": "Step 4: Sum up all the flowers to get the total count.", "step_answers": ["80", "80", "18", "18"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n### Step 1: Calculate the number of purple flowers.\nMark has ten yellow flowers. According to the question, there are 80% more purple flowers ..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "18", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["18", "18", "7", "18"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Remember to show your work!\n\nLet's solve this problem:\n\n1. Step 1: Calculate the number of purple flowers.\n   - Given that there are 80% more purple flowers than yellow ones.\n   - Number of yellow ..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "24"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we will follow these steps:\n\nStep 1: Find out how many clips Natalia sold in May.\nNatalia sold half as many clips in May as she did in April.\n\nGiven that she sold 48 clips in..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "120", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "120"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\nStep 1: Find out how many clips Natalia sold in May.\n\nWe know from the previous part that Natalia sold 72 clips in May.\n\nSo, we have:\n- Clips ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "20", "last_subgoal": "Add up all the money Betty now", "step_answers": ["75", "9", "20", "100"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "The total amount of money Betty has now:\n\nStep 1: Calculate how much money Betty already has:  \nBetty initially has half of what she needs, which is $100 / 2 = $50.\n\nStep 2: Determine how much mone..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["49.999", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "Let's solve the problem step by step:\n\nGiven:\n- Hourly wage = $12\n- Time worked = 50 minutes\n\nFirst, we need to convert the time from minutes to hours. \n\nStep 1: Convert the time from minutes to ho..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "9.33", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["47.67", "12"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To determine how much Weng earned for 50 minutes of babysitting at a rate of $12 per hour, follow these steps:\n\nStep 1: Convert the time from minutes to hours.\n\nSince there are 60 minutes in an hou..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "50", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["12", "3"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The answer is 3."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "3", "last_subgoal": "Step 5: The result will be the additional amount needed.", "step_answers": ["30", "50", "25", "75", "3"], "flags": [], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we will follow the plan outlined:\n\n1. Calculate how much money Betty currently has:  \n   Since Betty initially had only half of the m..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "10", "last_subgoal": "Step 5: Determine how much more money Betty needs.", "step_answers": ["30", "15", "30", "95", "100"], "flags": [], "last_reasoning": "To solve the problem, we will follow each step as described:\n\n### Step 1: Calculate how much money Betty currently has.\nBetty initially saves only half of the $100 wallet.\n\\[ \\text{Money saved} = \\..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "3", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["95", "65", "30", "3"], "flags": [], "last_reasoning": "To solve the problem, we'll follow the plan provided:\n\n### Plan:\n\n1. Calculate how much money Betty currently has.\n2. Add the amount given by her parents.\n3. Add the amount given by her grandparent..."}`
