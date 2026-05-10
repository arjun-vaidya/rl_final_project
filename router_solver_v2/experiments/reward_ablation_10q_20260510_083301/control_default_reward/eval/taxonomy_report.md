# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/reward_ablation_10q_20260510_083301/control_default_reward/eval/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2667`
- relaxed_numeric_accuracy: `0.2667`
- analyzed_failures: `44`
- invalid_counts: `{}`
- final_answer_source_counts: `{"synthesis": 60}`

## Primary Categories
- wrong_numeric_final: `18`
- correct_number_in_trace_wrong_final: `14`
- copied_intermediate_as_final: `7`
- plan_endpoint_mismatch: `3`
- non_numeric_final_answer: `2`

## Flags
- correct_number_appears_in_trace: `15`
- last_subgoal_not_answer_like: `12`
- copied_intermediate_as_final: `11`
- non_numeric_final_answer: `2`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Determine the number of clips Natalia sold in May by finding half of the number of clips sold in April.\n2. Calculate the total number of cl..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "20", "last_subgoal": "Subtract the total amount from the cost of the wallet: Finally, we subtract the total amount of money Betty now has f...", "step_answers": ["2", "50", "20", "95"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Let\u2019s go through the steps:\n\nStep 1: Calculate how much money Betty currently has\nBetty needs $100 and has half of that, so she has:\n\\[ \\frac{1}{2} \\times 100 = 50 \\text{ dollars} \\]\n\nStep 2: Add t..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "15", "last_subgoal": "Finally, we subtract the total amount of money Betty now has from the cost of the wallet ($100) to find out how much ...", "step_answers": ["100", "15", "30", "95"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem, let's break it down into smaller steps:\n\n### Step-by-Step Solution:\n\nStep 1: Calculate how much money Betty currently has.\nBetty has saved half of the $100 wallet price.\n\\[ \\..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "25", "last_subgoal": "Step 6: The result will be the amount of money Betty still needs.", "step_answers": ["100", "95", "25", "30", "25", "50"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To solve this problem, we need to follow the given plan step-by-step.\n\n### Plan:\n\n1. Calculate how much money Betty already has.\n   - Betty initially had half the cost of the wallet.\n   - Cost of t..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Add the amount given by her grandparents: Her grandparents gave her twice as much as her parents, which is $30", "step_answers": ["30", "50", "50"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem step-by-step:\n\nStep 1: Calculate how much money Betty currently has.\nBetty needs $100, but she already has half of that:\n\\[\n\\frac{100}{2} = 50 \\text{ dollars}\n\\]\nSo, Betty curr..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "100", "last_subgoal": "Step 5: Determine how much more money Betty needs.", "step_answers": ["95", "125", "50", "50", "30"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we will follow these steps:\n\nStep 1: Calculate how much money Betty currently has.\n- Betty has only half of the money she needs for t..."}`

### non_numeric_final_answer
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "Note: The above steps ensure that all necessary calculations have been performed correctly according to the original ...", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "120"], "flags": ["non_numeric_final_answer", "correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we will follow the plan provided:\n\n1. Find out how many clips Natalia sold in May:\n   - According to the previous step, Natalia sold 72 clips in May.\n\n2. Add the number of cl..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "Please provide additional information about the shoes Alexis bought so that I can complete the solution. Thank you! \ufffd", "last_subgoal": "Step 2: Subtract the remaining money from the budget to find out how much she paid for the shoes.", "step_answers": ["143", "46"], "flags": ["non_numeric_final_answer"], "last_reasoning": "The following must be present:\n\nPlan:\n1. Step 1: ...\n2. Step 2: ...\n\n[Insert calculations]\n\nFinal answer: [Number] Let's solve this step-by-step according to the plan provided.\n\nPlan:\n1. Step 1: Ca..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "1", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["12", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To determine how much Weng earned yesterday, we need to follow these steps:\n\nStep 1: Convert the time from minutes to hours.\nWeng babysat for 50 minutes. Since there are 60 minutes in an hour, we c..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["5", "50"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, let's follow the plan:\n\n1. Convert the time from minutes to hours:\n   - We know that there are 60 minutes in one hour.\n   - Therefore, we need to convert 50 minutes into hour..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "12", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["-4", "48", "60", "24"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To find out how many pages Julie needs to read tomorrow, let's follow these steps:\n\n1. Calculate the number of pages Julie read today:  \n   Yesterday, Julie read 12 pages. Today, she read twice as ..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Step 5: The result will be the additional money needed.", "step_answers": ["65", "10", "50", "55", "30"], "flags": [], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, let's go through each step methodically:\n\n### Step-by-Step Solution\n\nStep 1: Calculate how much money Betty currently has.\nBetty need..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "704", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["52", "12", "6"], "flags": [], "last_reasoning": "You must use the variable $n$ for any unknown quantity (like the number of times James writes) and follow along as I walk through the solution:\n\nJames writes a 3-page letter to 2 different friends ..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "1092", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["12", "52", "52"], "flags": [], "last_reasoning": "To solve the problem:\n\nGiven:\n- James writes a 3-page letter to 2 different friends twice a week.\n- He does this twice a week.\n\nSteps to solve:\n1. Calculate the number of letters written per week: ..."}`
