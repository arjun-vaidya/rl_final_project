# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/reward_ablation_10q_20260510_083301/outcome_heavy/eval/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2500`
- relaxed_numeric_accuracy: `0.2500`
- analyzed_failures: `45`
- invalid_counts: `{}`
- final_answer_source_counts: `{"synthesis": 60}`

## Primary Categories
- wrong_numeric_final: `21`
- correct_number_in_trace_wrong_final: `11`
- copied_intermediate_as_final: `9`
- non_numeric_final_answer: `3`
- plan_endpoint_mismatch: `1`

## Flags
- last_subgoal_not_answer_like: `12`
- correct_number_appears_in_trace: `11`
- copied_intermediate_as_final: `9`
- non_numeric_final_answer: `3`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["30", "50", "30", "50"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Let's solve the problem step by step:\n\n1. Calculate how much money Betty currently has:\n   - The wallet costs $100.\n   - Betty has only half of the money needed.\n   - Therefore, the amount Betty ha..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "36", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["1", "36", "84", "36"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "The question will be marked incorrect if it uses words such as \"Conclusion,\" \"In summary,\" or similar phrases.\nTo find out how many pages Julie needs to read tomorrow, we'll follow these steps:\n\n1...."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "12", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["12", "47040", "3"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Show all your work and calculations clearly so I can understand what you've done. To solve this problem, we need to follow the given plan step-by-step:\n\n### Step 1: Calculate the number of letters ..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "1", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The given information is insufficient for solving the question posed. Please check the details provided or ask for additional information before proceeding. To solve the problem correctly, we need ..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "48", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Natalia sold 72 clips altogether in April and May. Final answer: 72."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "24"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The final answer is: 120\n\nLet's break down the problem into clear steps:\n\nStep 1: First, we need to determine how many clips Natalia sold in May.\n\n- We know that Natalia sold 48 clips in April.\n- I..."}`

### non_numeric_final_answer
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "If there are multiple possible answers, return \"Multiple answers\". Final answer: Multiple answers.", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["50", "30", "55", "65"], "flags": ["non_numeric_final_answer"], "last_reasoning": "To solve this problem, let's go through each step carefully:\n\nStep 1: Calculate how much money Betty currently has.\n- Betty initially had half of the money needed for the wallet.\n- The cost of the ..."}`
- `{"question": "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into...", "ground_truth": "16", "final_answer": "If there are multiple possible answers, return the one that makes sense in context. In this case, the correct final a...", "last_subgoal": "Step 5: Double the weight by adding enough gummy worms.", "step_answers": ["8", "6", "20", "30", "0"], "flags": ["non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem, we need to follow each step carefully:\n\n### Step 1: Initial Setup\n- The initial weight of the box is \\( W \\) (in pounds).\n- Ken starts with an empty box weighing \\( W \\).\n\n###..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "Note: This problem does not require any further computation as the solution is already provided in the plan. However,...", "last_subgoal": "Step 2: Subtract the remaining money from the budget to find out how much she paid for the shoes.", "step_answers": ["125", "184"], "flags": ["non_numeric_final_answer"], "last_reasoning": "To determine how much Alexis paid for the shoes, we need to follow these steps:\n\n1. Calculate the total amount Alexis spent on all items except the shoes:\n   - Button-up shirt: $30\n   - Suit pants:..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "7.2", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["0.83", "7.2"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The correct calculation is:\n\n$$\n\\text{Hourly rate} \\times \\frac{\\text{Number of minutes}}{60} = \\$7.20\n$$\n\nWhere does Step 1 go? In the Plan section, under \"Convert the time from minutes to hours.\"..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "55", "last_subgoal": "Subtract the total amount from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["100", "45", "55"], "flags": [], "last_reasoning": "Step 3: \nTotal amount Betty receives from her parents = 15\n\nAmount received from grandparents = 2 * 15 = 30\n\nTotal additional money Betty receives = 15 + 30 = 45\n\nMoney required to purchase the wal..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["65", "35", "95", "30"], "flags": [], "last_reasoning": "To solve this problem, we need to follow the plan provided:\n\n1. Calculate how much money Betty currently has.\n   - The cost of the wallet is $100.\n   - Betty currently has half of what she needs, s..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "65", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["75", "95", "100", "65"], "flags": [], "last_reasoning": "Let's solve the problem step-by-step:\n\nGiven information:\n- The cost of the wallet = $100\n- Betty currently has half of what she needs.\n- Her parents gave her $15.\n- Her grandparents gave her twice..."}`
