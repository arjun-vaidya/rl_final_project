# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/parser_split_10q/baseline/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `51`
- exact_accuracy: `0.2000`
- relaxed_numeric_accuracy: `0.2000`
- analyzed_failures: `48`
- invalid_counts: `{"plan_parse_failed": 9}`
- final_answer_source_counts: `{"last_step": 51, "none": 9}`

## Primary Categories
- wrong_numeric_final: `15`
- copied_intermediate_as_final: `13`
- correct_number_in_trace_wrong_final: `10`
- plan_parse_failed: `9`
- plan_endpoint_mismatch: `1`

## Flags
- last_subgoal_not_answer_like: `18`
- copied_intermediate_as_final: `15`
- correct_number_appears_in_trace: `10`
- invalid_rollout: `9`
- non_numeric_final_answer: `9`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\nStep 1: Determine how many clips Natalia sold in May.\n- In April, Natalia sold 48 clips.\n- She sold half as many clips in May compared to Apri..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "95", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["15", "95", "30", "95"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we will follow these steps:\n\nStep 1: Calculate how much money Betty currently has.\n\nBetty has saved half of the $100 wallet, so:\n\\[ \\..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["180", "30", "125", "30"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we can follow these steps:\n\n1. Calculate how much money Betty currently has: \n   - Since Betty has only half of the money she needs f..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "2"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, let's follow the given plan:\n\nStep 1: Find out how many clips Natalia sold in May.\n- According to the question, Natalia sold half as many clips in May as she did in April.\n- ..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["60", "6"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve the second step, follow these detailed instructions:\n\nTo find out how much Weng earned from her 50-minute babysitting session, you need to convert the time into hours first and then calcul..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "120", "last_subgoal": "Step 3: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["84", "42", "120"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Remember to write legibly and neatly for proper scannability. The final answer is: 21. To solve this problem step by step:\n\nStep 1: Calculate the number of pages Julie has already read.\n\nJulie read..."}`

### plan_endpoint_mismatch
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "2"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Find out how many clips Natalia sold in May: \n   - We are given that Natalia sold twice the amount in May as compared to April. Since she s..."}`

### plan_parse_failed
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "", "last_subgoal": "", "step_answers": [], "flags": ["invalid_rollout", "non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": ""}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "", "last_subgoal": "", "step_answers": [], "flags": ["invalid_rollout", "non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": ""}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "", "last_subgoal": "", "step_answers": [], "flags": ["invalid_rollout", "non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": ""}`

### wrong_numeric_final
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "792", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["6", "12", "792"], "flags": [], "last_reasoning": "Final answer: 792 To determine how many pages James writes in a year, we need to follow these steps:\n\n### Step 1: Calculate the number of letters written per week.\nJames writes a letter to two diff..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "3", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["12", "12", "3"], "flags": [], "last_reasoning": "Let's solve it step-by-step:\n\n### Step 1: Calculate the number of letters written per week.\nJames writes a letter to each friend and sends these two letters twice a week.\n- Number of letters per we..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "52", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["3", "312", "52"], "flags": [], "last_reasoning": "To solve this step-by-step:\n\n### Step 1: Calculate the number of letters written per week.\nJames writes a 3-page letter to 2 different friends twice a week. \n- Number of letters per friend = 2 (tim..."}`
