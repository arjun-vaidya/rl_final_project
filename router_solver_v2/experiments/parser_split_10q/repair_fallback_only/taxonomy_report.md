# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/parser_split_10q/repair_fallback_only/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.1833`
- relaxed_numeric_accuracy: `0.1833`
- analyzed_failures: `49`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 60}`

## Primary Categories
- wrong_numeric_final: `19`
- copied_intermediate_as_final: `13`
- correct_number_in_trace_wrong_final: `11`
- plan_endpoint_mismatch: `6`

## Flags
- last_subgoal_not_answer_like: `15`
- copied_intermediate_as_final: `14`
- correct_number_appears_in_trace: `11`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To find out how many clips Natalia sold in May, we need to calculate half of the number of clips she sold in April. Since she sold 48 clips in April, we divide 48 by 2:\n\n$$ \\text{Clips Sold in May}..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "95", "last_subgoal": "Subtract the total amount from the cost of the wallet: Finally, we subtract the total amount of money Betty now has f...", "step_answers": ["30", "95", "10", "95"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, let's follow the steps provided:\n\n### Step 1: Calculate how much money Betty already has\n\nBetty has half of the money needed for the ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Add up all the money Betty now has: We add", "step_answers": ["50", "2", "30", "50"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, let's follow these steps:\n\nStep 1: Calculate how much money Betty already has\nBetty has only half of the money she needs ($100).\n\\[ \\..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "120", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "120"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Natalia sold 72 clips in May, so we add that to the 48 clips she sold in April:\n\n\\(48 + 72 = 120\\)\n\nFinal answer: 120"}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Step 3: Calculate how much money Betty's grandparents gave her.", "step_answers": ["50", "30", "30", "95", "95", "5", "30", "30"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To solve the problem, let's follow the steps:\n\n1. Calculate how much money Betty already has: \n   - The wallet costs $100.\n   - Betty has only half of the money needed, so we calculate \\( \\frac{100..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "36", "last_subgoal": "Step 3: Calculate how many pages Julie will read tomorrow.", "step_answers": ["42", "3", "36"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n### Step 1: Calculate the number of pages Julie has already read.\nJulie reads 12 pages yesterday. Today, she reads twice as many pages as yest..."}`

### plan_endpoint_mismatch
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "2"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To find out how many clips Natalia sold in May, we need to calculate half of the number of clips sold in April. Since she sold 48 clips in April, we divide 48 by 2:\n\n$$\n\\text{Clips sold in May} = \\..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "12", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["6", "12"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To determine how much Weng earned yesterday, we need to follow these steps:\n\nStep 1: Convert the time from minutes to hours.\n- We know that there are 60 minutes in one hour.\n- Weng worked for 50 mi..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "8", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["3", "8"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To convert the 50 minutes into hours, you divide by 60 since there are 60 minutes in an hour. So:\n\n\\[ \\text{Time in hours} = \\frac{\\text{Minutes}}{60} = \\frac{50}{60} \\]\n\nNow, multiply this result ..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "2", "last_subgoal": "Subtract the total amount from the cost of the wallet.", "step_answers": ["65", "95", "1", "2"], "flags": [], "last_reasoning": "To solve this problem systematically, let's follow the steps provided:\n\n### Step-by-Step Solution\n\n#### Step 1: Calculate how much money Betty currently has.\nWe know that Betty needs $100 for the w..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "65", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["95", "95", "50", "65"], "flags": [], "last_reasoning": "To solve this problem systematically:\n\nGiven:\n- Cost of the wallet = $100\n- Betty has half the money needed: $100 / 2 = $50\n- Parents' contribution: $15\n- Grandparents' contribution (twice the pare..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "4", "last_subgoal": "Step 4: Calculate how many pages Julie needs to read tomorrow.", "step_answers": ["36", "84", "24", "4"], "flags": [], "last_reasoning": "To find out how many pages Julie needs to read tomorrow, we can follow these steps:\n\n1. Calculate the number of pages Julie read today:  \n   Today, she read twice as many pages as she did yesterday..."}`
