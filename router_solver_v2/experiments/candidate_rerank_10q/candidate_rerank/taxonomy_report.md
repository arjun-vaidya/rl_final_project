# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/candidate_rerank_10q/candidate_rerank/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2833`
- relaxed_numeric_accuracy: `0.2833`
- analyzed_failures: `43`
- invalid_counts: `{}`
- final_answer_source_counts: `{"candidate_rerank": 60}`

## Primary Categories
- correct_number_in_trace_wrong_final: `15`
- wrong_numeric_final: `13`
- copied_intermediate_as_final: `12`
- plan_endpoint_mismatch: `3`

## Flags
- copied_intermediate_as_final: `21`
- correct_number_appears_in_trace: `15`
- last_subgoal_not_answer_like: `11`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Step 3: Subtract the total amount from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["30", "2", "45"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, let's follow the plan step-by-step:\n\n### Step 1: Calculate how much money Betty currently has.\nBetty initially has half of what she n..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "48", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["1.5", "48", "52"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how many pages James writes in a year, we will follow the steps outlined:\n\nStep 1: Calculate the number of letters written per week.\nJames writes a 3-page letter to 2 different friends..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "18", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["18", "1", "18", "3"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve the problem, let's follow each step:\n\n1. Calculate the number of purple flowers.\n   - We know that there are ten yellow flowers.\n   - There are 80% more purple flowers than yellow flowers...."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem step by step:\n\n1. Find out how many clips Natalia sold in May:\n   - According to the plan, Natalia sold half as many clips in May as she did in April.\n   - In April, Natalia so..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["5", "30", "30", "10"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "Let's solve this step-by-step:\n\n### Step 1: Calculate how much money Betty currently has.\nBetty has saved up half of what the wallet costs, so:\n\\[ \\text{Amount Betty has} = \\frac{\\$100}{2} = \\$50 \\..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "45", "last_subgoal": "The result will be the amount of money Betty still needs to buy the wallet.", "step_answers": ["30", "45", "50", "4", "96", "30"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To find out how much more money Betty needs to buy the wallet, we can follow these steps:\n\n1. Calculate how much money Betty already has: \n   - We know Betty initially had \\( \\frac{1}{2} \\) of what..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "60", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["6", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem:\n\nStep 1: Convert the time from minutes to hours.\nWeng's babysitting was done for 50 minutes. Since there are 60 minutes in an hour, we convert as follows:\n\\[ \\text{Time in ho..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "60", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["6", "60"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To convert the time from minutes to hours, you divide the number of minutes by 60 because there are 60 minutes in one hour. So,\n\n\\[ \\text{Number of hours} = \\frac{\\text{Number of minutes}}{60} \\]\n\n..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "60", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["8.33", "60"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem of calculating how much Weng earned for 50 minutes of babysitting at a rate of $12 per hour, let's follow the plan:\n\n### Step-by-Step Solution:\n\n1. Convert the time from minute..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "45", "last_subgoal": "Finally, we subtract the total amount from the cost of the wallet to determine how much more money Betty needs.", "step_answers": ["4", "55", "45"], "flags": [], "last_reasoning": "To find out how much more money Betty needs to buy the wallet, let's follow the plan step-by-step:\n\n### Step 1:\nFirst, we calculate how much money Betty currently has. According to the problem, Bet..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "12", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["52", "2", "12"], "flags": [], "last_reasoning": "Let's solve the problem step by step:\n\nStep 1: We know James writes a 3-page letter to 2 different friends twice a week.\n   - Number of pages per letter = 3\n   - Number of friends = 2\n   - Times pe..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "3", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["5.25", "273", "3"], "flags": [], "last_reasoning": "To solve the problem, let's follow the given steps:\n\n### Step 1: Calculate the number of letters written per week.\nJames writes a 3-page letter to 2 different friends twice a week. Therefore, we ne..."}`
