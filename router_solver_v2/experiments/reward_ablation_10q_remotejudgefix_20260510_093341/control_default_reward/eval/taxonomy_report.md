# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/reward_ablation_10q_remotejudgefix_20260510_093341/control_default_reward/eval/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2833`
- relaxed_numeric_accuracy: `0.2833`
- analyzed_failures: `43`
- invalid_counts: `{}`
- final_answer_source_counts: `{"synthesis": 60}`

## Primary Categories
- wrong_numeric_final: `15`
- correct_number_in_trace_wrong_final: `10`
- copied_intermediate_as_final: `7`
- non_numeric_final_answer: `6`
- plan_endpoint_mismatch: `5`

## Flags
- correct_number_appears_in_trace: `12`
- last_subgoal_not_answer_like: `10`
- copied_intermediate_as_final: `9`
- non_numeric_final_answer: `6`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "84", "last_subgoal": "Step 4: Calculate how many pages Julie needs to read tomorrow.", "step_answers": ["24", "36", "84", "84"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how many pages Julie needs to read tomorrow, we can follow these steps:\n\n1. Total pages in the book: 120 pages\n2. Pages read yesterday: 12 pages\n\nToday's reading:\n   - Pages read today..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "2", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["52", "2", "52"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "We are looking for only the numerical solution here.\n\nTo calculate the number of pages James writes in a year:\n\n1. Number of pages per letter = 3\n2. Number of friends = 2\n3. Number of times written..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "28", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["18", "18", "28", "18"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To find out how many flowers Mark has in his garden, we need to follow these steps:\n\nStep 1: Calculate the number of purple flowers.\n\nWe know that Mark has 10 yellow flowers and that there are 80% ..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "9.6", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["45", "12"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To convert the time from minutes to hours:\n\n- Divide the total minutes (50) by 60 since there are 60 minutes in one hour.\n- \\( \\frac{50}{60} = 0.8333... \\)\n\nNow multiply this result by her hourly r..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "35", "last_subgoal": "Add up all the money Betty now has: We add the money Betty currently has (from step 1", "step_answers": ["30", "30", "30", "30"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem, let's break down each step:\n\n### Step 1: Calculate how much money Betty currently has\nBetty already has half of the money needed for the wallet, which is $100.\n\\[ \\text{Money ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "95", "last_subgoal": "Step 3: Determine how much money Betty's grandparents gave her.", "step_answers": ["50", "5", "15", "30", "95", "50", "85", "95"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To solve this problem, let's go through each step carefully:\n\n### Step-by-Step Solution:\n\nStep 1: Calculate how much money Betty already has.\n- The wallet costs $100.\n- Betty has only half of the m..."}`

### non_numeric_final_answer
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "### Corrected Plan:", "last_subgoal": "Step 3: Determine how much money Betty's grandparents gave her.", "step_answers": ["50", "5", "95", "30", "50", "100", "30", "5"], "flags": ["non_numeric_final_answer", "correct_number_appears_in_trace"], "last_reasoning": "To solve this problem systematically:\n\nStep-by-Step Solution\n\n1. Initial Calculation: Betty needs $100 for the wallet but has only half of it, so we calculate:\n   \\[\n   \\text{Money Betty currently ..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "If there are multiple possible answers, return the largest one. In cases where the question has no clear numerical an...", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["3", "6", "728"], "flags": ["non_numeric_final_answer"], "last_reasoning": "To solve the problem step-by-step:\n\n### Step 1: Calculate the number of letters written per week.\nJames writes a 3-page letter to 2 different friends twice a week. \n\n- Number of letters per day = \\..."}`
- `{"question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a sm...", "ground_truth": "48", "final_answer": "The final answer is:", "last_subgoal": "Step 3: Add the two totals together to get the final answer.", "step_answers": ["32", "40", "48"], "flags": ["non_numeric_final_answer", "correct_number_appears_in_trace"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n### Step-by-Step Solution:\n\nStep 1: Calculate the total number of slices from the large pizzas.\n- Number of large pizzas = 2\n- Slices per larg..."}`

### plan_endpoint_mismatch
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "12", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "2"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem:\n\n1. Find out how many clips Natalia sold in May:  \n   Natalia sold half as many clips in May as she did in April. Since she sold 24 clips in May, we can calculate it as follow..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "96", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we'll follow these steps:\n\nStep 1: Calculate the number of clips Natalia sold in May.\n- We know that Natalia sold 48 clips in April.\n- In May, she sold half as many clips as ..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "9.96", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["0.83", "9.96"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To convert the 50 minutes into hours:\n\n\\[\n\\text{Hours} = \\frac{\\text{Minutes}}{60}\n\\]\n\n\\[\n\\text{Hours} = \\frac{50}{60}\n\\]\n\n\\[\n\\text{Hours} = 0.8333...\n\\]\n\nTo calculate the earnings:\n\n\\[\n\\text{Earni..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "100", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["30", "65", "65", "50"], "flags": [], "last_reasoning": "Let's solve the problem step-by-step.\n\n### Step 1: Calculate how much money Betty already has.\nBetty needs a total of $100. She currently has half of this amount:\n\\[\n\\frac{1}{2} \\times 100 = 50\n\\]\n..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "84", "last_subgoal": "Step 3: Determine how many pages Julie plans to read tomorrow by finding half of the remaining pages.", "step_answers": ["3", "2", "84"], "flags": [], "last_reasoning": "To solve the problem, let's go through each step:\n\nStep 1: Calculate the total number of pages Julie has already read.\nJulie read 12 pages yesterday and twice as many today, which is \\(2 \\times 12 ..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "100", "last_subgoal": "Step 4: Sum up all the flowers to get the total number.", "step_answers": ["10", "28", "25", "100"], "flags": [], "last_reasoning": "To solve the problem, we will follow the given steps:\n\nStep 1: Calculate the number of purple flowers.\n\n- Yellow flowers: 10\n- Purple flowers (80% more than yellow): \n\\[ \\text{Purple} = 10 + (10 \\t..."}`
