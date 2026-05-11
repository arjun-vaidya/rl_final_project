# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/noncore_matrix_20260510_041437/guarded_heuristic_fallback/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2500`
- relaxed_numeric_accuracy: `0.2500`
- analyzed_failures: `45`
- invalid_counts: `{}`
- final_answer_source_counts: `{"heuristic_guard_fallback": 3, "last_step": 1, "synthesis": 56}`

## Primary Categories
- correct_number_in_trace_wrong_final: `16`
- wrong_numeric_final: `14`
- copied_intermediate_as_final: `10`
- plan_endpoint_mismatch: `5`

## Flags
- copied_intermediate_as_final: `18`
- correct_number_appears_in_trace: `16`
- last_subgoal_not_answer_like: `12`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "48"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To find out how many clips Natalia sold in May, we need to take the number of clips sold in April (which is 48) and divide it by two since she sold half as many in May. This means:\n\n\\[\n\\text{Clips ..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "84", "last_subgoal": "Step 3: Determine how many pages Julie plans to read tomorrow by dividing the remaining pages by two.", "step_answers": ["24", "36", "36", "84", "24", "84"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem systematically, let's go through each step carefully:\n\n### Given Information:\n- Total pages in the book = 120 pages\n- Pages read yesterday = 12 pages\n- Pages read today = Twic..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "8", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["8", "672", "52"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Let's solve this step-by-step:\n\nWe already know:\n- James writes a 3-page letter to 2 different friends.\n- He does this twice a week.\n- There are 52 weeks in a year.\n\nLet's calculate how many pages ..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "1", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["10", "2"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The problem is over at this point.\n\nStep 2: Multiply the number of hours by the hourly rate.\n\nWeng earns $12 per hour and babysat for 10 minutes (which is equivalent to $\\frac{1}{6}$ of an hour).\n\n..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Step 2: Determine how much money Betty's parents gave her.", "step_answers": ["50", "50", "30", "95", "30", "5", "95", "4"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To solve this problem, we need to follow the steps provided:\n\n### Plan:\n1. Step 1: Calculate how much money Betty currently has.\n   - Betty initially has half the amount needed for the wallet.\n   -..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "55", "last_subgoal": "Step 3: Determine how much money Betty's grandparents gave her.", "step_answers": ["30", "55", "55", "95", "30", "95", "50", "95"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To solve this problem, we need to determine how much more money Betty needs to buy the wallet. Let's go through the steps:\n\n### Step-by-Step Solution\n\nStep 1: Calculate how much money Betty current..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "46.67", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["0.83333", "46.67"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To convert the time from minutes to hours, we divide the number of minutes by 60 (since there are 60 minutes in an hour). So, 50 minutes divided by 60 equals approximately 0.83333 hours. \n\nFinal an..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "60", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["0.83", "60"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To convert 50 minutes into hours, you need to divide it by 60 because there are 60 minutes in one hour. So,\n\n\\[ \\text{Time in hours} = \\frac{\\text{Time in minutes}}{60} = \\frac{50}{60} \\]\n\nNow, let..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "18", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["24", "84", "36", "36"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "In some countries that is considered rude or offensive. Final answer: 18\n\nTo determine how many pages Julie should read tomorrow, we need to follow these steps:\n\n1. Calculate the number of pages Ju..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "100", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["65", "30", "95", "30"], "flags": [], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, let's follow the plan:\n\n1. Step 1: Calculate how much money Betty already has.\n   - Betty initially had half of the $100 wallet.\n   \\..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "640", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["8", "2", "640"], "flags": [], "last_reasoning": "The final answer is:\n\nFinal answer: 640\n\nExplanation of the steps used for solving:\nThe total number of pages written per week is found first by multiplying the number of pages per letter (3) by th..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "67", "last_subgoal": "Step 4: Sum up all the flowers to get the total count.", "step_answers": ["10", "28", "18", "18"], "flags": [], "last_reasoning": "To solve this step-by-step:\n\nStep 1: Calculate the number of purple flowers.\n\nWe know that there are 80% more purple flowers than yellow flowers. Since there are 10 yellow flowers:\n\n\\[\n\\text{Number..."}`
