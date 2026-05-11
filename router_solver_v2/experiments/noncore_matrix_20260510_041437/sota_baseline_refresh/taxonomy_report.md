# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/noncore_matrix_20260510_041437/sota_baseline_refresh/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2500`
- relaxed_numeric_accuracy: `0.2500`
- analyzed_failures: `45`
- invalid_counts: `{}`
- final_answer_source_counts: `{"synthesis": 60}`

## Primary Categories
- wrong_numeric_final: `20`
- correct_number_in_trace_wrong_final: `10`
- non_numeric_final_answer: `7`
- copied_intermediate_as_final: `6`
- plan_endpoint_mismatch: `2`

## Flags
- correct_number_appears_in_trace: `14`
- copied_intermediate_as_final: `10`
- non_numeric_final_answer: `7`
- last_subgoal_not_answer_like: `6`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To find out how many clips Natalia sold in May, we need to follow these steps:\n\nStep 1: First, determine the number of clips sold in April by noting that it is given as 48.\n\\[ \\text{Clips sold in A..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "18", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["28", "18", "0.25", "18"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem, we need to follow the given steps:\n\nStep 1: Calculate the number of purple flowers.\n\nWe know that there are 10 yellow flowers and that the number of purple flowers is 80% mor..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "28", "last_subgoal": "Step 4: Sum up all the flowers to get the total count.", "step_answers": ["28", "10", "1", "28"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine the total number of flowers in Mark's garden, we need to follow the plan provided:\n\n### Step 1: Calculate the number of purple flowers.\nMark initially planted ten yellow flowers. The p..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "100", "last_subgoal": "Step 2: Determine how much money Betty's parents gave her.", "step_answers": ["50", "15", "100", "30", "-5", "5", "30", "50"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To solve this problem, let's go through each step carefully:\n\nStep 1: Calculate how much money Betty currently has.\n- We know that Betty needs $100 and she already has half of it, so we calculate:\n..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "25", "last_subgoal": "Subtract the total from the cost of the wallet: We subtract the sum of the amounts received from her parents and gran...", "step_answers": ["4", "5", "30", "2"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, let's follow the plan:\n\n1. Calculate how much money Betty currently has: Since Betty has only half of the money she needs ($100), we ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Step 3: Determine how much money Betty's grandparents gave her.", "step_answers": ["100", "50", "30", "15", "50", "15", "95", "30"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To solve the problem, let's go through each step systematically:\n\n### Step 1: Calculate how much money Betty currently has.\nBetty has already saved some money, but we don't know the exact amount ye..."}`

### non_numeric_final_answer
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "Instead, continue with further calculations if necessary. If there is no final answer based on the provided informati...", "last_subgoal": "Subtract the total amount from the cost of the wallet: The", "step_answers": ["50", "65", "9", "65"], "flags": ["non_numeric_final_answer"], "last_reasoning": "Here's how you can proceed:\n\nSolution:\n\n1. Calculate how much money Betty currently has:\n   - Betty has half of the money needed for the wallet.\n   \\[\n   \\text{Amount Betty has} = \\frac{\\$100}{2} =..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "Therefore, the final answer is:", "last_subgoal": "Step 3: Calculate the number of pages Julie plans to read tomorrow.", "step_answers": ["42", "84", "84"], "flags": ["non_numeric_final_answer", "correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Calculate the total number of pages Julie has already read.\n   - Yesterday, Julie read 12 pages.\n   - Today, Julie read twice as many pages..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "Jupyter will evaluate all cell outputs.", "last_subgoal": "Subtract the number of pages she has already read from Step 3 to find out how many pages she needs to read tomorrow.", "step_answers": ["42", "42", "21", "2", "36", "36", "36", "9"], "flags": ["non_numeric_final_answer", "correct_number_appears_in_trace"], "last_reasoning": "The given information does not contain any additional steps for solving the question.\n\nNote by Khoi Vo\n3\u00a0years, 9\u00a0months ago\n\nThis discussion board is a place to discuss our Daily Challenges and th..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "1", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this step-by-step, let's follow the plan:\n\n### Step 1: Convert the time from minutes to hours.\n- Given: Weng worked for 50 minutes yesterday.\n- There are 60 minutes in one hour.\n- To conve..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "5", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["6", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "Let's proceed with the next step:\n\nTo convert the time from minutes to hours, we use the fact that there are 60 minutes in an hour. So, we divide the number of minutes by 60.\n\n\\[ \\text{Number of ho..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "95", "last_subgoal": "Finally, we subtract the total amount of money Betty now has from the cost of the wallet ($100) to find out how much ...", "step_answers": ["4", "30", "50", "95"], "flags": [], "last_reasoning": "To solve this problem, let's follow the plan:\n\nStep 1: Calculate how much money Betty currently has.\n\n- Betty needs $100 and she has only half of it.\n\\[ \\text{Money Betty has} = \\frac{\\$100}{2} = \\..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "1", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["50", "30", "30", "30"], "flags": [], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we will follow the plan outlined:\n\n### Plan:\n1. Calculate how much money Betty already has: \n   - Betty initially has half the cost o..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "36", "last_subgoal": "Step 3: Determine how many pages Julie will read tomorrow by taking half of the remaining pages.", "step_answers": ["84", "84", "36"], "flags": [], "last_reasoning": "Remember to show your work for full credit. To determine how many pages Julie should read tomorrow, we need to follow these steps:\n\n1. Calculate the total number of pages Julie has already read:\n  ..."}`
