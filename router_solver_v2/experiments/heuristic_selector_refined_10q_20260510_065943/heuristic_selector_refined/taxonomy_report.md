# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/heuristic_selector_refined_10q_20260510_065943/heuristic_selector_refined/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2000`
- relaxed_numeric_accuracy: `0.2000`
- analyzed_failures: `48`
- invalid_counts: `{}`
- final_answer_source_counts: `{"heuristic_selector_refined": 43, "synthesis": 17}`

## Primary Categories
- wrong_numeric_final: `24`
- correct_number_in_trace_wrong_final: `15`
- copied_intermediate_as_final: `6`
- plan_endpoint_mismatch: `3`

## Flags
- correct_number_appears_in_trace: `15`
- last_subgoal_not_answer_like: `14`
- copied_intermediate_as_final: `11`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, let's follow the plan:\n\n1. Step 1: First, we need to determine how many clips Natalia sold in May. According to the information given, Natalia sold half as many clips in May ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "95", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["65", "4", "95", "95"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Let's solve the problem step-by-step:\n\nStep 1: Calculate how much money Betty already has.\nAnswer: Betty has half of what she needs, so she has \\( \\frac{1}{2} \\times 100 = 50 \\) dollars.\n\nStep 2: A..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Finally, we subtract the total amount from the cost of the wallet ($100) to find out how much more money Betty needs.", "step_answers": ["30", "95", "50", "50"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem, let's follow the plan:\n\nStep 1: Calculate how much money Betty currently has.\n- Betty initially has half the money needed for the wallet.\n- The wallet costs $100, so Betty ne..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "24"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The given problem involves basic arithmetic operations and understanding of proportions. Here's the breakdown for each step:\n\n### Step 1: Finding the number of clips sold in May\nNatalia sold twice ..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "2", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["2", "2"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "The correct calculation is:\n\nStep 2: Multiply the number of hours by the hourly rate.\n\nTo convert 50 minutes into hours, we divide by 60 since there are 60 minutes in an hour:\n\n$$\n\\frac{50}{60} = \\..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["0.8333", "6"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this step-by-step:\n\nStep 1: Convert the time from minutes to hours.\nWeng babysat for 50 minutes yesterday. Since there are 60 minutes in an hour, we can convert the time as follows:\n\n\\[\n\\t..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "9.999", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["0.833", "60"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The final answer is:\n\nFinal answer: 9.999\nShow Work: \nTo convert the time from minutes to hours, we use the conversion factor that there are 60 minutes in one hour. Therefore, we divide the number ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "2", "last_subgoal": "Add up all the money Betty now has: We add together", "step_answers": ["75", "30", "30", "2"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The next problem begins below:\n\nWork:\n\n---\n\n1. Calculate how much money Betty already has: Since Betty has only half of the money she needs ($100), we can calculate how much she currently has by di..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "72", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["36", "36", "49", "72"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To determine how many pages Julie needs to read tomorrow, we will follow the plan outlined:\n\nStep 1: Calculate the number of pages Julie read today.\nJulie read 12 pages yesterday and 2 * 12 = 24 pa..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "4", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["50", "35", "30", "30"], "flags": [], "last_reasoning": "The number will be the answer to the original question. To solve this problem, let's go through each step carefully:\n\n1. Calculate how much money Betty currently has:\n   - Betty initially has half ..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "52", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["6", "6", "52"], "flags": [], "last_reasoning": "Let's go through the steps systematically:\n\n### Step 1: Calculate the number of letters written per week\n\nJames writes to two different friends, and he sends a 3-page letter to each friend. Therefo..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "249", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["1.5", "6", "249"], "flags": [], "last_reasoning": "Let's solve this step-by-step:\n\nGiven information:\n- Each letter is 3 pages long.\n- James writes 2 letters per friend (so 4 letters total).\n- He writes these letters twice a week.\n- There are appro..."}`
