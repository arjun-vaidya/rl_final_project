# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/heuristic_plus_selfconsistency_10q_20260510_073155/heuristic_plus_selfconsistency_3x/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2333`
- relaxed_numeric_accuracy: `0.2333`
- analyzed_failures: `46`
- invalid_counts: `{}`
- final_answer_source_counts: `{"heuristic_selector": 60}`

## Primary Categories
- wrong_numeric_final: `22`
- correct_number_in_trace_wrong_final: `11`
- plan_endpoint_mismatch: `7`
- copied_intermediate_as_final: `6`

## Flags
- correct_number_appears_in_trace: `11`
- last_subgoal_not_answer_like: `11`
- copied_intermediate_as_final: `6`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "36", "last_subgoal": "Step 3: Determine how many pages Julie plans to read tomorrow by finding half of the remaining pages.", "step_answers": ["84", "36", "36"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how many pages Julie should read tomorrow, we need to follow these steps:\n\n1. Calculate the total number of pages Julie has already read:\n   - Yesterday, she read 12 pages.\n   - Today,..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "52", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["4410", "52", "52"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Calculate the number of letters written per week.\n   - James writes a 3-page letter to 2 different friends twice a week.\n   - The total num..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "28", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["90", "0.25", "28", "28"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine the total number of flowers in Mark's garden, we need to follow each step carefully:\n\n### Step 1: Calculate the number of purple flowers.\nThe problem states that there are 80% more pur..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "24"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we will follow the plan given:\n\nStep 1: We are told that Natalia sold clips to 48 friends in April. In May, she sold half as many clips as she did in April. So, let's find ou..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "120", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "120"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "In your work showing these two steps, use only numbers and mathematical symbols (no words). Do not write \"Step\" before the first numbered step. To solve the problem:\n\n1. First, we need to find out ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "10", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["5", "2", "45", "100"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "The following is the first part of the solution:\n\nStep 4: Step 4: Subtract the total from the cost of the wallet.\nAnswer: To find out how much more money Betty needs, we'll follow these calculation..."}`

### plan_endpoint_mismatch
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "5", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "5"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "Natalia sold clips to her friends in four different ways:\n\nFirst method: She sold to her friends on April $9^{th}$ at $5$ dollars per clip.\n\nSecond method: She sold to her friends on April $16^{th}..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "0.8333", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["6", "0.8333"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this step, let's follow the plan:\n\n1. Convert the time from minutes to hours.\n   - We know that there are 60 minutes in an hour.\n   - Weng babysat for 50 minutes yesterday.\n\n   Now, we con..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["50", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The numbers used should be exact (e.g., 43 instead of 43. ) Do not round answers. If necessary, omit units in your calculations and your final answer. To solve this step:\n\nTo find out how much Weng..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["80", "65", "95", "30"], "flags": [], "last_reasoning": "To solve the problem step-by-step:\n\n1. Calculate how much money Betty currently has:\n   - The wallet costs $100.\n   - Betty has only half of what she needs.\n   - Therefore, Betty currently has \\( \\..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "2", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["50", "30", "95", "2"], "flags": [], "last_reasoning": "Remember you are writing code; do not include comments unless explicitly instructed to do so. To solve this problem systematically using Python, we will follow the outlined steps:\n\n```python\n# Init..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "25", "last_subgoal": "Subtract the total from the cost of the wallet: Finally, we subtract the total amount of money Betty now has (from st...", "step_answers": ["15", "1", "65", "2"], "flags": [], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, let's follow the plan:\n\nStep 1: Calculate how much money Betty already has\n\nSince Betty has only half of the money she needs ($100):\n..."}`
