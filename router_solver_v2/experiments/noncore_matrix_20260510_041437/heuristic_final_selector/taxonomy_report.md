# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/noncore_matrix_20260510_041437/heuristic_final_selector/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.3500`
- relaxed_numeric_accuracy: `0.3500`
- analyzed_failures: `39`
- invalid_counts: `{}`
- final_answer_source_counts: `{"heuristic_selector": 60}`

## Primary Categories
- copied_intermediate_as_final: `16`
- wrong_numeric_final: `12`
- correct_number_in_trace_wrong_final: `8`
- plan_endpoint_mismatch: `3`

## Flags
- copied_intermediate_as_final: `17`
- last_subgoal_not_answer_like: `9`
- correct_number_appears_in_trace: `8`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["6", "50"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To determine how much Weng earned from babysitting for 50 minutes yesterday, follow these steps:\n\n1. Convert the time from minutes to hours: \n   - There are 60 minutes in one hour.\n   - Therefore, ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "80", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["65", "65", "80", "65"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we will follow the plan outlined:\n\n1. Calculate how much money Betty currently has:\n   - Betty initially had half the cost of the wal..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["65", "95", "50", "50"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem, let's follow the plan step-by-step:\n\n1. Calculate how much money Betty currently has.\n   - The cost of the wallet is $100.\n   - Betty has half of what she needs.\n   - Therefo..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "12", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["10", "60"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To convert the 50 minutes into hours, you need to divide it by 60 (since there are 60 minutes in an hour). \n\nSo the calculation is:\n\n\\[ \\text{Number of hours} = \\frac{\\text{Total time in minutes}}{..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "1", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["6", "10"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The final answer is:\n\nFinal answer: $7.20 Step 2: Multiply the number of hours by her hourly rate.\nAnswer: She earned \\( 0.83 \\times \\$12 = \\$10 \\) (since 50 minutes is half an hour). \n\nFinal answe..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Step 3: Determine how much money Betty's grandparents gave her.", "step_answers": ["50", "30", "30", "30", "30", "2", "30", "50"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To solve this problem systematically:\n\n### Plan:\n1. Step 1: Calculate how much money Betty currently has.\n2. Step 2: Determine how much money Betty's parents gave her.\n3. Step 3: Determine how much..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "4.1667", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["0.8333", "0.8333"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we will follow the plan and perform each step as outlined:\n\nStep 1: Convert the time from minutes to hours.\n- We know that there are 60 minutes in an hour.\n- Weng babysat for..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "50", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["6", "60"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To determine how much Weng earned from her 50-minute babysitting session yesterday, we will follow these steps:\n\n1. Convert the time from minutes to hours:\n   - There are 60 minutes in one hour.\n  ..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "24", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["2", "12", "36", "2"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "In your final answer, do not include any extraneous information unrelated to solving the problem or answering the question at hand. To determine the number of pages Julie needs to read tomorrow, we..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "95", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["65", "30", "65", "95"], "flags": [], "last_reasoning": "To solve the problem step-by-step:\n\n### Step 1: Calculate how much money Betty currently has.\nBetty initially has half of what the wallet costs, so:\n\\[ \\text{Amount Betty currently has} = \\frac{\\$1..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["55", "70", "65", "50"], "flags": [], "last_reasoning": "To solve the problem, let's follow the steps:\n\n### Step-by-Step Solution\n\nStep 1: First, we calculate how much money Betty currently has.\nBetty initially has half the money needed for the wallet. T..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "2912", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["12", "1456", "2912"], "flags": [], "last_reasoning": "To solve the final step:\n\nStep 3 (Multiply the result from Step 2 by the number of times James writes each week):\n\\[ 1456 \\text{ pages/week} \\times 2 \\text{ times/week} = 2912 \\text{ pages/year} \\]..."}`
