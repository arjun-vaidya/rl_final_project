# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/constrained_final_decode_10q/constrained_final_decode/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2333`
- relaxed_numeric_accuracy: `0.2333`
- analyzed_failures: `46`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 48, "synthesis_constrained": 12}`

## Primary Categories
- wrong_numeric_final: `19`
- correct_number_in_trace_wrong_final: `14`
- copied_intermediate_as_final: `8`
- plan_endpoint_mismatch: `5`

## Flags
- last_subgoal_not_answer_like: `17`
- correct_number_appears_in_trace: `14`
- copied_intermediate_as_final: `12`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "Natalia sold <number> clips in total in April and May. To solve for the total number of clips Natalia sold in both April and May, we follow these steps:\n\n1. Find out how many clips Natalia sold in ..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow the given steps:\n\nStep 1: Determine the number of clips Natalia sold in May.\n- We know that Natalia sold 24 clips in May because she sold half as many as in..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "65", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["65", "50", "65", "65"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, let's follow the plan step-by-step:\n\n1. Calculate how much money Betty currently has:\n   - The problem states that Betty has \"only ha..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem, we need to follow these steps:\n\n1. Determine how many clips Natalia sold in May.\n2. Calculate the total number of clips sold in both April and May.\n\nStep 1: Find out how many ..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "2", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["6", "2"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "You may use either long or short division to perform this calculation. To determine how much Weng earned for her babysitting yesterday, we will follow these steps:\n\n1. Convert the time from minutes..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "4", "last_subgoal": "Step 3: Add the amount given by her grandparents.", "step_answers": ["5", "30", "15", "30", "95", "50", "30", "4"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we will follow the steps outlined:\n\n1. Calculate how much money Betty currently has.\n   - Betty has half of the $100 wallet cost: \n  ..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "last_subgoal": "Multiply the number of hours by her hourly rate.", "step_answers": ["0.8333", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The previous lines are intended only to give context or instructions. No additional writing is required on those lines. The next task will be introduced with \"Question:\" followed by \"Plan:\" and so ..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "9.9", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["0.8333", "12"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem, follow these steps:\n\n1. Convert the time from minutes to hours:  \n   - Given: Weng babysat for 50 minutes.\n   - There are 60 minutes in one hour.\n   - Conversion formula: \n   ..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "60", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["2", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, let's follow the plan and perform each step carefully:\n\n### Step-by-Step Solution:\n\nStep 1: Convert the time from minutes to hours.\n\nWe know that there are 60 minutes in one ..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "6", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["50", "25", "95", "6"], "flags": [], "last_reasoning": "To solve this problem, we will follow the plan provided:\n\n### Step-by-Step Solution:\n\n1. Calculate how much money Betty currently has:\n   - The wallet costs $100.\n   - Betty has only half of the mo..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "95", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["65", "25", "50", "95"], "flags": [], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we will follow the plan provided:\n\nStep 1: Calculate how much money Betty already has.\nBetty initially had half of the required $100,..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Step 3: Subtract the total amount from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["45", "55", "30"], "flags": [], "last_reasoning": "To solve this problem systematically, let's go through each step:\n\n### Step 1: Calculate how much money Betty currently has.\nWe are told that Betty initially had only half of what was needed for th..."}`
