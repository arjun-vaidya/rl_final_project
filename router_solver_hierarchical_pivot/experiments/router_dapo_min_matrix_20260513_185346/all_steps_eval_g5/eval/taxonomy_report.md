# Failure Taxonomy Report

## Summary
- trace_path: `/home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/router_dapo_min_matrix_20260513_185346/all_steps_eval_g5/eval/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `50`
- valid_rollouts: `50`
- exact_accuracy: `0.2200`
- relaxed_numeric_accuracy: `0.2200`
- question_majority_exact_accuracy: `0.3000`
- question_majority_relaxed_accuracy: `0.3000`
- question_any_relaxed_accuracy: `0.7000`
- analyzed_failures: `39`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 50}`

## Primary Categories
- correct_number_in_trace_wrong_final: `22`
- wrong_numeric_final: `13`
- copied_intermediate_as_final: `4`

## Flags
- correct_number_appears_in_trace: `22`
- copied_intermediate_as_final: `12`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the ...", "ground_truth": "70000", "final_answer": "150", "last_subgoal": "Calculate the profit by subtracting the initial cost from the new value.", "step_answers": ["130000", "120000", "0", "205000", "150", "130000", "13000", "150"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve the problem, let's follow the steps:\n\n### Step 1: Calculate the total cost of buying the house.\n- Initial purchase price: $80,000\n- Repair costs: $50,000\n\n\\[\n\\text{Total cost} = \\text{Init..."}`
- `{"question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a rest...", "ground_truth": "160", "final_answer": "50", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Carla is dow...", "step_answers": ["70", "140", "140", "50", "50", "50"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve the problem, we need to calculate the total time Carla takes to download the 200 GB file, considering the interruptions caused by the restart.\n\n1. Calculate the time taken to download the ..."}`
- `{"question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a rest...", "ground_truth": "160", "final_answer": "50", "last_subgoal": "Sum up all the times to get the total time required to complete the download.", "step_answers": ["999", "50", "70", "50"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve the problem, let's break down each part step by step:\n\n### Step 1: Calculate the time taken for the first half of the download.\nCarla normally downloads at a speed of 2 GB per minute. The ..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She ...", "ground_truth": "18", "final_answer": "13", "last_subgoal": "Step 4: Multiply the number of eggs sold by the price per egg to find out how much money is made.", "step_answers": ["9", "14", "13", "18", "13", "12", "14", "13"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "Plan steps should be numbered, separated by commas, and start with \"Step\". Each subsequent step should have its own unique identifier, such as \"Step 1\", \"Step 2\", etc. Answer the question directly ..."}`
- `{"question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She ...", "ground_truth": "18", "final_answer": "340", "last_subgoal": "Multiply the number of eggs sold by the price per egg to find the amount earned. Then answer the original question di...", "step_answers": ["34", "13", "4", "13", "44", "96", "34", "340"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "Plan step:\n\nStep 1: Step 1:\nAnswer: 34\n\nStep 2: Determine how many eggs are left after breakfast.\nAnswer: 13\n\nStep 3: Step 2:\nAnswer: 4\n\nStep 4: Calculate how many eggs are sold to friends.\nAnswer:..."}`
- `{"question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "ground_truth": "3", "final_answer": "1", "last_subgoal": "Half the amount of blue fiber Then answer the original question directly using the previous results.", "step_answers": ["1", "3", "1", "3", "3", "3", "3", "1"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "Plan step:\n\nStep 1: Calculate the amount of white fiber needed.\nAnswer: 3\n\nExplanation for each step:\n\nStep 1: We know that the robe requires 2 bolts of blue fiber and half that amount of white fib..."}`

### wrong_numeric_final
- `{"question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She ...", "ground_truth": "18", "final_answer": "10", "last_subgoal": "Step 4: Multiply the result from Step 3 by the price per egg ($2) to find out how much money she makes.", "step_answers": ["13", "5", "3", "9", "5", "8", "9", "10"], "flags": [], "last_reasoning": "The instructions will reject any submission that includes text after \"Final answer:\" Final answer: 10 To solve this problem, let's follow the steps:\n\nStep 1: Calculate how many eggs are left after ..."}`
- `{"question": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the ...", "ground_truth": "70000", "final_answer": "4", "last_subgoal": "Step 3: Determine the increase in value due to the repairs. Then answer the original question directly using the prev...", "step_answers": ["-5", "120000", "180000", "150", "1", "120000", "195000", "4"], "flags": [], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Calculate the total cost of buying the house: \n   The house costs $80,000 and Josh spends an additional $50,000 on repairs. Therefore, the ..."}`
- `{"question": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the ...", "ground_truth": "70000", "final_answer": "50000", "last_subgoal": "Step 3: Add the repair costs to the initial purchase price to find the total cost.", "step_answers": ["120000", "20", "80000", "5", "5000", "130000", "120000", "50000"], "flags": [], "last_reasoning": "The problem is:\n\nJosh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make? L..."}`
