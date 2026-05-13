# Failure Taxonomy Report

## Summary
- trace_path: `/home/machina/pvd2112/rl_final_project/router_solver_hierarchical_pivot/experiments/router_objective_credit_o2_streamed_20260513_063639/o2_all_steps/eval/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `50`
- valid_rollouts: `50`
- exact_accuracy: `0.4000`
- relaxed_numeric_accuracy: `0.4000`
- question_majority_exact_accuracy: `0.5000`
- question_majority_relaxed_accuracy: `0.5000`
- question_any_relaxed_accuracy: `0.7000`
- analyzed_failures: `30`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 50}`

## Primary Categories
- correct_number_in_trace_wrong_final: `19`
- wrong_numeric_final: `8`
- copied_intermediate_as_final: `3`

## Flags
- correct_number_appears_in_trace: `19`
- copied_intermediate_as_final: `8`
- last_subgoal_not_answer_like: `1`

## Example Failures
### copied_intermediate_as_final
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "1460", "last_subgoal": "Step 2: Determine how many days are in a year.", "step_answers": ["2", "1460", "365", "364", "208", "4380", "3", "1460"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how many pages James writes in a year, we need to follow these steps:\n\nStep 1: Calculate how many letters James writes in one day.\n- James writes 2 letters to each friend twice a week...."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "143", "last_subgoal": "Calculate the total amount Alexis spent on all items except the shoes.", "step_answers": ["117", "117", "46", "143", "75", "73", "3", "143"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "The first letter of your answer should be capitalized, as required by instructions. To solve the problem:\n\nGiven:\n- Budget = $200\n- Amount spent on each item (except shoes):\n  - Button-up shirt: $3..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "54", "last_subgoal": "If yes, calculate her overtime pay. Then answer the original question directly using the previous results.", "step_answers": ["180", "5", "720", "18", "180", "54", "720", "54"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much money Tina makes over 5 days, we need to follow these steps:\n\nStep 1: Calculate Tina's regular pay.\n\nTina earns $18.00 per hour and works 8 hours each day. Therefore, her dail..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the clips sold in April and May together. Then answer the original question directly using the previous r...", "step_answers": ["24", "72", "2", "1", "72", "72", "72", "2"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "Do not include any extra spaces before or after the equal signal in the final answer.\n\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. To find out how..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "45", "last_subgoal": "Step 3: Determine how much money Betty's grandparents gave her.", "step_answers": ["95", "95", "40", "30", "30", "30", "100", "45"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "Plan Step 3:\n\nTo determine how much money Betty's grandparents gave her, we start by calculating the total amount of money Betty already has. We know that Betty has half of the money needed for the..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "40", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["95", "5", "65", "5", "95", "95", "50", "40"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "We will use your method to verify your solution. To find out how much more money Betty needs to buy the wallet, let's follow the steps provided:\n\n### Plan:\n1. Step 1: Calculate how much money Betty..."}`

### wrong_numeric_final
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "4", "last_subgoal": "Find out how many purple flowers there are:", "step_answers": ["4"], "flags": [], "last_reasoning": "To solve the problem, let's follow the given information:\n\n1. Number of yellow flowers: 10\n2. Percentage increase for purple flowers (compared to yellow): 80%\n\nFirst, calculate the number of purple..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "80", "last_subgoal": "Step 4: Sum up all the flowers to get the total number of flowers in the garden.", "step_answers": ["28", "28", "10", "28", "18", "28", "28", "80"], "flags": [], "last_reasoning": "To determine the total number of flowers in Mark's garden, we will follow the given plan step-by-step:\n\nStep 1: Calculate the number of purple flowers.\n- We know that there are 10 yellow flowers.\n-..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "10", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["28", "28", "18", "3", "28", "51", "4", "10"], "flags": [], "last_reasoning": "To solve the problem, we will follow the given steps:\n\nStep 1: Calculate the number of purple flowers.\n- We know that there are 10 yellow flowers and 80% more purple flowers than yellow flowers.\n- ..."}`
