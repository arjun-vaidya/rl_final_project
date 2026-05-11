# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/parser_split_10q/router_prompt_hardening_only/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `9`
- exact_accuracy: `0.0333`
- relaxed_numeric_accuracy: `0.0333`
- analyzed_failures: `58`
- invalid_counts: `{"plan_parse_failed": 51}`
- final_answer_source_counts: `{"last_step": 9, "none": 51}`

## Primary Categories
- plan_parse_failed: `51`
- correct_number_in_trace_wrong_final: `3`
- wrong_numeric_final: `3`
- copied_intermediate_as_final: `1`

## Flags
- last_subgoal_not_answer_like: `52`
- invalid_rollout: `51`
- non_numeric_final_answer: `51`
- correct_number_appears_in_trace: `3`
- copied_intermediate_as_final: `1`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "75", "last_subgoal": "The result will be the amount Alexis paid for the shoes.", "step_answers": ["20", "75", "75"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much Alexis paid for the shoes, we need to follow these steps:\n\n1. Add up all the expenses excluding the shoes:\n   - Button-up shirt: $30\n   - Suit pants: $46\n   - Suit coat: $38\n ..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "3", "last_subgoal": "Subtract the total amount of money Betty has from the cost of the wallet to find out how much more money she needs.", "step_answers": ["50", "2", "5", "30", "3"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we will follow these steps:\n\nStep 1: Determine how much money Betty currently has.\nBetty has saved half of what the wallet costs, so ..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "4", "last_subgoal": "Compute half of the remaining pages for tomorrow's reading.", "step_answers": ["24", "36", "84", "4"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "To determine the number of pages Julie should read tomorrow, we need to follow these steps:\n\n1. Calculate the number of pages Julie read today:\n   - Julie read 12 pages yesterday.\n   - Today, she r..."}`
- `{"question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a sm...", "ground_truth": "48", "final_answer": "64", "last_subgoal": "Add the two totals together to find the overall number of slices.", "step_answers": ["48", "16", "64"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The question is:\n\nFinal answer: 64\n\nShow work: Step 1: Calculate the total number of slices from large pizzas.\nAlbert bought 2 large pizzas. Each large pizza has 16 slices.\n\\[\n2 \\text{ large pizzas..."}`

### plan_parse_failed
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "", "last_subgoal": "", "step_answers": [], "flags": ["invalid_rollout", "non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": ""}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "", "last_subgoal": "", "step_answers": [], "flags": ["invalid_rollout", "non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": ""}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "", "last_subgoal": "", "step_answers": [], "flags": ["invalid_rollout", "non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": ""}`

### wrong_numeric_final
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "25", "last_subgoal": "Step 4: Sum up all the flowers.", "step_answers": ["18", "18", "3", "25"], "flags": [], "last_reasoning": "Let's solve this step by step:\n\n1. Calculate the number of purple flowers:\n   - We know there are 10 yellow flowers.\n   - There are 80% more purple flowers than yellow flowers.\n   - To find 80% of ..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "53", "last_subgoal": "The result will be the amount Alexis paid for the shoes.", "step_answers": ["47", "47", "53"], "flags": [], "last_reasoning": "It's okay if your final sentence makes sense before or after the final answer line.\nFinal answer: $53 Step 3: The result will be the amount Alexis paid for the shoes.\n\nTo solve for the amount Alexi..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "184", "last_subgoal": "Subtract the sum of the known costs (excluding the shoes) from the total amount spent to determine the cost of the sh...", "step_answers": ["125", "16", "153", "184"], "flags": [], "last_reasoning": "To solve this problem systematically:\n\n### Step-by-Step Solution\n\nStep 1: Calculate the total amount Alexis spent excluding the shoes.\n- Button-up shirt: $30\n- Suit pants: $46\n- Suit coat: $38\n- So..."}`
