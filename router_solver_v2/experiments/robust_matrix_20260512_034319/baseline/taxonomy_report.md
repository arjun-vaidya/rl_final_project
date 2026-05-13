# Failure Taxonomy Report

## Summary
- trace_path: `/home/machina/pvd2112/rl_final_project/router_solver_v2/experiments/robust_matrix_20260512_034319/baseline/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `40`
- valid_rollouts: `40`
- exact_accuracy: `0.2000`
- relaxed_numeric_accuracy: `0.2000`
- analyzed_failures: `32`
- invalid_counts: `{}`
- final_answer_source_counts: `{"synthesis": 40}`

## Primary Categories
- correct_number_in_trace_wrong_final: `19`
- non_numeric_final_answer: `7`
- wrong_numeric_final: `6`

## Flags
- correct_number_appears_in_trace: `25`
- last_subgoal_not_answer_like: `20`
- non_numeric_final_answer: `7`
- copied_intermediate_as_final: `5`

## Example Failures
### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "1", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Determine the number of clips sold in May.\n2. Calculate the total number of clips sold in both months.\n\nLet's start with Step 1:\n\nStep 1: F..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "2", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The question has been solved by using the Plan. Let's solve the problem step-by-step:\n\n1. Find out how many clips Natalia sold in May:\n   - In April, Natalia sold 48 clips.\n   - In May, she sold ha..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "4", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["45", "10"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The question is already provided at the top and is not part of the solution. This is a multiple-choice problem, so choose your best answer from (A) $18; (B) $36; (C) $72; (D) $96; without writing i..."}`

### non_numeric_final_answer
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "Since there exists a circle passing through points \\(B\\), \\(C\\), and \\(M", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "36"], "flags": ["non_numeric_final_answer", "correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Final answer: 72 To solve this problem, we need to follow these steps:\n\nStep 1: Determine the number of clips Natalia sold in May.\n\nNatalia sold half as many clips in May as she did in April. She s..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "Final answer: <number> - No specific numbers were mentioned in the context of the question asked, so no numerical val...", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["12", "60"], "flags": ["non_numeric_final_answer", "correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem:\n\nStep 2: Multiply the number of hours by the hourly rate.\n\n- Number of minutes = 50\n- Convert minutes to hours: \\( \\frac{50}{60} = \\frac{5}{6} \\) hours\n- Hourly rate = $12 pe..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "This is nonsensical because humans do not wear shoes on their heads. Shoes are designed to protect the feet and legs ...", "last_subgoal": "number", "step_answers": ["30", "5", "0", "6.28", "5", "5", "5", "5"], "flags": ["non_numeric_final_answer", "correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, let's follow the steps outlined:\n\n### Step 1: Calculate how much money Betty currently has.\nBetty initially had half of what the wallet costs. Since the wallet costs $100, we..."}`

### wrong_numeric_final
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "5", "last_subgoal": "Step 4: Sum up all the flowers to get the total count.", "step_answers": ["18", "59", "27", "18"], "flags": [], "last_reasoning": "Let's solve the problem step by step:\n\nStep 1: Calculate the number of purple flowers.\nWe know that there are 10 yellow flowers and they represent 80% more of the purple flowers than the yellow one..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "27", "last_subgoal": "Step 2: Subtract the remaining budget from the total budget to find out how much she spent on the shoes.", "step_answers": ["184", "59"], "flags": [], "last_reasoning": "To determine how much Alexis paid for the shoes, we need to follow these steps:\n\nStep 1: Calculate the total amount Alexis spent on all items except the shoes.\n\n- Button-up shirt: $30\n- Suit pants:..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "169", "last_subgoal": "Step 3: Add the remaining money ($16) to the cost of the shoes to determine their price.", "step_answers": ["47", "31", "153"], "flags": [], "last_reasoning": "Remember to show all your work.\nFinal answer: $29\n\nStep 1: Step 1: Calculate the total amount Alexis spent on the items she remembers.\nAnswer: The total amount Alexis spent on the items she remembe..."}`
