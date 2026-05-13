# Failure Taxonomy Report

## Summary
- trace_path: `/home/machina/pvd2112/rl_final_project/router_solver_v2/experiments/robust_matrix_20260512_0352/robust_guarded_selector/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `40`
- valid_rollouts: `40`
- exact_accuracy: `0.1500`
- relaxed_numeric_accuracy: `0.1500`
- analyzed_failures: `34`
- invalid_counts: `{}`
- final_answer_source_counts: `{"heuristic_selector_refined": 27, "synthesis_self_consistency": 13}`

## Primary Categories
- correct_number_in_trace_wrong_final: `23`
- wrong_numeric_final: `7`
- copied_intermediate_as_final: `2`
- plan_endpoint_mismatch: `2`

## Flags
- correct_number_appears_in_trace: `23`
- last_subgoal_not_answer_like: `16`
- copied_intermediate_as_final: `4`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "2", "last_subgoal": "Step 3: Add up the cost of the shoes to get the final answer.", "step_answers": ["2", "59", "184"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "We will check your work using these numbers: -59, 59, 178, 228\n\nStep 3: Step 3: Add up the cost of the shoes to get the final answer.\n\nLet's break it down:\n\nTotal spent excluding shoes = $30 (shirt..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "31", "last_subgoal": "Multiply the price per cupcake by the number of cupcakes sold to get the total revenue.", "step_answers": ["31", "31"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much Alexis paid for the shoes, we need to follow these steps:\n\n1. Calculate the total amount Alexis initially had before spending on the other items.\n2. Subtract the cost of all t..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "0", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["1", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The first sentence is your plan, followed by your work on the problem, and finally the final answer. Use correct mathematical notation and show all steps. Make sure to label your answers clearly.\nL..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "3", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "12"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "In previous months, Natalia had sold clips at different rates each month:\n\n- January: 50 clips\n- February: 36 clips\n\nSo now we have three more calculations needed before summing them up.\n\nWork to b..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "123", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["6", "10"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To convert the time from minutes to hours, we use the conversion factor that there are 60 minutes in an hour. So, we divide the number of minutes by 60:\n\n\\[\n\\text{Hours} = \\frac{\\text{Minutes}}{60}..."}`

### plan_endpoint_mismatch
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "32", "last_subgoal": "Subtract the apples given away from the initial amount:", "step_answers": ["28", "90"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The number should be whole. Step 2: Subtract the apples given away from the initial amount:\n\nJohn initially had 360 apples.\nHe gave away 28 apples.\n\nTo find out how many apples he has left, we subt..."}`
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "-1", "last_subgoal": "Overtime pay (hours over 8)", "step_answers": ["371", "8"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The question asks you to solve the problem step-by-step.\n\nLet's start from where we left off:\n\n### Step 1: Calculate the regular pay\nTina earns $18.00 an hour and works up to 8 hours at her normal ..."}`

### wrong_numeric_final
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "49", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["7", "5", "2", "49"], "flags": [], "last_reasoning": "Let's solve the problem step by step:\n\n1. Calculate the number of purple flowers:\n   - We know that there are 10 yellow flowers.\n   - There are 80% more purple flowers than yellow flowers.\n   - To ..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "1", "last_subgoal": "Calculate the total number of yellow and purple flowers: Now that we know both the number of yellow and purple flower...", "step_answers": ["18", "28"], "flags": [], "last_reasoning": "To solve this problem step-by-step, let's break it down:\n\n### Step 1: Find the number of purple flowers\n\nWe are given that there are 10 yellow flowers and that there are 80% more purple flowers tha..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "59", "last_subgoal": "Step 2: Subtract the remaining money from the initial budget to find out how much she paid for the shoes.", "step_answers": ["4", "59"], "flags": [], "last_reasoning": "To determine how much Alexis paid for the shoes, we need to follow these steps:\n\n1. Calculate the total amount spent on all items except the shoes:\n   - Button-up shirt: $30\n   - Suit pants: $46\n  ..."}`
