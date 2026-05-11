# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/reward_ablation_10q_remotejudgefix_20260510_093341/outcome_heavy/eval/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.3167`
- relaxed_numeric_accuracy: `0.3167`
- analyzed_failures: `41`
- invalid_counts: `{}`
- final_answer_source_counts: `{"synthesis": 60}`

## Primary Categories
- correct_number_in_trace_wrong_final: `15`
- wrong_numeric_final: `13`
- copied_intermediate_as_final: `8`
- non_numeric_final_answer: `3`
- plan_endpoint_mismatch: `2`

## Flags
- correct_number_appears_in_trace: `16`
- copied_intermediate_as_final: `13`
- last_subgoal_not_answer_like: `10`
- non_numeric_final_answer: `3`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "36", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["36", "24", "84", "36"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To determine the number of pages Julie needs to read tomorrow, we will follow these steps:\n\nStep 4: Calculate the number of pages Julie needs to read tomorrow.\n\n1. Total Pages Read Today: \n   - Yes..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "36", "last_subgoal": "Step 4: Calculate how many pages Julie needs to read tomorrow.", "step_answers": ["36", "36", "48", "36"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve for the number of pages Julie needs to read tomorrow, let's follow the plan:\n\n1. Calculate the number of pages Julie read today:\n   - Julie read 12 pages yesterday.\n   - Today, she read tw..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "588", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["7", "588", "12"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how many pages James writes in a year, we need to follow these steps:\n\n1. Calculate the number of letters written per week:\n   - James writes a 3-page letter to each of two friends twi..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "72"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "Final answer: 72"}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "50", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["6", "10"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The final answer is 72. To convert the time from minutes to hours, divide the number of minutes by 60 (since there are 60 minutes in an hour).\n\n$$ \\frac{50}{60} = \\frac{5}{6} $$\n\nThis fraction repr..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Step 6: The result will be the amount of money Betty still needs.", "step_answers": ["2", "30", "30", "47", "53", "30"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Determine how much money Betty currently has: \n   - Betty initially has half the amount needed for the wallet.\n   - Cost of the wallet = $1..."}`

### non_numeric_final_answer
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "If you cannot determine the final answer from the given information, return \"Insufficient data\". Final answer: Insuff...", "last_subgoal": "Step 4: Sum up all the flowers to get the total count.", "step_answers": ["18", "28", "43", "18"], "flags": ["non_numeric_final_answer"], "last_reasoning": "To solve this problem, let's follow the given plan step by step:\n\nStep 1: Find the number of purple flowers.\n- We know that Mark has ten yellow flowers.\n- The number of purple flowers is 80% more t..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "No unique solution. The reasoning provided shows two distinct paths leading to different totals, indicating ambiguity...", "last_subgoal": "Step 4: Add up all the flowers to get the total number.", "step_answers": ["18", "25", "7", "3"], "flags": ["non_numeric_final_answer"], "last_reasoning": "Let's solve the problem step by step:\n\n1. Calculate the number of purple flowers:\n   - Given that there are 10 yellow flowers.\n   - Purple flowers are 80% more than yellow flowers.\n   - First, calc..."}`
- `{"question": "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into...", "ground_truth": "16", "final_answer": "If there is no correct final answer, return \"No final answer\". No final answer.", "last_subgoal": "Step 5: Double the weight by adding enough gummy worms.", "step_answers": ["5", "14", "16", "4", "12", "16"], "flags": ["non_numeric_final_answer", "correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "The question is from chapter 7.3. To find the total number of candies in the box:\n\nFirst, we start with the initial weight before any candies were added:\n\\[ \\text{Initial weight} = 2 \\text{ pounds}..."}`

### plan_endpoint_mismatch
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "12", "last_subgoal": "Step 3: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["84", "18", "2"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To determine how many pages Julie needs to read tomorrow, we will follow these steps:\n\n### Step 1: Calculate the total number of pages Julie has already read.\nJulie read 12 pages yesterday and twic..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "472", "last_subgoal": "Step 3: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["84", "84", "4"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The problem statement will be given before the solution starts.\n\nProblem Statement:\nJulie's new goal is to reduce her daily page count from 15 pages each day for the first three days to 9 pages per..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["50", "3", "50", "30"], "flags": [], "last_reasoning": "To solve this problem systematically, let's follow each step carefully:\n\n### Step 1: Calculate how much money Betty currently has.\nBetty needs $100 and she already has half of that amount. Therefor..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "312", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["2", "42", "312"], "flags": [], "last_reasoning": "To determine how many pages James writes in a year, we will follow the steps outlined:\n\n1. Calculate the number of letters written per week.\n   - James writes 3-page letters to 2 different friends ..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "1872", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["12", "312", "1872"], "flags": [], "last_reasoning": "The problem is solved using a formula or direct calculation and does not require an explanation beyond answering the question directly. Do not include units since they are optional and can vary dep..."}`
