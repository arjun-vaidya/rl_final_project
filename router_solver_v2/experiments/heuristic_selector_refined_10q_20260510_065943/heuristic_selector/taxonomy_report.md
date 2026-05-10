# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/heuristic_selector_refined_10q_20260510_065943/heuristic_selector/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.2167`
- relaxed_numeric_accuracy: `0.2167`
- analyzed_failures: `47`
- invalid_counts: `{}`
- final_answer_source_counts: `{"heuristic_selector": 60}`

## Primary Categories
- wrong_numeric_final: `20`
- correct_number_in_trace_wrong_final: `18`
- copied_intermediate_as_final: `7`
- plan_endpoint_mismatch: `2`

## Flags
- correct_number_appears_in_trace: `18`
- last_subgoal_not_answer_like: `15`
- copied_intermediate_as_final: `12`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "36", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["48", "36", "120", "36"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "Let's solve the problem step by step:\n\nStep 1: Calculate the number of pages Julie read today.\nJulie read 12 pages yesterday and twice that amount today, which is \\(2 \\times 12 = 24\\) pages.\n\nStep ..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "84", "last_subgoal": "Step 4: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["36", "36", "84", "84"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To find out the number of pages Julie needs to read tomorrow, we need to follow these steps:\n\nStep 1: We know that today Julie read twice as many pages as she did yesterday. Since she read 12 pages..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "312", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of pages per letter.", "step_answers": ["312", "1870", "312"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve the problem, we need to follow the given plan and complete each step:\n\nStep 1: Calculate the number of letters written per week.\nJames writes a 3-page letter to 2 different friends twice a..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "1", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "72"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Let's solve it:\n\nTo find out how many clips Natalia sold in May, we need to follow these steps:\n\nStep 1: Calculate the number of clips sold in May.\nNatalia sold half as many clips in May as she did..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "24"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Determine how many clips Natalia sold in May.\n2. Calculate the total number of clips sold in both months by adding the sales from April and..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "The numbers used or referred to in your response should be exact per the question, since it is an open-book test. To solve the problem, let's follow the plan given:\n\nStep 1: Find out how many clips..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "2", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["6", "2"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem:\n\n1. Convert the time from minutes to hours.\n   - Since there are 60 minutes in an hour, we divide 50 minutes by 60 to convert it to hours.\n   \n   \\[\n   \\text{Time in hours} = ..."}`
- `{"question": "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into...", "ground_truth": "16", "final_answer": "3", "last_subgoal": "Step 5: Double the weight by adding enough gummy worms.", "step_answers": ["2", "12", "2", "8", "20", "5"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "Assume the weight gain from one item is negligible except where explicitly stated.\nLet's solve step-by-step:\n\n### Initial conditions:\n- Step 1: Place a box on a scale.\n  - The initial weight is 0 p..."}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "100", "last_subgoal": "Finally, we subtract the total amount from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["30", "45", "30"], "flags": [], "last_reasoning": "Let's start solving it step-by-step:\n\n### Step 1: Calculate how much money Betty currently has\n\n- The original problem states that Betty already has \"half\" of the money she needs for the wallet.\n- ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "60", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet to find out how much more money Betty needs.", "step_answers": ["15", "30", "45", "60"], "flags": [], "last_reasoning": "The question was solved before it began in my case so I will provide a solution without writing any text.\nTo solve this step-by-step:\n\nGiven data:\n- Cost of the wallet: $100\n- Betty's initial savin..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "480", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["6", "1440", "4"], "flags": [], "last_reasoning": "To solve this problem systematically:\n\nStep 1: First, let's determine how many letters James writes per week.\n\n- James writes 2 letters (one to each friend) twice a week. So we multiply these two q..."}`
