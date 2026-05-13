# Failure Taxonomy Report

## Summary
- trace_path: `experiments/highest_likelihood_single_variant_20260512/20260512_133500/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.1667`
- relaxed_numeric_accuracy: `0.1667`
- analyzed_failures: `50`
- invalid_counts: `{}`
- final_answer_source_counts: `{"heuristic_selector": 60}`

## Primary Categories
- correct_number_in_trace_wrong_final: `18`
- wrong_numeric_final: `15`
- plan_endpoint_mismatch: `11`
- copied_intermediate_as_final: `6`

## Flags
- last_subgoal_not_answer_like: `27`
- correct_number_appears_in_trace: `18`
- copied_intermediate_as_final: `8`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "50", "last_subgoal": "Step 4: Subtract the total amount from the cost of the wallet.", "step_answers": ["2", "30", "50", "50"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how much more money Betty needs to buy the wallet, we will follow the steps provided:\n\n1. Calculate how much money Betty currently has.\n   - The problem states that Betty has only half..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "84", "last_subgoal": "The Gourmet Kitchen", "step_answers": ["36", "84", "36", "84"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem, we need to follow the given plan step by step:\n\n### Step-by-Step Solution:\n\nStep 1: Calculate the total number of pages Julie has already read.\nJulie read 12 pages yesterday a..."}`
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "18", "last_subgoal": "variables", "step_answers": ["18", "18", "18", "21", "10", "21", "0.2", "18"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, let's follow the plan:\n\nStep 1: Calculate the number of purple flowers.\nMark has 10 yellow flowers. There are 80% more purple flowers than yellow ones. We can calculate the n..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "24"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Remember that Natalia sold clips twice in the same month! Let's start solving it:\n\nTo find out how many clips Natalia sold in May, we need to calculate half of the number of clips sold in April bec..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "0", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "24"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Show all your work above or below this sentence. Let's start solving it:\n\nTo find out how many clips Natalia sold in May, we need to follow these steps:\n\nStep 1: We know that Natalia sold 48 clips ..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["correct_number_appears_in_trace", "copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "The total number of clips sold in April and May is:\n\nFinal answer: 72 clips. To solve this problem step by step, we'll follow the plan provided and use the information given.\n\nGiven:\n- Natalia sold..."}`

### plan_endpoint_mismatch
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "3", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["24", "24"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "Let's solve it:\n\nStep 1: To find out how many clips Natalia sold in May, we need to divide the number of clips sold in April by two because she sold half as many clips in May as she did in April. \n..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "0.833", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["6", "0.833"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "The number should be used as is in your response to me. Do you understand? Let's solve this problem together:\n\n### Step-by-Step Solution\n\n#### Step 1: Convert the time from minutes to hours\n- You h..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "last_subgoal": "Both inputs must be lists", "step_answers": ["60", "50", "6"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this step, we need to convert the 50 minutes into hours first before multiplying it by her hourly rate.\n\nStep 1: Convert the time from minutes to hours.\n\n- Given: 50 minutes\n- There are 60..."}`

### wrong_numeric_final
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "2184", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["4", "6", "2184"], "flags": [], "last_reasoning": "To solve the problem:\n\n1. Calculate the number of letters written per week: \n   - James writes 3 pages to 2 different friends twice a week.\n   - Number of letters per day = 3 (pages) * 2 (friends) ..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "8160", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["2", "408", "8160"], "flags": [], "last_reasoning": "Hint: Think about how many times James writes each week and how many pages are written for each friend. Then, calculate how many pages he writes each week and finally multiply that by the number of..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "312", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["3", "768", "312"], "flags": [], "last_reasoning": "You have calculated correctly up to now. Now calculate how many pages James writes per week and then multiply by the number of weeks in a year.\n\nShow your work:\n\nTo determine how many pages James w..."}`
