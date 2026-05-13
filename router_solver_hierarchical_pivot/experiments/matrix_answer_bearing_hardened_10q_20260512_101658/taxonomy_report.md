# Failure Taxonomy Report

## Summary
- trace_path: `experiments/matrix_answer_bearing_hardened_10q_20260512_101658/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.3667`
- relaxed_numeric_accuracy: `0.3667`
- question_majority_exact_accuracy: `0.5000`
- question_majority_relaxed_accuracy: `0.5000`
- question_any_relaxed_accuracy: `0.9000`
- analyzed_failures: `38`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 60}`

## Primary Categories
- correct_number_in_trace_wrong_final: `29`
- wrong_numeric_final: `7`
- copied_intermediate_as_final: `2`

## Flags
- correct_number_appears_in_trace: `29`
- copied_intermediate_as_final: `5`
- last_subgoal_not_answer_like: `1`

## Example Failures
### copied_intermediate_as_final
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "2", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: James writes...", "step_answers": ["312", "2", "6072", "6072", "2"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To solve this problem:\n\n1. First, calculate the total number of letters written in one week:\n   \\[ \\text{Letters per week} = \\text{Pages per letter} \\times \\text{Number of letters per friend} \\time..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "57", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Alexis is ap...", "step_answers": ["125", "31", "57", "31", "31", "57", "57"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "Instead continue to present the final solution or reasoning if appropriate. Do not include any intro text or references to questions unless asking for clarification.\nStep 1: Calculate the total amo..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "2", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Weng earns $...", "step_answers": ["6", "10", "1", "10", "10", "10", "10", "2"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "Plan Step 8: To find out how much Weng earned yesterday, we first need to determine how many hours she worked. Since she earns $12 per hour and worked for 50 minutes, we convert the time to hours b..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "2", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Weng earns $...", "step_answers": ["50", "100", "2"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "We will accept either integer or decimal values, and round as appropriate (so 4.7 would be acceptable but 4.68 should be rounded up). Please show all work and explain each step of your solution. \n\n..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "30", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Betty is sav...", "step_answers": ["1", "7.5", "100", "5", "5", "5", "30"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "The problem is solved when you reach the final line in the required format.\n\nTo solve the problem, we will follow the steps outlined:\n\nStep 1: Determine how much money Betty currently has.\n- Betty ..."}`

### wrong_numeric_final
- `{"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of thos...", "ground_truth": "35", "final_answer": "90", "last_subgoal": "Step 4: Sum up all the flowers.", "step_answers": ["0", "100", "22", "90"], "flags": [], "last_reasoning": "The given data was insufficient for any conclusion beyond Step 2. To solve this problem correctly, we need more information about the total number of flowers in Mark's garden. We know that he plant..."}`
- `{"question": "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into...", "ground_truth": "16", "final_answer": "7", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Ken created ...", "step_answers": ["19", "130", "54", "8", "130", "27", "7"], "flags": [], "last_reasoning": "The task is strictly following instructions for math-solving format. To solve the problem:\n\nStep 1: Weigh the empty box. Answer: 19\nStep 2: Add jelly beans to make it 2 pounds. Answer: 130\nStep 3: ..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "168", "last_subgoal": "Determine the cost of the shoes by subtracting the known costs from the total cost.", "step_answers": ["184", "184", "125", "168"], "flags": [], "last_reasoning": "The given information does not contain any further instructions or prompts, so it seems that no additional steps need to be completed beyond what was provided in Step 4. However, based on the detai..."}`
