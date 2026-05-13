# Failure Taxonomy Report

## Summary
- trace_path: `experiments/matrix_answer_bearing_base_10q_20260512_094203/rollout_traces.jsonl`
- records: `10`
- total_rollouts: `60`
- valid_rollouts: `60`
- exact_accuracy: `0.4833`
- relaxed_numeric_accuracy: `0.4833`
- question_majority_exact_accuracy: `0.7000`
- question_majority_relaxed_accuracy: `0.7000`
- question_any_relaxed_accuracy: `0.9000`
- analyzed_failures: `31`
- invalid_counts: `{}`
- final_answer_source_counts: `{"last_step": 60}`

## Primary Categories
- correct_number_in_trace_wrong_final: `26`
- wrong_numeric_final: `4`
- copied_intermediate_as_final: `1`

## Flags
- correct_number_appears_in_trace: `26`
- copied_intermediate_as_final: `4`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1...", "ground_truth": "990", "final_answer": "1170", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Tina makes $...", "step_answers": ["5", "198", "2250", "1350", "11", "198", "1170", "1170"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "The above method is used because it provides a logical sequence of operations to solve the problem presented in the initial question. To calculate Tina's earnings over 5 days working 10 hours each ..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "11", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Weng earns $...", "step_answers": ["10", "10", "10", "10", "50", "10", "10", "11"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "To convert 50 minutes to hours, we use the conversion factor that there are 60 minutes in an hour. Thus, 50 minutes divided by 60 equals 50/60 hours, which simplifies to 5/6 or approximately 0.8333..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "9.996", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Weng earns $...", "step_answers": ["2", "8", "9.96", "10", "50", "6", "10", "9.996"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "To convert 50 minutes into hours we divide by 60 because there are 60 minutes per hour. This gives us .833 hours. Multiplying this times 12 dollars an hour equals 10 dollars. Final answer: 10 Step ..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "8", "last_subgoal": "Step 2: Determine how much money Betty's parents gave her.", "step_answers": ["50", "5", "50", "95", "5", "5", "35", "8"], "flags": ["correct_number_appears_in_trace"], "last_reasoning": "To solve the problem, let's follow the steps provided:\n\nStep 1: Betty originally had half of what the wallet costs.\n- Cost of the wallet = $100\n- Half of $100 = \\( \\frac{1}{2} \\times 100 = \\$50 \\)\n..."}`

### wrong_numeric_final
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "184", "last_subgoal": "Answer the original question directly using the previous results. State the final quantity requested in: Alexis is ap...", "step_answers": ["127", "-5", "31", "57", "2", "184"], "flags": [], "last_reasoning": "To find out how much Alexis paid for the shoes, we first need to calculate the total amount she spent on all the other items:\n\n- Button-up shirt: $30\n- Suit pants: $46\n- Suit coat: $38\n- Socks: $11..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "16", "last_subgoal": "Calculate the total amount Alexis spent on the items she remembers.", "step_answers": ["91", "73", "11", "147", "73", "117", "15", "16"], "flags": [], "last_reasoning": "The first line of work should read \"Step 1:\" when including numbers in your response. For example:\n\nStep 1: $30 + $46 + $38 + $11 + $18 = $143\n\nStep 2: $200 - $143 = $57\n\nStep 3: $57 + $16 = $73\n\nS..."}`
- `{"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a...", "ground_truth": "41", "final_answer": "32", "last_subgoal": "Subtract the remaining budget from the total amount spent: After calculating the total spent, we subtract the remaini...", "step_answers": ["127", "32"], "flags": [], "last_reasoning": "Do not write \"The final answer is\" or any similar phrase at the beginning of your response. Your response should begin immediately following the final answer line. To solve the problem, let's follo..."}`
