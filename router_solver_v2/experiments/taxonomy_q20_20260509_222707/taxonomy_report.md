# Failure Taxonomy Report

## Summary
- trace_path: `/home/pvd2112/rl_final_project/router_solver_v2/experiments/taxonomy_q20_20260509_222707/rollout_traces.jsonl`
- records: `20`
- total_rollouts: `120`
- valid_rollouts: `104`
- exact_accuracy: `0.1083`
- relaxed_numeric_accuracy: `0.1083`
- analyzed_failures: `107`
- invalid_counts: `{"plan_parse_failed": 16}`

## Primary Categories
- wrong_numeric_final: `45`
- copied_intermediate_as_final: `19`
- plan_parse_failed: `16`
- correct_number_in_trace_wrong_final: `15`
- plan_endpoint_mismatch: `12`

## Flags
- last_subgoal_not_answer_like: `43`
- copied_intermediate_as_final: `23`
- invalid_rollout: `16`
- non_numeric_final_answer: `16`
- correct_number_appears_in_trace: `15`

## Example Failures
### copied_intermediate_as_final
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "12", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["12", "12"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Convert the time from minutes to hours: Since there are 60 minutes in an hour, we can convert 50 minutes into hours by dividing by 60.\n   \\..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "48", "last_subgoal": "Step 3: Calculate the number of pages Julie plans to read tomorrow.", "step_answers": ["48", "84", "48"], "flags": ["copied_intermediate_as_final", "last_subgoal_not_answer_like"], "last_reasoning": "In your show-work section for Step 3, solve for the number of pages that Julie should read tomorrow by subtracting the number of pages she's read from the number of pages that remain in the book.\n\n..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "2", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["4", "2", "2"], "flags": ["copied_intermediate_as_final"], "last_reasoning": "To determine how many pages James writes in a year, we need to follow these steps:\n\n### Step 1: Calculate the number of letters written per week.\nJames writes 4 letters (2 pages each) every time he..."}`

### correct_number_in_trace_wrong_final
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether...", "ground_truth": "72", "final_answer": "24", "last_subgoal": "Step 2: Add the number of clips sold in April and May together.", "step_answers": ["72", "24"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, let's follow the plan:\n\n### Step 1:\nWe are given that Natalia sold 48 clips in April. According to the question, she sold half as many clips in May as she did in April. There..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "3", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["0.833", "3"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Let's solve the problem step by step as outlined:\n\nStep 1: Convert the time from minutes to hours.\n- We know that there are 60 minutes in one hour.\n- Weng worked for 50 minutes yesterday.\n\nTo conve..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she w...", "ground_truth": "42", "final_answer": "4", "last_subgoal": "Step 3: Calculate the number of pages Julie needs to read tomorrow.", "step_answers": ["36", "84", "4"], "flags": ["correct_number_appears_in_trace", "last_subgoal_not_answer_like"], "last_reasoning": "Remember that you are providing calculations and explanations, not describing a story or writing about someone's life.\n\nFinal answer: 42\n\nExplanation: \nTo find out how many pages Julie should read ..."}`

### plan_endpoint_mismatch
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "12", "last_subgoal": "Calculate the earnings by multiplying the hourly rate by the number of hours worked.", "step_answers": ["60", "12"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve the problem step-by-step:\n\n1. Convert the time from minutes to hours:\n   - Weng's babysitting time is given as 50 minutes.\n   - Since there are 60 minutes in one hour, we convert the minut..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "60", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["6", "60"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To solve this problem, we need to follow these steps:\n\n### Step-by-Step Solution\n\nGiven Information:\n- Weng earns $12 per hour.\n- She did 50 minutes of babysitting yesterday.\n\nConversion of Time fr..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "9.996", "last_subgoal": "Multiply the number of hours by the hourly rate.", "step_answers": ["9.6", "9.996"], "flags": ["last_subgoal_not_answer_like"], "last_reasoning": "To convert the time from minutes to hours:\n\n\\[ \\text{Time in hours} = \\frac{\\text{Time in minutes}}{60} \\]\n\nGiven that Weng babysat for 50 minutes:\n\n\\[ \\text{Time in hours} = \\frac{50}{60} = \\frac{..."}`

### plan_parse_failed
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "", "last_subgoal": "", "step_answers": [], "flags": ["invalid_rollout", "non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": ""}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "", "last_subgoal": "", "step_answers": [], "flags": ["invalid_rollout", "non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": ""}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "", "last_subgoal": "", "step_answers": [], "flags": ["invalid_rollout", "non_numeric_final_answer", "last_subgoal_not_answer_like"], "last_reasoning": ""}`

### wrong_numeric_final
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 ...", "ground_truth": "5", "final_answer": "65", "last_subgoal": "Step 4: Subtract the total from the cost of the wallet.", "step_answers": ["95", "30", "30", "65"], "flags": [], "last_reasoning": "Let's go through each step systematically:\n\n### Given Data:\n- Cost of the wallet: \\$100\n- Amount Betty already has: \\( \\frac{1}{2} \\times 100 = \\$50 \\)\n- Amount given by Betty's parents: \\$15\n- Amo..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "52", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["1", "1456", "52"], "flags": [], "last_reasoning": "To solve the problem:\n\nGiven:\n- James writes a 3-page letter to 2 different friends twice a week.\n- He writes this twice a week.\n\nCalculation for Steps 1 and 2 (already completed):\n\n1. Letters writ..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "2", "last_subgoal": "Step 3: Multiply the result from Step 2 by the number of times James writes each week.", "step_answers": ["12", "52", "2"], "flags": [], "last_reasoning": "Let's go through the steps carefully:\n\n### Step 1: Calculate the number of letters written per week\n\nJames writes to 2 different friends and each friend gets a letter on a separate page for every p..."}`
