# Training Quality Diagnostics

## Config
- diagnostic_questions: `6`
- diagnostic_rollouts_per_q: `6`
- use_judge: `True`
- router_temperature: `0.2`
- solver_temperature: `1.0`

## Deterministic Train Slice
- total: `6`
- valid: `5`
- plan_validity_rate: `0.833`
- exact_accuracy: `0.000`
- strict_numeric_accuracy: `0.000`
- relaxed_numeric_accuracy: `0.000`
- avg_plan_len: `3.00`
- invalid_counts: `{"plan_parse_failed": 1}`

### Audit Samples
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "48", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "72\n\nWork:\n\nTo find out how many clips Natalia sold in May, we need to take half of the number of clips she sold in April. Since she sold 48 clips in April, we calculate:\n\n\\[\n\\text{Clips sold in May} = \\frac{\\text{Clip..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "$7.20\n\nWork:\n\nTo find out how much Weng earned for 50 minutes of babysitting at a rate of $12 per hour, we need to follow these steps:\n\n1. **Convert the time from minutes to hours**:\n   - There are 60 minutes in one h..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents deci...", "ground_truth": "5", "final_answer": null, "invalid_reason": "plan_parse_failed", "router_raw_text": "{\"plan\": [\"Step 1: Calculate how much money Betty already has.\", \"Step 2: Add the amount given by her parents.\", \"Step 3: Add the amount given by her grandparents.\", \"Step 4: Subtract the total amo..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as...", "ground_truth": "42", "final_answer": "36", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "18\n\nTo solve the problem, let's follow the steps:\n\n1. **Calculate the number of pages Julie read today:**\n   - Yesterday, Julie read 12 pages.\n   - Today, she read twice as many pages as yesterday.\n   \\[\n   \\text{Page..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "4", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "48\n\nWork:\n\nTo solve this problem, we need to follow the plan and perform the calculations step-by-step.\n\n### Step 1: Calculate the number of letters written per week.\nJames writes a 3-page letter to 2 different friend..."}`

## Deterministic Test Slice
- total: `6`
- valid: `6`
- plan_validity_rate: `1.000`
- exact_accuracy: `0.167`
- strict_numeric_accuracy: `0.167`
- relaxed_numeric_accuracy: `0.167`
- avg_plan_len: `3.83`
- invalid_counts: `{}`

### Audit Samples
- `{"question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every...", "ground_truth": "18", "final_answer": "4", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "$8\n\nTo solve this problem, let's follow the steps provided:\n\n### Step-by-Step Solution:\n\n#### Step 1: Calculate the total number of eggs laid by Janet's ducks per day.\nJanet's ducks lay 16 eggs per day. Therefore,\n\\[ ..."}`
- `{"question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "ground_truth": "3", "final_answer": "1", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "4\n\nExplanation:\n\nTo solve this problem, we need to follow these steps:\n\n1. **Calculate the amount of white fiber needed:** The question states that a robe takes 2 bolts of blue fiber and half that much white fiber. Th..."}`
- `{"question": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increas...", "ground_truth": "70000", "final_answer": "120000", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "Profit = New Value - Original Purchase Price\n\nLet's solve it:\n\nNew Value = Original Purchase Price + Increase in Value Due to Repairs\nNew Value = $80,000 + ($80,000 * 1.5)\nNew Value = $80,000 + $120,000\nNew Value = $2..."}`
- `{"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetab...", "ground_truth": "20", "final_answer": "2", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "9\n\nTo solve this problem, we need to follow these steps:\n\n### Step 1: Calculate the total amount of feed given to all chickens for the entire day.\n- Each chicken gets 3 cups of feed per day.\n- There are 20 chickens.\n\n..."}`
- `{"question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 6...", "ground_truth": "64", "final_answer": "8", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "8\n\nTo solve this problem, let's follow the plan step-by-step:\n\n### Step 1: Calculate the cost of one glass.\nThe original price of each glass is $5.\n\n### Step 2: Determine how many glasses are at full price and how man..."}`

## Stochastic Train Group Diagnostics
- questions: `6`
- questions_with_valid_rollouts: `6`
- valid_rollout_rate: `5.5000`
- avg_router_reward: `0.7061`
- avg_step_reward: `0.6581`
- avg_outcome_reward: `0.1515`
- exact_accuracy: `0.1515`
- strict_numeric_accuracy: `0.1515`
- relaxed_numeric_accuracy: `0.1515`
- mean_group_router_std: `0.0166`
- mean_group_step_std: `0.0895`
- mean_group_outcome_std: `0.1407`
- zero_router_adv_group_frac: `0.3333`
- zero_step_adv_group_frac: `0.0000`
- zero_outcome_adv_group_frac: `0.6667`
- router_reward_relaxed_corr: `-0.1049`
- step_reward_relaxed_corr: `0.2704`
- outcome_reward_relaxed_corr: `1.0000`
- invalid_counts: `{"plan_parse_failed": 3}`

### Failed Rollout Audits
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "24", "router_reward": 0.7, "step_reward": 0.75, "outcome_reward": 0.0, "last_reasoning": "Total = Clips sold in April + Clips sold in May\n\nTotal = 48 + 72 = 120\nTo solve this problem, we need to follow the given plan step by step:\n\n### Step 1: Find out how many clips Natalia sold in May.\nNatalia sold clips..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "2", "router_reward": 0.7, "step_reward": 0.65, "outcome_reward": 0.0, "last_reasoning": "To find the total number of clips Natalia sold in both months, we'll follow these steps:\n\n1. **Find out how many clips Natalia sold in May**:\n   - We know that Natalia sold half as many clips in May as she did in Apri..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "router_reward": 0.62, "step_reward": 0.5, "outcome_reward": 0.0, "last_reasoning": "6\nTo solve the problem of determining how much Weng earned from her 50-minute babysitting session yesterday, follow these steps:\n\n1. **Convert the time from minutes to hours**:\n   - There are 60 minutes in one hour.\n ..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "12", "router_reward": 0.62, "step_reward": 0.65, "outcome_reward": 0.0, "last_reasoning": "To calculate how much Weng earned yesterday, we first need to convert the 50 minutes into hours and then multiply that by her hourly rate of $12 per hour.\n\n1. **Convert the time from minutes to hours:**\n   - There are..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "12", "router_reward": 0.66, "step_reward": 0.75, "outcome_reward": 0.0, "last_reasoning": "$6\n\nExplanation:\n\nWe know that Weng earns $12 per hour for babysitting. We need to calculate how much she earned for 50 minutes of babysitting. First, we convert 50 minutes into hours since her pay is given in dollars..."}`
