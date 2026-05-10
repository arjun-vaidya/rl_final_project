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
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "2", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. Determine how many clips Natalia sold in May.\n2. Calculate the total number of clips sold in both months by adding the sales from April and May.\n\n**Step 1: Fin..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "12", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. **Convert the time from minutes to hours**:\n   - We know that there are 60 minutes in one hour.\n   - Weng babysat for 50 minutes yesterday.\n   - To convert min..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents deci...", "ground_truth": "5", "final_answer": null, "invalid_reason": "plan_parse_failed", "router_raw_text": "{\"plan\": [\"Step 1: Calculate how much money Betty already has.\", \"Step 2: Add the amount given by her parents.\", \"Step 3: Add the amount given by her grandparents.\", \"Step 4: Subtract the total amo..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as...", "ground_truth": "42", "final_answer": "36", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To determine how many pages Julie needs to read tomorrow, we can follow these steps:\n\n1. **Calculate the number of pages Julie read today:**\n   - Yesterday, Julie read 12 pages.\n   - Today, she read twice as many page..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "4", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To solve this problem, we need to follow the given plan and perform the necessary calculations step-by-step.\n\n### Step 1: Calculate the number of letters written per week.\nJames writes a 3-page letter to 2 different f..."}`

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
- `{"question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every...", "ground_truth": "18", "final_answer": "3", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To solve this problem, we need to follow the steps provided:\n\n### Step-by-Step Solution\n\n**Step 1:** Calculate the total number of eggs laid by Janet's ducks per day.\nJanet's ducks lay 16 eggs per day. So, the total n..."}`
- `{"question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "ground_truth": "3", "final_answer": "2", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. **Calculate the amount of white fiber needed**:\n   - The question states that a robe takes 2 bolts of blue fiber.\n   - It also mentions that half as much white..."}`
- `{"question": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increas...", "ground_truth": "70000", "final_answer": "80000", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To solve this problem, we need to follow the steps provided:\n\n### Step-by-Step Solution\n\n**Step 1:** Calculate the total cost of buying the house.\nThe house was bought for $80,000, and there were no other costs mentio..."}`
- `{"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetab...", "ground_truth": "20", "final_answer": "40", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To solve this problem, we need to follow these steps:\n\n### Step 1: Calculate the total amount of feed given to all chickens for the entire day.\n- Each chicken gets 3 cups of feed per day.\n- There are 20 chickens.\n\nTot..."}`
- `{"question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 6...", "ground_truth": "64", "final_answer": "4", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To solve this problem, we will follow the plan outlined:\n\n### Plan:\n1. **Calculate the cost of one glass.**\n   - The first glass costs $5.\n   - Every second glass costs 60% of the original price.\n\n2. **Determine how m..."}`

## Stochastic Train Group Diagnostics
- questions: `6`
- questions_with_valid_rollouts: `6`
- valid_rollout_rate: `5.1667`
- avg_router_reward: `0.7119`
- avg_step_reward: `0.5858`
- avg_outcome_reward: `0.2581`
- exact_accuracy: `0.2581`
- strict_numeric_accuracy: `0.2581`
- relaxed_numeric_accuracy: `0.2581`
- mean_group_router_std: `0.0162`
- mean_group_step_std: `0.1031`
- mean_group_outcome_std: `0.2240`
- zero_router_adv_group_frac: `0.5000`
- zero_step_adv_group_frac: `0.1667`
- zero_outcome_adv_group_frac: `0.5000`
- router_reward_relaxed_corr: `-0.3226`
- step_reward_relaxed_corr: `0.4481`
- outcome_reward_relaxed_corr: `1.0000`
- invalid_counts: `{"plan_parse_failed": 5}`

### Failed Rollout Audits
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "24", "router_reward": 0.7, "step_reward": 0.65, "outcome_reward": 0.0, "last_reasoning": "To solve this problem, let's follow the given plan:\n\nStep 1: First, we need to determine how many clips Natalia sold in May. According to the information provided, she sold half as many clips in May as she did in Apri..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "24", "router_reward": 0.7, "step_reward": 0.6, "outcome_reward": 0.0, "last_reasoning": "To solve this problem, we need to follow these steps:\n\n### Plan:\n1. **Find out how many clips Natalia sold in May.**\n   - We know that Natalia sold 48 clips in April.\n   - She sold half as many clips in May compared t..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "router_reward": 0.7, "step_reward": 0.25, "outcome_reward": 0.0, "last_reasoning": "To solve this problem, we need to follow these steps:\n\n1. **Convert the time from minutes to hours:** We know that there are 60 minutes in one hour. Since Weng did 50 minutes of babysitting, we can convert this time i..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "router_reward": 0.7, "step_reward": 0.3, "outcome_reward": 0.0, "last_reasoning": "The next solver will need this explicit final answer as input. You may use either a decimal or fraction for your calculations and answers. To ensure you are understood, please include units (e.g., dollars) in your fin..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "6", "router_reward": 0.7, "step_reward": 0.25, "outcome_reward": 0.0, "last_reasoning": "Let's solve it:\n\nTo find out how much Weng earned yesterday for babysitting, we need to follow these steps:\n\nStep 1: Convert the time from minutes to hours.\nWeng worked for 50 minutes. Since there are 60 minutes in an..."}`
