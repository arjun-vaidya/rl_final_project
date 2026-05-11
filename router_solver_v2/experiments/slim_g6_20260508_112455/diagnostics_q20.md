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
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "#### Step 2: Calculate the total number of", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "Natalia sold a total of _________ clips in April and May. Let's break down the problem:\n\n### Given Information\n- Natalia sold 48 clips in April.\n- She sold half as many clips in May compared to April.\n\n### Step-by-Ste..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "\\text{Hours} = \\frac{5}{6} \\text{", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "Step 2: Multiply the number of hours by the hourly rate:\n\n\\[ \\text{Earnings} = \\left( \\frac{5}{6} \\right) \\times \\$12 \\]\n\n\\[ \\text{Earnings} = \\$10 \\] To solve this step-by-step, let's follow the plan and perform each..."}`
- `{"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents deci...", "ground_truth": "5", "final_answer": null, "invalid_reason": "plan_parse_failed", "router_raw_text": "{\"plan\": [\"Step 1: Calculate how much money Betty already has.\", \"Step 2: Add the amount given by her parents.\", \"Step 3: Add the amount given by her grandparents.\", \"Step 4: Subtract the total amo..."}`
- `{"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as...", "ground_truth": "42", "final_answer": "\\[ \\text{Remaining Pages} =", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "Julie needs to read 42 pages tomorrow. \n\nTo find out how many pages Julie needs to read tomorrow, we can break down the problem into smaller parts:\n\n1. **Total Pages in the Book:** The book has 120 pages.\n2. **Pages R..."}`
- `{"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "ground_truth": "624", "final_answer": "### Step 3: Multiply the result from Step", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "________ pages per year To solve this problem, let's break it down into the three steps as outlined:\n\n### Step 1: Calculate the number of letters written per week\n\nJames writes a 3-page letter to 2 different friends t..."}`

## Deterministic Test Slice
- total: `6`
- valid: `6`
- plan_validity_rate: `1.000`
- exact_accuracy: `0.000`
- strict_numeric_accuracy: `0.000`
- relaxed_numeric_accuracy: `0.167`
- avg_plan_len: `3.83`
- invalid_counts: `{}`

### Audit Samples
- `{"question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every...", "ground_truth": "18", "final_answer": "\\]", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "\\[\n\\boxed{0}\n\\]"}`
- `{"question": "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increas...", "ground_truth": "70000", "final_answer": "The problem states that the repairs increased the value of the house by 150%. To calculate the increase in value, we", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "Profit = $200,000 - $80,000 = $120,000\n\nLet's go through each step carefully:\n\n### Step 1: Calculate the total cost of buying the house.\nThe house was initially bought for $80,000. No additional costs are mentioned ot..."}`
- `{"question": "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a w...", "ground_truth": "540", "final_answer": "", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "_______ meters\n\nTo solve this problem, we need to follow these steps:\n\n### Step 1: Calculate the distance for one complete set of three sprints.\n- Each sprint is 60 meters long.\n- There are 3 sprints in one set.\n\nSo, ..."}`
- `{"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetab...", "ground_truth": "20", "final_answer": "- In the", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "The number of cups of feed needed for the final meal is **20**. \n\nExplanation: \nTo solve this problem, let's break it down into clear steps:\n\n### Step-by-Step Solution\n\n#### Step 1: Calculate the Total Amount of Feed ..."}`
- `{"question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 6...", "ground_truth": "64", "final_answer": "Cost of discounted", "exact": false, "strict_numeric": false, "relaxed_numeric": false, "last_reasoning": "To solve this problem, we'll follow the plan outlined:\n\n### Step-by-Step Solution\n\n#### Step 1: Calculate the cost of one glass.\nThe cost of one glass is given as $5.\n\n#### Step 2: Determine how many glasses are at fu..."}`

## Stochastic Train Group Diagnostics
- questions: `6`
- questions_with_valid_rollouts: `6`
- valid_rollout_rate: `5.3333`
- avg_router_reward: `0.7050`
- avg_step_reward: `0.6609`
- avg_outcome_reward: `0.0000`
- exact_accuracy: `0.0000`
- strict_numeric_accuracy: `0.0000`
- relaxed_numeric_accuracy: `0.1250`
- mean_group_router_std: `0.0189`
- mean_group_step_std: `0.0874`
- mean_group_outcome_std: `0.0000`
- zero_router_adv_group_frac: `0.3333`
- zero_step_adv_group_frac: `0.0000`
- zero_outcome_adv_group_frac: `1.0000`
- router_reward_relaxed_corr: `-0.2015`
- step_reward_relaxed_corr: `0.1513`
- outcome_reward_relaxed_corr: `None`
- invalid_counts: `{"plan_parse_failed": 4}`

### Failed Rollout Audits
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "\\text{Total clips sold in April and May}", "router_reward": 0.7, "step_reward": 0.75, "outcome_reward": 0.0, "last_reasoning": "In total, Natalia sold 72 clips in April and May. **Work:** \n1. First, we determine how many clips Natalia sold in May. She sold half as many clips in May compared to April. So, we calculate:\n   \\[\n   \\text{Clips sold..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "\\[ \\text{Total clips sold} = 4", "router_reward": 0.7, "step_reward": 0.8, "outcome_reward": 0.0, "last_reasoning": "\\_\\_\\_\\_ clips\n\nTo solve this problem, we need to follow the plan provided:\n\n### Step-by-Step Solution:\n\n**Step 1:** Find out how many clips Natalia sold in May.\n\nGiven:\n- In April, Natalia sold 48 clips.\n\nIn May, Nat..."}`
- `{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Nat...", "ground_truth": "72", "final_answer": "", "router_reward": 0.7, "step_reward": 0.85, "outcome_reward": 0.0, "last_reasoning": "_____ Clips\n\nGiven Information:\n- Number of clips sold in April = 48\n- Number of clips sold in May = 1/2 * (Number of clips sold in April)\n\nSolution:\n\nLet's find the total number of clips sold by Natalia in April and ..."}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "She earned $", "router_reward": 0.62, "step_reward": 0.25, "outcome_reward": 0.0, "last_reasoning": "She earned $"}`
- `{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "ground_truth": "10", "final_answer": "Now that we know how many hours Weng worked (\\( \\frac{5", "router_reward": 0.66, "step_reward": 0.75, "outcome_reward": 0.0, "last_reasoning": "To solve the problem, follow these steps:\n\n**Step 1:** Convert the time from minutes to hours.\n\nWeng's babysitting time is given in minutes (50 minutes). We need to convert this into hours because her hourly wage is p..."}`
