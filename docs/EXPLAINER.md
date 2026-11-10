# Putting a €60 improvement to the test

In one fictional lease case, moving the interest rate from 9% to 10% appeared to add about €60 in expected contribution: financial return after allowing for acceptance and decision costs. It was a tempting improvement. The question behind our experiments was what evidence would justify acting on it.

We began with a deliberately simple world. Customers, responses and economics were invented, with acceptance following a smooth pattern. Counting past responses gave us a reference; logistic regression learned relationships between business characteristics, price and acceptance. This gave us a manageable starting point before asking the models to handle harder conditions.

Even here, a good fit to the past was insufficient. We separated examples used for training from those used to judge settings, and kept a final test untouched until model selection was complete. We varied features and hyperparameters—the settings controlling how a model learns—and compared prediction losses, where lower means better. The aim was to find a model that carried useful patterns into unseen examples.

Then we made the world more complex. Business characteristics were linked, while hidden urgency and competing offers affected responses. We compared logistic regression with boosted decision trees and pretrained TabPFN, which brings learning from other prediction tasks.

The training curves showed why that comparison needed care. In the complex world, the selected tree model’s validation loss reached its minimum after 15 rounds. Further training kept reducing errors on its training examples while worsening performance on the separate validation examples. We retained the earlier version: learning the training data better had stopped helping elsewhere.

The results resisted a simple “more complex wins” story. With 512 fitting examples, TabPFN slightly beat logistic regression in the complex world, but logistic regression did better in the simple world. With more fitting examples and bounded tuning, selected logistic regression had the lowest final prediction loss among the main methods in both worlds. This second comparison used unequal development budgets; TabPFN also brought external pretraining. The advantage depended on the conditions of the comparison.

Then we changed customer behaviour itself. Yesterday’s relationship between price and acceptance no longer held. This exposed a different weakness: probability estimates could become systematically misleading.

Calibration asks whether forecasts of 70% correspond to roughly seven acceptances in ten comparable cases. In a separate changed-response experiment, we adjusted the probability scale using 900 observed responses, then evaluated it on another 900. The average calibration gap fell from about 18 percentage points to 4. This improved the probabilities in that test. It did not validate uncertainty estimates, and the main model comparison had used no additional calibration.

Better forecasts still had to support decisions as new responses arrived. Could a policy learn its way through change? Thompson sampling sometimes tries an offer because it might prove better. In stable conditions, it beat the fixed offer policy on average expected contribution with both uncertainty models. When behaviour changed halfway through the replay, both finished behind it on average across 20 simulation runs. Learning had value, but these learners could not be assumed to adapt quickly enough.

We also ran breakpoint experiments, deliberately pushing assumptions until they failed. Accepted-only histories could not establish refusal rates; sparse histories and unfamiliar prices required review. In test cases where examples almost perfectly separated acceptances from refusals, one method’s uncertainty estimates failed their accuracy check. A separate attempt to estimate tree uncertainty missed its precision requirement, so we kept their probability predictions without using them for cautious decisions. Prediction, calibration and reliable uncertainty each needed their own evidence.

All of this returned us to the original decision. Maximising average estimated profit was one possible rule. We also considered a cautious rule requiring the lower end of estimated improvement to clear a predefined margin.

In the separate counting-based uncertainty model, the growing-business case still favoured 10% by about €60 on average. But roughly one in twenty model estimates put the improvement below −€27. The cautious rule required that lower estimate to exceed €20, so it kept 9%. These were estimates of expected contribution, not individual customer profits; the prediction leaderboard supplied no such uncertainty bounds.

We started by asking which offer looked best. The experiments gradually made the question more demanding: had the model generalised, were its probabilities calibrated, did its uncertainty hold up, and did relevant evidence support the decision? The resulting recommendation was only as useful as those answers. Testing its value for real lease customers remains a further step.

---

[Results and limits](RESULTS.md) · [Methods and calibration details](METHODS.md) · [One-minute walkthrough](WALKTHROUGH.md)

Built with PriorLabs-TabPFN.
