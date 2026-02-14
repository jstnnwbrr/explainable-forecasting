Implementation Guide for Non-Technical Stakeholders



1\. The Core Strategy



Stop showing stakeholders coefficients, Shapley values, or architecture diagrams. They are looking for Agency and Safety.



Agency: "If I change Price, what happens?" (Solved by Sensitivity Analysis).



Safety: "Has this happened before?" (Solved by Nearest Neighbors).



2\. Model 1 vs. Model 2 Definitions



Model 1 (The Time-Series Ensemble)



Composition: N-BEATS, N-HiTS, Prophet, Auto-ARIMA, Simple Exp. Smoothing.



Philosophy: "History Repeats." These models look at the shape of the line. They are excellent at catching seasonality and subtle trends but can be slow to react to sudden external shocks (like a price change) unless explicitly trained with covariates.



Model 2 (The Regression Ensemble)



Composition: ElasticNet, PassiveAggressive, Polynomial Reg, TheilSen, MLP.



Philosophy: "Drivers Cause Outcomes." These models don't care about 'Time'; they care about inputs. They are excellent at "What-If" scenarios (pricing, marketing) but often fail to capture organic momentum or seasonality.



3\. How to use the "Counterfactual" argument



When a stakeholder asks: "Why is the fancy AI (Model 1) predicting higher sales than the Regression (Model 2)?"



Don't say: "The N-BEATS architecture has a trend block that identified a non-linear uplift."



Do say: "Model 2 thinks sales are purely driven by Marketing Spend. Model 1 disagrees—it sees 'Organic Momentum' in the data. For Model 2 to be right, that organic momentum would have to suddenly disappear today."



4\. The "Accuracy Tax" Conversation



Keep the Scatter Plot (from Tab 4 in the app) handy.

When they push for transparency:

"We can switch to ElasticNet (Model 2). It will be perfectly explainable. But based on backtesting, that decision will cost us roughly $50k in inventory errors per quarter. Are we comfortable paying that 'Tax' for the comfort of knowing 'Why'?"

