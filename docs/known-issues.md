Code has certain known issues that I am documenting below -
1. Target App Run -> Langsmith Trace -> Parse back to local Trace dataobject
	This execution flow is super slow today as it happens at per trace level when the input calls are made. At the very least, an improvement would move this to once per run level. At best, I would eliminate Langsmith tracing entirely from the code and send traces straight to my local Trace object. This issue crept it after multiple phases were executing with Claude Code and I only realized later about the double data eventing.

2. The run executed for this turn is only on single turn conversations, the repo has full support for multi-turn but did not run due to time constraints.

3. The benchmark reports by ablating a smaller fraction of total traces for errors than the actual number of traces that can be ablated. This fraction can be increased for better benchmarking opportunity.  

4. `state_loss` category of errors although was injected and accounted for in the ground truth, engine can only detect these in multi-turn convos. Since the dataset I benchmarked against was single turn only, engine missing this got penalized but it shouldn't have actually gotten penalized. This should reduce GT by 10 traces. Ideally there should be algorithm level gates on error category eligibility - if this can be generalized to certain error categories.  

5. Similar to above, the ablation engine injected 2 error types under the `other` category which it should ideally have ignored (that's the E_h pool for engine). These also become ineligible ground truth errors to judge engine against. In general, the `other` category is expected to always have 0 P,R because no known ablation should be of the `other` type. Due to this, it should also be pulled out of the macro-f1 reporting. For this pipeline run - it implies GT reduction by 10 traces. Net, both #4 and #5 combined, the GT should only be on 50 traces max instead of the ~70 (65) it was measured against.  

6. The dummy engine I built seems to have captured a high recall of the injected issues but categorized it as a different error category. This is an interesting and nuanced case of how to interpret an error - the same error can be hallucination or instruction_violation or tool_misuse - and because my scorers are too strict (either check for exact match of error category or cosine similarity on TfiDF vectors at 0.5 threshold) the engine's labels are not reconciled with ground truth accurately (i.e. E_p --> E_k matcher is too strict hence very few matches established). To fix this - I can - 
	a. ablate better with less overlap i.e. specific surgical edits only; 
	b. make engine predict multiple error categories (or secondary categories) and use all categories in the matcher; 
	c. use LLM-as-judge (last resort) to establish error matches;  
	d. calibrate label matching algo to not be super strict or explore other matching algos that are robust  

7. For the error-level scorer to compute PRF1 per error, the foundational pre-req is to be able to map each predicted error to a known error E_p_i --> E_k_i; I considered this during ablation and made ablation engine disjoint at the error-level by error-category - i.e. same trace will not be ablated for 2 distinct error types of the same category (diff categories is fine). This was done so that (trace_id, error_category) will uniquely map to E_k_i; But the problem I retrospectively realized is the matching algorithm guarantees Precision = 1, since only those E_p_i will be considered that found a match to an E_k_i - hence by definition FP = 0 when there is any resolution, or TP = 0 otherwise. So error-level precision can only be 0 or 1, making the metric moot. This should ideally be measured on Layer 2 resolution instead of Layer 1 resolution. Layer 2 resolution actually accomplishes the error-level map from E_p_i --> E_k_i by argmax(intersection) per predicted error.  

8. My trace ETL has problems due to which tool args got stripped out from the spans and the dummy engine flagged a large number of traces for missing tool args which is not a FP.  
