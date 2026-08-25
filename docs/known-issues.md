Code has certain known issues that I am documenting below -
1. Target App Run -> Langsmith Trace -> Parse back to local Trace dataobject
	This execution flow is super slow today as it happens at per trace level when the input calls are made. At the very least, an improvement would move this to once per run level. At best, I would eliminate Langsmith tracing entirely from the code and send traces straight to my local Trace object. This issue crept it after multiple phases were executing with Claude Code and I only realized later about the double data eventing.

2. The run executed for this turn is only on single turn conversations, the repo has full support for multi-turn but did not run due to time constraints.

3. The benchmark reports by ablating a smaller fraction of total traces for errors than the actual number of traces that can be ablated. This fraction can be increased for better benchmarking opportunity.

