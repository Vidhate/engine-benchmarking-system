Instructions: 
We recently launched Engine - an agent that runs over traces. Specifically:
Customer is running agent, it sends traces
We run our agent (Engine) over those traces to:
Identify clusters of issues
Propose fixes
Build a benchmark for Engine. A single task. Your result should return:
Inputs: json file of traces (at least 300), issueboard
Expected Outputs: updated issueboard
Function for scoring updated issueboard Engine produces against the real expected outputs. Does not need to focus on “fixes” generated, just the issueboard itself
1-2 pages with thoughts and details on the below
Things to think about:
What is right data structure for a trace
What is right data structure for issueboard
What is the right evaluation function
How do you create realistic traces
How does your methodology scale to creating many tasks of similar or larger size
Does Sol perform better than 5.1 mini on this task? You won’t be able to run real Engine on this, but you can run a coding agent (with some custom instructions) to simulate it
