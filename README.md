# SP4LTL
Solving an LTL specified problem through Self-Play
__________________________________________________
Problem description:
Suppose we have a grid world and an agent within that grid world. Suppose also that we have been given a set of LTL formulas specifying the rules and laws
That the agent should abide within this grid world. For example, assume we have the following grid world:

A -> Location of the agent

E -> Empty cell

O -> Obsticales

G -> Location of the goal


| E  | O | G |
| ------------- | ------------- | ------------- |
| E  | E  | E  |
| a  | O  | E  |

Also assume we have the following simple LTL specification: Eventually G, and always not O. (Meanting the agent a should reach G withough crossing over obstacles O)

<[]~O /\ <>G>

Now, we aim to create policies that abide by this rule and lead to trajectories which satisfy the given specs. We are using an AlphaGo zero approach to this problem, meaning that we have a policy+value network that outputs a policy (for movement) and a value (chances of satisfying the specs) at each time-step. We then run multiple Monte-Carlo rollouts from the current position to improve the given policy, and then, to make a move we sample from the resulting policy.
__________________________________________________
How to run the provided code:
1) get CSRL's code from their repo: https://gitlab.oit.duke.edu/cpsl/csrl and place it in the root folder (where al code has access to CSRL's filder)
2) Some changes inside CSRL's code might be needed depending on your version of packages and your OS, as it is using slightly older versions of packages such as numpy, but nothing too difficult.
   - hint: change np.objects to objects
   - hint: Add shell=True to the 'check_output' function arguments at line 83 of oa.py
   - Might require further changes if you are getting errors.
4) install needed packages to run the neural network, etc., like: numpy, TF (can be seen in the .ipynb codes). if you get a package not found error, you need to download it. The following are the packages we used:
   - keras==2.11.0
   - numpy==1.24.3
   - ply==3.11
   - tensorflow==2.11.0
2) get RABINIZER4 (CSRL needs it): https://www7.in.tum.de/~kretinsk/rabinizer4.html and add it to the PATH so that OS calls can find it.
    - hint: basically, if you run 'ltl2ldba' in the terminal, your OS should recognize it
4) the code in .ipynb files are straight forward and each correspond to one of the cases. You can change the hyperparameter inside the code and run them.
