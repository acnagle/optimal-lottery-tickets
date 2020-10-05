---- Requirements and Setup ----
Python 3.6.9, CUDA Version 10.2
Create a new python environment and install all package dependencies provided in requirements.txt. Once your environment is created, you can install the requirements with 

pip install -r requirements.txt 

or with

conda install --file requirements.txt

depending on how you created your environment. If using Anaconda, be sure to add the Gurobi channel before installing from requirements.txt:

conda config --add channels http://conda.anaconda.org/gurobi

If you're using pip, you will need to perform the installation as specified in the resources below. 

Experiment 1 uses a package called gurobipy, which is a package developed by Gurobi which allows the users to model optimization problems and solve them using Gurobi's solvers. gurobipy is installed differently depending on the package manager:
- Anaconda
    - Mac: https://www.gurobi.com/gurobi-and-anaconda-for-mac/
    - Windows: https://www.gurobi.com/gurobi-and-anaconda-for-windows/
    - Linux: https://www.gurobi.com/gurobi-and-anaconda-for-linux/
- Install gurobipy with a different package manager
    - Mac: https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html
    - Windows: https://www.gurobi.com/documentation/9.0/quickstart_windows/the_grb_python_interface_f.html
    - Linux: https://www.gurobi.com/documentation/9.0/quickstart_linux/the_grb_python_interface_f.html

Once gurobipy is installed, a license is required to use the package; Gurobi offers unlimited free academic licenses. Information about downloading Gurobi and obtaining a license may be found at:
- https://www.gurobi.com/academia/academic-program-and-licenses/
- https://www.gurobi.com/downloads/

Gurobi also provides quickstart guides:
- Mac: https://www.gurobi.com/wp-content/plugins/hd_documentations/documentation/9.0/quickstart_mac.pdf
- Windows: https://www.gurobi.com/wp-content/plugins/hd_documentations/documentation/9.0/quickstart_windows.pdf
- Linux: https://www.gurobi.com/wp-content/plugins/hd_documentations/documentation/9.0/quickstart_linux.pdf

---- Experiments ----
In both experiment directories, subset-sum and random-networks, one or more bash scripts contain all commands to reproduce the results in the experiments provided in the paper. Please run

python main.py --help

in the terminal to see information about all arguments and their default values. In order to execute a script, run the following commands:

chmod u+x script.sh

only once to allow the script to be executable, and then navigate to the results directory and run

../script.sh

If you would like to execute a script in the background and save all output to an output file, you may replace the above command with:

nohup ../script.sh > script.out &

Within the results directories, there is a paper directory, which includes all of the results of the experiments discussed in our paper.

-- Experiment 1: SubsetSum --
Because of the hardness of the SubsetSum problem, it can take a long time to solve SubsetSum for all 397,000 parameters of a two-layer fully connected network. Our experiments use Gurobi's MIP solver (https://www.gurobi.com/resource/mip-basics/), and this solver runs in parallel to solve a given optimization problem. Our implemention will utilize all cores on a machine to solve each SubsetSum problem; to limit the core usage, we recommend running the experiment in the following way:

taskset -c 1-16 nohup ../run.sh > run.out &

This limits the number of cores used when executing run.sh to be between 1 and 16. Since every python script is run one at a time within run.sh, executing run.sh in this way will dedicate 1 to 16 cores to main.py. Nearly all of these cores will be dedicated to solving each SubsetSum instance once that part of the code is reached. The approximation of a two-layer fully connected network presented in our paper completed in about 21.5 hours on 36 cores of a c5.18xlarge AWS EC2 instance.

While most arguments to main.py are self-explanatory, there are a few arguments that would benefit from further explanation. Arguments --c and --epsilon determine the number of a_i coefficents used to approximate every weight in a target network via SubsetSum. The number of a_i coefficients is given by n = round(c * log2(1 / epsilon)). The --check_w_lt_eps argument, when enabled, will check whether a weight magnitude is less than or equal to epsilon before running SubsetSum on that weight value. If the magnitude is less than or equal to epsilon, the weight is approximated to be zero. This allows the script to terminate much earlier, have all weights approximations be within epsilon error, and the approximated target network will still an accuracy very close to the target network.

In the subuset-sum directory, a run.sh script contains the necessary commands to reproduce our results. Be sure to execute these scripts from the results directory.

-- Experiment 2: Pruning Random Networks --

Although most arguments used when running main.py are straight-forward, some arguments and naming conventions deserve a more thorough explanation:
(1) Throughout the code, the word "redundant" (sometimes shortened to the prefix "red") occurs frequently. This is a naming convention that refers to our structure. Essentially, our structure introduces some "redundancy" into a given architecture by replacing a single weight value with r different weight values, which are then pruned. The number of redundant units in a network with our structure is specified by the argument --r. For example, running

python main.py --model "redfc2" --r 5

instructs main.py to run an experiment for a two-layer fully connected network with our structure and 5 units of redundancy.

(2) For networks that use our structure and for baseline networks, the argument --hidden-size  works as one might expect: it specifies the number of hidden nodes in the fully connected layers of a network. There are two exceptions to this rule: (1) LeNet5 architectures have a predetermined number of hidden nodes based on the literature, so --hidden-size is ignored. (2) When pruning a wide network, such as "widefc2" (wide two-layer fully connected network), the number of hidden nodes in the architecture is calculated based off the value specified by --hidden-size and --r. The the number hidden nodes are calculated this way so that the number of parameters in a wide network approximately match the number of parameters in a network that uses our structure when the same architecture is used. For example, if a "redfc2" (our network) is defined for --hidden-size 500 and --r 5, then defining a "widefc2" with --hidden-size 500 and --r 5 will prune a wide network with the same number of parameters as the "redfc2" network.

(3) To avoid confusion, the sparsity argument has the following interpretation: a sparsity of 0 means "no sparsity" (i.e. "keep all the weights") and a sparsity of 1 means "complete sparsity" (i.e. "prune all the weights"). By this convention, a sparsity of 0.75 means "keep 25% of the weights."

In the random-networks directory, there are three bash scripts for our experiments: (1) base.sh contains all the commands for our baseline network results, (2) fc.sh contains all the commands for two- and four-layer fully connected networks that use our redundant structure and widened networks, and (3) lenet5.sh contains all the commands for our implementing our pruned redundant structure and the wide fully connected layers on LeNet5. Be sure to execute these scripts from the results directory.
