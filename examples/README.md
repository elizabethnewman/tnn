## Running Experiments

The syntax to run experiments from the command line is the following:

```console
python run_experiments.py test.csv
```

The ```csv``` files contain the arguments one wants to pass through the script.

## Experiments from the Paper

The following ```csv``` files will run the same experiments as those presented in the paper.
* MNIST Autoencoder: ```autoencoder_paper.csv```
* MNIST Classification: ```mnist_paper2.csv```
* CIFAR10 Classification: ```cifar10_paper2.csv```
* CIFAR10 Timing: ```cifar10_paper.csv```
* CIFAR100 Classification: ```cifar100_paper.csv```

To replicate the visualizations for Hamiltonian layers, run the following:
```console
python ex_trajectories/ex_euler2.py
python ex_trajectories/ex_leapfrog.py
```
