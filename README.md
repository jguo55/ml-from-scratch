# optim-visualizer
visualization of gradient descent optimization algorithms
https://arxiv.org/pdf/1609.04747

# Momentum:

Loss is plotted at each point of one weight being adjusted.
One weight being adjusted, then the network trains.

Not sure how effective this example is. Using a higher dimensional input with PCA would work better, but I don't know if it would work.


# Contours:

The loss contours for a random dataset are plotted for a simple network, and then various optimizers are used and compared to one another.

The network consists of 3 linear layers, with two inputs. Only the first layer is used and adjusted during backpropagation. This restricts the dimensionality of the problem to two dimensions, making it easier to visualize each optimizer reducing its loss.

After the contours are plotted, the network will start from the highest point from the -7.5 to 7.5 area and each model is updated at every step. The models should eventually reach the lower points, at various points.

Observations:
The momentum based optimizer seems to perform the best, but I'm assuming this is because of how simple the problem is - The problem is not complicated enough for Adam to show its advantages

Additional Notes:
If the graph does not show the loss converging to 0 (no solution), you can simply close and rerun the program.