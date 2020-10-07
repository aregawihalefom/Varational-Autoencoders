# Implements auto-encoding variational Bayes.
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
import matplotlib.pyplot as plt

from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import flatten  # This is used to flatten the params (transforms a list into a numpy array)


# images is an array with one row per image, file_name is the png file on which to save the images
def save_images(images, file_name): return s_images(images, file_name, vmin=0.0, vmax=1.0)


# Sigmoid activiation function to estimate probabilities

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Relu activation function for non-linearity

def relu(x):    return np.maximum(0, x)


# This function intializes the parameters of a deep neural network

def init_net_params(layer_sizes, scale=1e-2):
    """Build a (weights, biases) tuples for all layers."""

    return [(scale * npr.randn(m, n),  # weight matrix
             scale * npr.randn(n))  # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


# This will be used to normalize the activations of the NN

# This computes the output of a deep neuralnetwork with params a list with pairs of weights and biases

def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)  # nonlinear transformation

    # Last layer is linear

    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs


# This implements the reparametrization trick

def sample_latent_variables_from_posterior(encoder_output):
    # Params of a diagonal Gaussian.

    D = np.shape(encoder_output)[-1] // 2
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    # TODO use the reparametrization trick to generate one sample from q(z|x) per each batch datapoint
    # use npr.randn for that.
    epsilon = npr.randn(mean.shape[0], mean.shape[1])

    return mean + np.exp(log_std) * epsilon


# This evlauates the log of the term that depends on the data

def bernoulli_log_prob(targets, logits):
    # logits are in R
    # Targets must be between 0 and 1

    # skip this one , its the longer form
    # result = np.sum(np.log(np.multiply(targets,probs) + np.multiply((1-targets),(1-probs))),axis = 1)

    probs = sigmoid(logits)

    interal_part = targets * probs + (1 - targets) * (1 - probs)
    log_part = np.log(interal_part)
    sum_accross = np.sum(log_part, axis=1)

    return sum_accross


# This evaluates the KL between q and the prior
def compute_KL(q_means_and_log_stds):
    D = np.shape(q_means_and_log_stds)[-1] // 2
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]

    # WARNING : THE FOLLOWING FORMULA DOESN'T WORK FOR ME
    #  kl2 = np.sum( np.square(np.exp(log_std)) + np.square(mean)-1-np.log(np.square(np.exp(log_std))),axis = 1)

    # IN THIS CASE I ENCOUTERED A PROBLEM WITH THE FORMULA
    # GIVEN IN EQUATION 12 , THEN AFTER RESERCHING ON THE INTERNET
    # I FINID OUT THAT THERE IS COFFICINET 2 INFRONT OF THE LAST
    # TERM THE LOG OG THE variance

    std = np.exp(log_std)
    div = 0.5 * (np.square(std) + (np.square(mean) - 1 - 2 * np.log(std)))

    # sum accross latent dimention
    KL = np.sum(div, axis=-1)

    return KL


# This evaluates the lower bound
def vae_lower_bound(gen_params, rec_params, data):
    # TODO compute a noisy estiamte of the lower bound by using a single Monte Carlo sample:

    # 1 - compute the encoder output using neural_net_predict given the data and rec_params
    encoder_prediction = neural_net_predict(rec_params, data)

    # 2 - sample the latent variables associated to the batch in data
    #     (use sample_latent_variables_from_posterior and the encoder output)
    z_space = sample_latent_variables_from_posterior(encoder_prediction)

    # 3 - use the sampled latent variables to reconstruct the image and to compute the log_prob of the actual data
    #     (use neural_net_predict for that)
    recon_output = neural_net_predict(gen_params, z_space)

    # 4 - compute the KL divergence between q(z|x) and the prior (use compute_KL for that)
    KL_div = compute_KL(encoder_prediction)

    # 5 - return an average estimate (per batch point) of the lower bound
    # by substracting the KL to the data dependent term
    bern_log_probs = bernoulli_log_prob(data, recon_output)

    estimate = np.mean(bern_log_probs - KL_div)

    return estimate


# Model hyper-parameters
npr.seed(0)  # We fix the random seed for reproducibility

latent_dim = 50
data_dim = 784  # How many pixels in each image (28x28).
n_units = 200
n_layers = 2

gen_layer_sizes = [latent_dim] + [n_units for i in range(n_layers)] + [data_dim]
rec_layer_sizes = [data_dim] + [n_units for i in range(n_layers)] + [latent_dim * 2]

# Training parameters
batch_size = 200
num_epochs = 1
learning_rate = 0.001

print("Loading training data...")

N, train_images, _, test_images, _ = load_mnist()

# Parameters for the generator network p(x|z)
init_gen_params = init_net_params(gen_layer_sizes)

# Parameters for the recognition network p(z|x)
init_rec_params = init_net_params(rec_layer_sizes)

combined_params_init = (init_gen_params, init_rec_params)

num_batches = int(np.ceil(len(train_images) / batch_size))

# We flatten the parameters (transform the lists or tupples into numpy arrays)

flattened_combined_params_init, unflat_params = flatten(combined_params_init)


# Actual objective to optpimize that receives flattened params
def objective(flattened_combined_params):
    combined_params = unflat_params(flattened_combined_params)
    data_idx = batch
    gen_params, rec_params = combined_params

    # We binarize the data
    on = train_images[data_idx, :] > npr.uniform(size=train_images[data_idx, :].shape)
    images = train_images[data_idx, :] * 0.0
    images[on] = 1.0

    return vae_lower_bound(gen_params, rec_params, images)


# Get gradients of objective using autograd.
objective_grad = grad(objective)
flattened_current_params = flattened_combined_params_init

# TODO write here the initial values for the ADAM parameters (including the m and v vectors)
# you can use np.zeros_like(flattened_current_params) to initialize m and v

# ADAM parameters

alpha = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = pow(10, -8)

m = np.zeros_like(flattened_current_params)
v = np.zeros_like(flattened_current_params)
t = 1

# We do the actual training
for epoch in range(num_epochs):

    elbo_est = 0.0

    for n_batch in range(int(np.ceil(N / batch_size))):
        batch = np.arange(batch_size * n_batch, np.minimum(N, batch_size * (n_batch + 1)))

        # TODO Use the estimated noisy gradient in grad to update the paramters using the ADAM updates

        grad_t = objective_grad(flattened_current_params)

        m = beta1 * m + (1 - beta1) * grad_t
        v = beta2 * v + (1 - beta2) * (np.square(grad_t))

        m_hat = m / (1 - (beta1 ** t))
        v_hat = v / (1 - (beta2 ** t))

        # since it is maximatization it should be plus
        flattened_current_params = flattened_current_params + alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        elbo_est += objective(flattened_current_params)
        t = t + 1
    print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N / batch_size)))

# We obtain the final trained parameters
gen_params, rec_params = unflat_params(flattened_current_params)

# TODO Generate 25 images from prior (use neural_net_predict) and save them using save_images

# generate 25  z from uniform prior
z = np.random.normal(0, 1, (25, latent_dim))

# generate x using generator network
logit = neural_net_predict(gen_params, z)

# actual images
gen_images = sigmoid(logit)

# save images
save_images(gen_images, './images/generated_images.png')

# TODO Generate image reconstructions for the first 10 test images (use neural_net_predict for each model)
# and save them alongside with the original image using save_images

# generate 10 images form the recognition model


ten_test_images = test_images[:10, :]

ten_gen_images = neural_net_predict(rec_params, ten_test_images)

# sample form the postorior
latent_space = sample_latent_variables_from_posterior(ten_gen_images)

# now predict using the prediction model
ten_logits = neural_net_predict(gen_params, latent_space)

# get the images [ black and white]
ten_new_images = sigmoid(ten_logits)

# concatinate witht he original images ..
orginal_and_generated = np.concatenate([test_images[:10, :], ten_new_images])

# save images
save_images(orginal_and_generated, './images/combined.png')

# TODO Generate 5 interpolations from the first test image to the second test image,

D = np.shape(ten_gen_images)[-1] // 2
latent_mu, log_std = ten_gen_images[:, :D], ten_gen_images[:, D:]

# generate s [30 in this case according to the example in the manual]
s_values = np.linspace(0, 1, 25)
# to accumlate each results

# TODO Generate 5 interpolations from the first test image to the second test image,

num_interpolations = 5
for i in range(num_interpolations):

    # interpolation vectors
    # from the weighted latent mixture
    inter_vectors = []

    for j in range(25):

        idx_1 = 2 * i  # first image index
        idx_2 = 2 * i + 1  # second image index

        # weight the means of the latent variables
        # The weights are reversed to (1-s) to first image and the remaining s to second image
        # to match the results fo the reference document
        z_mix = (1 - s_values[j]) * latent_mu[idx_1, :] + s_values[j] * latent_mu[idx_2, :]

        # append the results .. to accumulate results
        inter_vectors.append(z_mix)

    # generate using the gernetor form the mixutre latent
    gen_image = neural_net_predict(gen_params, inter_vectors)

    # get the actual results of the ACTIVATION FUNCTION
    result_image = sigmoid(gen_image)

    file_name = "./images/Interpolation_image_{}".format(i + 1)
    save_images(result_image, file_name)
