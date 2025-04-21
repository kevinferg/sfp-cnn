import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

def bounded_spline(f, xmin=0, xmax=1):
    return lambda x: np.where((x < xmin) | (x > xmax), 0, f(x))

def normalize_pdf_spline(spline, n_grid=1000):
    x = np.linspace(0, 1, n_grid)
    y = np.clip(spline(x), 0, None)
    area = np.trapz(y, x)
    return lambda t: np.clip(spline(t), 0, None) / area

def random_spline(n_control_points=5, random_type='uniform', settings=(0, 1), k=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(0, 1, n_control_points)
    if random_type == 'normal':
        y = np.random.normal(*settings, size=n_control_points)
    else:
        y = np.random.uniform(*settings, size=n_control_points)
    return make_interp_spline(x, y, k=k)

def spline_sampler(pdf_spline, n_grid=1000):
    x = np.linspace(0, 1, n_grid)
    pdf_vals = np.clip(pdf_spline(x), 0, None)
    dx = x[1] - x[0]
    cdf_vals = np.cumsum(pdf_vals) * dx
    cdf_vals /= cdf_vals[-1]
    
    # Remove duplicate CDF values (monotonic required)
    cdf_vals_unique, idx = np.unique(cdf_vals, return_index=True)
    x_unique = x[idx]
    
    inv_cdf = make_interp_spline(cdf_vals_unique, x_unique, k=1)
    return lambda size=1: np.clip(inv_cdf(np.random.uniform(0, 1, size)), 0, 1)


if __name__ == "__main__":


    ####################
    # Generate Data

    pdf_raw = random_spline(n_control_points=7, random_type='uniform', k=5,seed=44)
    pdf = normalize_pdf_spline(pdf_raw)
    fx = bounded_spline(random_spline(5, 'normal', (.5, .1), k=3, seed=2))
    eps = bounded_spline(random_spline(4, 'uniform', (0, .2), k=1, seed=686))

    sample = spline_sampler(pdf)
    x_data = sample(500)

    y_data = fx(x_data) + np.random.normal(0, eps(x_data))

    #
    ####################



    ####################
    # Create Plot

    x_grid = np.linspace(0, 1, 1000)
    fig, ax1 = plt.subplots(figsize=(6,4),dpi=400)

    ax1.plot(x_grid, fx(x_grid), 'g', label='f(x)', linewidth=2)


    ax1.fill_between(x_grid, fx(x_grid) - 2 * eps(x_grid), fx(x_grid) + 2 * eps(x_grid), 
                    color='lightgray', alpha=0.6, label=r'f(x) $\pm$ 2u(x)')

    ax1.scatter(x_data, y_data, s=10, alpha=0.8, color='black', label='y ~ N(f(x), u(x))')

    ax2 = ax1.twinx()  # instantiate a second y-axis
    ax2.plot(x_grid, pdf_raw(x_grid), 'r', label='p(x)', linestyle='--', linewidth=2)
    ax2.set_ylabel('p(x)', color='r')  # set the y-axis label for p(x)
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_xticks([]) 
    ax1.set_yticks([])
    ax2.set_yticks([])

    ax1.set_xlabel('x')
    ax1.set_ylabel('y and f(x)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.set_ylim([0,1])
    ax2.set_ylim([-.25,1.15])

    plt.tight_layout()
    plt.savefig("uq-figure.png", bbox_inches="tight")
    plt.close()

    #
    ####################
