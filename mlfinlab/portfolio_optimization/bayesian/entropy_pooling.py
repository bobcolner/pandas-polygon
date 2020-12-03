# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
#pylint: disable=missing-docstring
import math
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_slsqp
import matplotlib.pyplot as plt


class EntropyPooling:
    """
    This class implements the Entropy Pooling algorithm proposed in the following paper: Meucci, Attilio, Fully Flexible
    Views: Theory and Practice (August 8, 2008). Fully Flexible Views: Theory and Practice. By using historical factor
    observations as a prior, EP combines it and additional investor views on the portfolio, to find a posterior
    distribution which is close to the prior and also satisfies the specified views. It also removes any assumptions on the
    distribution of the prior and produces the posterior probabilities in a non-parametric way.
    """

    def __init__(self):
        """
        Initialise.

        Class Variables:

        - ``posterior_probabilities`` - (pd.DataFrame) Final posterior probabilities calculated using Entropy Pooling algorithm.
        """

        self.posterior_probabilities = None

    def calculate_posterior_probabilities(self, prior_probabilities, equality_matrix=None, equality_vector=None, inequality_matrix=None,
                                          inequality_vector=None, view_confidence=1.0):
        """
        Calculate posterior probabilities from an initial set of probabilities using the Entropy Pooling algorithm.

        :param prior_probabilities: (Numpy array/Python list) List of initial probabilities of market simulations.
        :param equality_matrix: (pd.DataFrame/Numpy matrix) A (J x N1) matrix of equality constraints where N1 = number of equality views
                                                            and J = number of historical simulations. Denoted as 'H' in the "Meucci - Flexible
                                                            Views Theory & Practice" paper in formula 86 on page 22.
        :param equality_vector: (Numpy array/Python list) A vector of length J corresponding to the equality matrix. Denoted as 'h' in the "Meucci -
                                                          Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param inequality_matrix: (pd.DataFrame/Numpy matrix) A (J x N2) matrix of inequality constraints where N2 = number of inequality
                                                              views and J = number of historical simulations. Denoted as 'F' in the "Meucci -
                                                              Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param inequality_vector: (Numpy array/Python list) A vector of length J corresponding to the inequality matrix. Denoted as 'f' in the "Meucci -
                                                            Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param view_confidence: (float) An overall confidence in the specified views.
        """

        # Initial check of inputs
        self._error_checks(prior_probabilities, equality_matrix, equality_vector, inequality_matrix, inequality_vector)

        num_equality_constraints = 0 if equality_matrix is None else equality_matrix.shape[1]
        num_inequality_constraints = 0 if inequality_matrix is None else inequality_matrix.shape[1]
        prior_probabilities = np.reshape(prior_probabilities, (len(prior_probabilities), 1))
        initial_guess = np.zeros(shape=(num_equality_constraints + num_inequality_constraints, 1))

        if inequality_matrix is None:
            equality_vector = np.reshape(equality_vector, (len(equality_vector), 1))
            self.posterior_probabilities = self._solve_unconstrained_optimisation(initial_guess, prior_probabilities, equality_matrix, equality_vector)
        else:
            equality_vector = np.reshape(equality_vector, (len(equality_vector), 1))
            inequality_vector = np.reshape(inequality_vector, (len(inequality_vector), 1))
            self.posterior_probabilities = self._solve_constrained_optimisation(initial_guess, prior_probabilities, equality_matrix, equality_vector,
                                                                                inequality_matrix, inequality_vector, num_equality_constraints,
                                                                                num_inequality_constraints)

        # Confidence weighted sum of prior and posterior probabilities
        self.posterior_probabilities = view_confidence * self.posterior_probabilities + (1 - view_confidence) * prior_probabilities

    def generate_histogram(self, historical_market_vector, num_bins):
        """
        Given the final probabilities, generate the probability density histogram from the historical market data points.

        :param historical_market_vector: (pd.Series/Numpy array) Vector of historical market data.
        :param num_bins: (int) The number of bins to break the histogram into.
        :return: (plt.BarContainer) The plotted histogram figure object.
        """

        historical_market_vector = np.array(historical_market_vector).reshape((len(historical_market_vector, )))
        _, bin_breaks, _ = plt.hist(x=historical_market_vector, bins=num_bins)
        bin_width = bin_breaks[1] - bin_breaks[0]
        num_of_breaks = len(bin_breaks)
        new_probabilities = np.zeros((num_of_breaks, 1))

        frequency = None
        for bin_index in range(num_of_breaks):
            indices = (historical_market_vector >= bin_breaks[bin_index] - bin_width / 2) & \
                    (historical_market_vector < bin_breaks[bin_index] + bin_width / 2)
            new_probabilities[bin_index] = np.sum(self.posterior_probabilities[indices])
            frequency = new_probabilities / bin_width
        frequency = frequency.reshape((len(frequency), ))
        figure = plt.bar(bin_breaks, frequency, width=bin_width)
        return figure

    @staticmethod
    def _solve_unconstrained_optimisation(initial_guess, prior_probabilities, equality_matrix, equality_vector):
        """
        Solve the unconstrained optimisation using Lagrange multipliers. This will give us the final posterior probabilities.

        :param initial_guess: (Numpy array) An initial starting vector for the optimisation algorithm.
        :param prior_probabilities: (Numpy array) List of initial probabilities of market simulations.
        :param equality_matrix: (Numpy matrix) An (N1 x J) matrix of equality constraints where N1 = number of equality views
                                               and J = number of historical simulations.
        :param equality_vector: (Numpy array) A vector of length J corresponding to the equality matrix.
        :return: (Numpy array) Posterior probabilities.
        """

        equality_matrix = equality_matrix.T

        def _cost_func(equality_lagrange_multplier):
            # pylint: disable=invalid-name
            """
            Cost function of the unconstrained optimisation problem.

            :param equality_lagrange_multplier: (Numpy matrix) The Lagrange multiplier corresponding to the equality constraints.
            :return: (float) Negative of the value of the Langrangian.
            """

            equality_lagrange_multplier = equality_lagrange_multplier.reshape(-1, 1)
            x = np.exp(np.log(prior_probabilities) - 1 - equality_matrix.T.dot(equality_lagrange_multplier))

            # L = x'(log(x) - log(p)) + v'(Hx - h)
            langrangian = x.T.dot(np.log(x) - np.log(prior_probabilities)) + equality_lagrange_multplier.T.dot(equality_matrix.dot(x) - equality_vector)
            return -langrangian

        def _cost_func_jacobian(equality_lagrange_multplier):
            # pylint: disable=invalid-name
            """
            Jacobian of the cost function.

            :param equality_lagrange_multplier: (Numpy matrix) The Lagrange multiplier corresponding to the equality constraints.
            :return: (float) Negative of the value of the Langrangian gradient.
            """

            equality_lagrange_multplier = equality_lagrange_multplier.reshape(-1, 1)
            x = np.exp(np.log(prior_probabilities) - 1 - equality_matrix.T.dot(equality_lagrange_multplier))
            langrangian_gradient = equality_matrix.dot(x) - equality_vector
            return -langrangian_gradient

        optimal_multiplier, _, _ = fmin_l_bfgs_b(x0=initial_guess,
                                                 func=_cost_func,
                                                 fprime=_cost_func_jacobian,
                                                 maxfun=1000,
                                                 pgtol=1.e-6,
                                                 disp=0)
        optimal_multiplier = optimal_multiplier.reshape(-1, 1)
        optimal_p = np.exp(np.log(prior_probabilities) - 1 - equality_matrix.T.dot(optimal_multiplier))
        return optimal_p

    @staticmethod
    def _solve_constrained_optimisation(initial_guess, prior_probabilities, equality_matrix, equality_vector, inequality_matrix, inequality_vector,
                                        num_equality_constraints, num_inequality_constraints):
        """
        Solve the constrained optimisation using Lagrange multipliers. This will give us the final posterior probabilities.

        :param initial_guess: (Numpy array) An initial starting vector for the optimisation algorithm.
        :param prior_probabilities: (Numpy array) List of initial probabilities of market simulations.
        :param equality_matrix: (Numpy matrix) An (N1 x J) matrix of equality constraints where N1 = number of equality views
                                               and J = number of historical simulations.
        :param equality_vector: (Numpy array) A vector of length J corresponding to the equality matrix.
        :param inequality_matrix: (Numpy matrix) An (N2 x J) matrix of inequality constraints where N2 = number of inequality
                                                              views and J = number of historical simulations.
        :param inequality_vector: (Numpy array) A vector of length J corresponding to the inequality matrix.
        :param num_equality_constraints: (int) Number of equality views/constraints.
        :param num_inequality_constraints: (int) Number of inequality views/constraints.
        :return: (Numpy array) Posterior probabilities.
        """

        equality_matrix = equality_matrix.T
        inequality_matrix = inequality_matrix.T
        identity = np.identity(n=(num_equality_constraints + num_inequality_constraints), dtype=float)
        identity = identity[:num_inequality_constraints]

        def _inequality_constraints_func(all_constraints_vector):
            """
            Calculate inequality cost function.

            :param all_constraints_vector: (Numpy matrix) Combined vector of all the constraints - equality and inequality.
            :return: (Numpy matrix) Vector of inequality constraints.
            """

            return -identity.dot(all_constraints_vector)

        def _inequality_constraints_func_jacobian(all_constraints_vector):
            #pylint: disable=unused-argument
            """
            Jacobian of the inequality constraints cost function.

            :param all_constraints_vector: (Numpy matrix) Combined vector of all the constraints - equality and inequality.
            :return: (Numpy matrix) Identity matrix.
            """

            return -identity

        def _cost_func(lagrange_multipliers):
            # pylint: disable=invalid-name
            """
            Cost function of the constrained optimisation problem.

            :param lagrange_multipliers: (Numpy matrix) Values of the Lagrange multipliers for inequality and equality constraints.
            :return: (float) Negative of the value of the Langrangian.
            """

            inequality_multiplier = lagrange_multipliers[:num_inequality_constraints].reshape(-1, 1)
            equality_multiplier = lagrange_multipliers[num_inequality_constraints:].reshape(-1, 1)
            x = np.exp(np.log(prior_probabilities) - 1 - inequality_matrix.T.dot(inequality_multiplier) - equality_matrix.T.dot(equality_multiplier))

            # L = x'(log(x) - log(p)) + l'(Fx - f) + v'(Hx - h)
            langrangian = x.T.dot(np.log(x) - np.log(prior_probabilities)) + \
                          inequality_multiplier.T.dot(inequality_matrix.dot(x) - inequality_vector) + \
                          equality_multiplier.T.dot(equality_matrix.dot(x) - equality_vector)
            return -langrangian

        def _cost_func_jacobian(lagrange_multipliers):
            #pylint: disable=invalid-unary-operand-type, invalid-name
            """
            Jacobian of the cost function.

            :param lagrange_multipliers: (Numpy matrix) Values of the Lagrange multipliers for inequality and equality constraints.
            :return: (Numpy matrix) Negative of the value of the Langrangian gradients.
            """

            inequality_multiplier = lagrange_multipliers[:num_inequality_constraints].reshape(-1, 1)
            equality_multiplier = lagrange_multipliers[num_inequality_constraints:].reshape(-1, 1)
            x = np.exp(np.log(prior_probabilities) - 1 - inequality_matrix.T.dot(inequality_multiplier) - equality_matrix.T.dot(equality_multiplier))
            langrangian_gradient = np.vstack([
                inequality_matrix.dot(x) - inequality_vector,
                equality_matrix.dot(x) - equality_vector
            ])
            return -langrangian_gradient

        optimal_lagrange_multipliers = fmin_slsqp(x0=initial_guess,
                                                  func=_cost_func,
                                                  fprime=_cost_func_jacobian,
                                                  f_ieqcons=_inequality_constraints_func,
                                                  fprime_ieqcons=_inequality_constraints_func_jacobian,
                                                  disp=0)

        # Inequality Lagrange multiplier values (l)
        optimal_inequality_multiplier = optimal_lagrange_multipliers[:num_inequality_constraints]
        optimal_inequality_multiplier = np.reshape(optimal_inequality_multiplier, (len(optimal_inequality_multiplier), 1))

        # Equality Lagrange multiplier values (v)
        optimal_equality_multiplier = optimal_lagrange_multipliers[num_inequality_constraints:]
        optimal_equality_multiplier = np.reshape(optimal_equality_multiplier, (len(optimal_equality_multiplier), 1))

        # Calculate final posterior probabilities by finding the optimal Langrangian value.
        optimal_p = np.log(prior_probabilities) - \
                    1 - \
                    inequality_matrix.T.dot(optimal_inequality_multiplier) - \
                    equality_matrix.T.dot(optimal_equality_multiplier)
        optimal_p = np.exp(optimal_p)
        return optimal_p

    @staticmethod
    def _error_checks(prior_probabilities, equality_matrix, equality_vector, inequality_matrix, inequality_vector):
        """
        Initial error checks on inputs.

        :param prior_probabilities: (Numpy array/Python list) List of initial probabilities of market simulations.
        :param equality_matrix: (pd.DataFrame/Numpy matrix) An (N1 x J) matrix of equality constraints where N1 = number of equality views
                                                            and J = number of historical simulations. Denoted as 'H' in the "Meucci - Flexible
                                                            Views Theory & Practice" paper in formula 86 on page 22.
        :param equality_vector: (Numpy array/Python list) A vector of length J corresponding to the equality matrix. Denoted as 'h' in the "Meucci -
                                                          Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param inequality_matrix: (pd.DataFrame/Numpy matrix) An (N2 x J) matrix of inequality constraints where N2 = number of inequality
                                                              views and J = number of historical simulations. Denoted as 'F' in the "Meucci -
                                                              Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param inequality_vector: (Numpy array/Python list) A vector of length J corresponding to the inequality matrix. Denoted as 'f' in the "Meucci -
                                                            Flexible Views Theory & Practice" paper in formula 86 on page 22.
        """

        if not math.isclose(np.sum(prior_probabilities), 1.0):
            raise ValueError("Sum of prior probabilities is not 1.")

        if equality_matrix is None and inequality_matrix is None:
            raise ValueError("Please specify at least one equality or inequality.")

        if equality_matrix is not None and not equality_vector:
            raise ValueError("Please specify an equality vector with the equality constraint matrix.")

        if inequality_matrix is not None and not inequality_vector:
            raise ValueError("Please specify an inequality vector with the inequality constraint matrix.")

        if equality_matrix is not None and equality_matrix.shape[1] != len(equality_vector):
            raise ValueError("Number of rows in equality matrix and length of equality vector do not match.")

        if inequality_matrix is not None and inequality_matrix.shape[1] != len(inequality_vector):
            raise ValueError("Number of rows in inequality matrix and length of inequality vector do not match.")
