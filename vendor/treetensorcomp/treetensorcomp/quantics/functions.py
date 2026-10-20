import inspect
from collections.abc import Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import minimize

from treetensorcomp.helper.plotting import fig_to_base64
from treetensorcomp.quantics.index_settings import ExternalIndexSetting, IndexID


class QuanticsFunction:
    """
    A function f : R^n -> R or f : R^n -> C discretized over a domain (e.g. [0,1]^n).

    As a result of the discretized domain the final object corresponds to a function
    f : {0, 1}^b -> R  or f : {0, 1}^b -> C for b bits.

    """

    function: Callable
    variable_names: tuple[str, ...]
    variable_domains: dict[str, tuple]
    parameter_assignments: dict[str, float | NDArray]
    name: str
    latex: str | None
    global_max_precision: int

    def __init__(
        self,
        function: Callable,
        variable_names: Sequence[str],
        parameter_assignments: dict[str, float | NDArray],
        variable_domains: Sequence[tuple] | None = None,
        name: str = "f",
        latex: str | None = None,
        global_max_precision: int = 32,
        assert_safe_signature: bool = True,
    ) -> None:
        """
        Construct a QuanticsFunction object for a given function and configuration.

        Parameters
        ----------
        function : Callable
            Function to be wrapped for discretization. Must accept arguments matching
            "variable_names" and "parameter_assignments".
        variable_names : Sequence[str]
           String identifiers for each of the variables the function depends on.
           Identifiers must not contain underscores ('_').
        parameter_assignments : dict[str, float | NDArray]
            Mapping between parameter identifiers and values. Identifiers must not
            overlap with "variable_names".
        variable_domains : Sequence[tuple], optional
            Numerical domain for each variable the function depends on (default:
            (0, 1)).
        name : str, optional
            A string identifier for the function (default: "f").
        latex : str, optional
            A LaTeX representation of the function (for display and prototyping).
        global_max_precision : int, optional
            Maximum precision (in bits) to use for internal calculations (default: 32).
        assert_safe_signature : bool, optional
            Whether to validate if the function signature matches exactly the union of
            "variable_names" and "parameter_assignments" (default: True).

        """
        # Check the function is actually a function
        if not callable(function):
            msg = "function must be callable (i.e. a function)"
            raise TypeError(msg)

        # Check that the variable ids are a list or tuple
        if not isinstance(variable_names, (list, tuple)):
            msg = "variable_names must be a list or tuple"
            raise TypeError(msg)

        # Check that the variable ids are strings
        if not all(isinstance(var_name, str) for var_name in variable_names):
            msg = "All elements of variable_names must be strings"
            raise TypeError(msg)

        # Check no underscores in variable_names (we use these to separate significant-bit counter)
        for var_name in variable_names:
            if "_" in var_name:
                msg = "variable_names must not contain underscores"
                raise ValueError(msg)

        # Check that the parameter values are a dict (for **kwargs use)
        if not isinstance(parameter_assignments, dict):
            msg = "parameter_assignments must be a dict"
            raise TypeError(msg)

        # Check that the parameter keys are strings
        if not all(isinstance(key, str) for key in parameter_assignments):
            msg = "All keys of parameter_assignments must be strings"
            raise TypeError(msg)

        # Check that there are no common elements between variable_names and parameter_assignments.keys()
        common_elements = set(variable_names) & set(parameter_assignments.keys())
        if common_elements:
            msg = f"variable_names and parameter_assignments.keys() must not have common elements, but found: {common_elements}"
            raise ValueError(msg)

        if assert_safe_signature:
            # Check correct arguments/signature
            sig = inspect.signature(function)
            assert all(
                param.kind != inspect.Parameter.POSITIONAL_ONLY
                for param in sig.parameters.values()
            ), "Positional-only arguments in the function are not supported."

            # Check that there's an exact match between function arguments and
            # variable_names + parameter_assignments (i.e. no arguments missed or extra
            # arguments unaccounted for)
            assert (set(variable_names) | set(parameter_assignments.keys())) == set(
                sig.parameters.keys()
            ), (
                "variable_names and parameter_assignments must exactly match the function's arguments"
            )

        # Now store the attributes
        self.function = function
        self.variable_names = tuple(variable_names)
        self.parameter_assignments = parameter_assignments
        self.name = name
        self.latex = latex
        self.global_max_precision = global_max_precision

        # Precompute place values for faster evaluation of index settings
        self._place_values = 2 ** np.arange(1, global_max_precision + 1, dtype=float)

        self.variable_domains = {}
        if variable_domains is None:
            for var_name in variable_names:
                self.variable_domains[var_name] = (0, 1)

        else:
            # Check that the variable names and domains have the same length
            assert len(variable_names) == len(variable_domains), (
                "variable_names and variable_domains must have the same length"
            )

            for var_name, domain in zip(variable_names, variable_domains):
                assert isinstance(domain, tuple) and len(domain) == 2, (
                    "Each domain must be a tuple of (min, max)"
                )
                assert domain[0] < domain[1], "Each domain must have min < max"
                self.variable_domains[var_name] = domain

        # TODO: Somehow check it is vectorized in the variables (numpy).
        # e.g. maybe call it for ndarray values and check for any exceptions thrown.
        # Also check that input shape = output shape.

    def _repr_html_(self) -> str:
        """Construct HTML representation for Jupyter notebooks."""
        unique_id = f"quantics-func-{id(self)}"

        html = "<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'></script>"

        html += f"""
        <div id="{unique_id}" style="border: 1px solid #ccc; padding: 10px; margin: 5px;">
            <strong>QuanticsFunction</strong><br>
            <strong>Name:</strong> {self.name}<br>
        """

        if self.latex is not None:
            html += f"<strong>Formula:</strong> $$ {self.latex} $$ <br>"

        if len(self.variable_names) == 1:
            fig, evaluation_kwargs = self.visualize_1D(self.variable_names[0])
        else:
            fig, evaluation_kwargs = self.visualize_2D(self.variable_names[:2])
        plot_data = fig_to_base64(fig)

        fixed_variable_string = ", ".join(
            f"{k}={v}" for k, v in evaluation_kwargs.items()
        )

        # TODO add domains to this
        html += f"""
            <strong>Variables:</strong> {", ".join(self.variable_names)}<br>
            <strong>Parameters:</strong> {", ".join(f"{k}={v}" for k, v in self.parameter_assignments.items())}<br>
            <strong>Preview:</strong> ({fixed_variable_string})<br><img src="data:image/png;base64,{plot_data}" style="max-width:100%;">
        """

        html += f"""
        <script>
            (function() {{
                if (typeof MathJax !== 'undefined') {{
                    // For MathJax v3
                    if (MathJax.typeset) {{
                        MathJax.typeset([document.getElementById('{unique_id}')]);
                    }}
                    // For MathJax v2
                    else if (MathJax.Hub) {{
                        MathJax.Hub.Queue(["Typeset", MathJax.Hub, "{unique_id}"]);
                    }}
                    // If neither method worked, try a small delay
                    else {{
                        setTimeout(function() {{
                            if (MathJax.typeset) {{
                                MathJax.typeset([document.getElementById('{unique_id}')]);
                            }} else if (MathJax.Hub) {{
                                MathJax.Hub.Queue(["Typeset", MathJax.Hub, "{unique_id}"]);
                            }}
                        }}, 500);
                    }}
                }}
            }})();
        </script>
        """

        return html

    def evaluate_numerical(
        self,
        variable_settings: dict[str, float],
        check_var_ids: bool = True,
        ) -> NDArray[np.float64]:
        """
        Evaluate the QuanticsFunction for a given set of variable inputs.

        Parameters
        ----------
        variable_settings : dict[str, float]
            Mapping between variable identifiers and values for the function evaluation.
        check_var_ids : bool, optional
            Whether to validate if the identifiers in "variable_values" match the
            QuanticsFunction's "variable_names" (default: True).

        Returns
        -------
        NDArray[np.float64]
            Result of the function evaluation.

        """
        if not isinstance(variable_settings, dict):
            msg = "variable_values must be a dict"
            raise TypeError(msg)

        if check_var_ids:
            assert set(variable_settings.keys()) == set(self.variable_names), (
                "Keys of variable_values must exactly match self.variable_names"
            )

        return self.function(**variable_settings, **self.parameter_assignments)

    # TODO(Matt): migrate this function to the ExternalIndexSetting class and make this a
    # wrapper instead
    def index_setting_to_variable_setting(
        self,
        index_setting: ExternalIndexSetting,
        check_var_ids: bool = True,
    ) -> dict[str, float]:
        """
        Convert an ExternalIndexSetting into variable settings for QuanticsFunction.

        Parameters
        ----------
        index_setting : ExternalIndexSetting
            The ExternalIndexSetting with the settings to convert.
        check_var_ids : bool, optional
            Whether to validate if the variable names from the ExternalIndexSetting and
            the QuanticsFunction match (default: True).

        Returns
        -------
        dict[str, float]
            Mapping between variable identifiers and values usable for QuanticsFunction
            evaluation.

        """
        if not isinstance(index_setting, ExternalIndexSetting):
            msg = "index_setting must be an ExternalIndexSetting"
            raise TypeError(msg)

        # Check that function variable ids and index setting variable names contain the
        # same elements (order-insensitive)
        if check_var_ids and set(index_setting.variable_names()) != set(
            self.variable_names
        ):
            msg = "Mismatch between function args and index setting variable names."
            raise ValueError(msg)

        # Now batch evaluate the variable values
        variable_values = {}
        for variable_name in self.variable_names:
            max_bit_position = index_setting.max_bit_position(variable_name)
            assert max_bit_position is not None  # TODO: check

            # Check that the setting doesn't exceed the global max precision for this
            # function, and that all consecutive bits are present
            assert max_bit_position <= self.global_max_precision, (
                f"Index setting for variable '{variable_name}' exceeds global max precision of {self.global_max_precision} bits."
            )

            assert index_setting.has_consecutive_bits(variable_name), (
                f"Index setting for variable '{variable_name}' does not have consecutive bits starting from 1."
            )

            # Create a B x N array for the bit values, where B is the bits of precision
            # for this variable and N is the number of settings
            bit_values = np.empty(
                (max_bit_position, index_setting.setting_count), dtype=float
            )

            # N.B.: bits are 1-indexed, idx is 0-indexed
            for idx in range(max_bit_position):
                bit_values[idx, :] = index_setting[IndexID(variable_name, idx + 1)]

            # Quantics index setting (Eq. 2, arXiv:2303.11819) using numpy broadcasting
            variable_values[variable_name] = np.sum(
                bit_values / self._place_values[:max_bit_position, None],
                axis=0,
            )

            # Rescale to the variable domain
            variable_values[variable_name] = (
                self.variable_domains[variable_name][1]
                - self.variable_domains[variable_name][0]
            ) * variable_values[variable_name] + self.variable_domains[variable_name][0]

        return variable_values

    def evaluate_index_setting(
        self,
        index_setting: ExternalIndexSetting,
        check_var_ids: bool = True,
    ) -> NDArray[np.float64]:
        """
        Evaluate the QuanticsFunction for a given ExternalIndexSetting.

        Convenience wrapper around "QuanticsFunction.index_setting_to_variable_setting"
        and "QuanticsFunction.evaluate_numerical".

        Parameters
        ----------
        index_setting : ExternalIndexSetting
            The ExternalIndexSetting with the settings to convert.
        check_var_ids : bool, optional
            Whether to validate if the variable names from the ExternalIndexSetting and
            the QuanticsFunction match (default: True).

        Returns
        -------
        NDArray[np.float64]
            Result of the function evaluation with the provided settings.

        """
        if not isinstance(index_setting, ExternalIndexSetting):
            msg = "index_setting must be an ExternalIndexSetting"
            raise TypeError(msg)

        return self.evaluate_numerical(
            self.index_setting_to_variable_setting(
                index_setting, check_var_ids=check_var_ids
            )
        )

    def visualize_1D(
        self,
        independent_variable: str,
        variable_settings: dict = {},
        resolution: int = 200,
    ) -> tuple[Figure, dict]:
        """
        Plot the effect of varying a single variable on the result of the function.

        Generates a line plot by varying the given variable while keeping the rest
        fixed.

        For complex-valued functions the figure contains separate subplots for real and
        imaginary parts.

        Parameters
        ----------
        independent_variable : str
            Identifier of the variable to vary along the x-axis. Must be one of the
            function's variable names.
        variable_settings : dict, optional
            Mapping from identifier to numerical values for the remaining, fixed,
            variables. Unspecified variables are assigned random values in [0, 1).
        resolution : int, optional
            Number of points to evaluate along the independent variable's range
            (default: 200).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure containing the plot(s).
        evaluation_kwargs : dict
            Mapping from identifiers to numerical values used for fixed variables.

        """
        assert independent_variable in self.variable_names, (
            "independent_variable must be one of the function's variable names"
        )

        assert isinstance(variable_settings, dict), "variable_settings must be a dict"
        assert independent_variable not in variable_settings, (
            "independent_variable must not be in variable_settings"
        )

        assert all(
            var_name in self.variable_names for var_name in variable_settings.keys()
        ), "All keys of variable_settings must be function variable names"

        evaluation_kwargs = variable_settings.copy()

        # randomize in domain the variable settings that are not specified
        for var_name in self.variable_names:
            if var_name not in variable_settings:
                evaluation_kwargs[var_name] = np.random.uniform(
                    low=self.variable_domains[var_name][0],
                    high=self.variable_domains[var_name][1],
                )

        x_vals = np.linspace(
            self.variable_domains[independent_variable][0],
            self.variable_domains[independent_variable][1],
            resolution,
        )
        evaluation_kwargs[independent_variable] = x_vals
        y_vals = self.evaluate_numerical(evaluation_kwargs)

        # complex, plot real and imaginary parts separately
        if np.iscomplexobj(y_vals):
            fig, (ax_re, ax_im) = plt.subplots(1, 2, sharex=True, figsize=(6, 3))
            ax_re.plot(x_vals, y_vals.real, color="C0")
            ax_im.plot(x_vals, y_vals.imag, color="C1")
            ax_re.set_ylabel(f"({self.name}({independent_variable}))")
            ax_re.set_title(f"Re({self.name})")
            ax_im.set_title(f"Im({self.name})")
            ax_re.set_xlabel(independent_variable)
            ax_im.set_xlabel(independent_variable)
            ax_re.set_xlim(0, 1)  # TODO update this to variable domain
            ax_re.set_box_aspect(1)
            ax_im.set_box_aspect(1)

        # real, plot function values
        else:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.plot(x_vals, y_vals)
            ax.set_xlabel(independent_variable)
            ax.set_ylabel(f"{self.name}({independent_variable})")
            ax.set_xlim(0, 1)  # TODO update this to variable domain
            ax.set_box_aspect(1)

        # Close the figure to avoid displaying it immediately in Jupyter notebooks
        plt.close(fig)

        # Clean up the kwargs so we can use them in return
        del evaluation_kwargs[independent_variable]

        return fig, evaluation_kwargs

    def visualize_2D(
        self,
        independent_variables: Sequence[str],
        variable_settings: dict = {},
        resolution: int = 30,
    ) -> tuple[Figure, dict]:
        """
        Plot the effect of varying two variable on the result of the function.

        Generates a surface plot by varying the two variables while keeping the rest
        fixed.

        For complex-valued functions the figure contains separate subplots for real and
        imaginary parts.

        Parameters
        ----------
        independent_variables : Sequence[str]
            Identifiers of the variables to vary. Must be distinct and present in the
            function's variable names.
        variable_settings : dict, optional
            Mapping from identifier to numerical values for the remaining, fixed,
            variables. Unspecified variables are assigned random values in [0, 1).
        resolution : int, optional
            Number of points to evaluate along the independent variables' ranges
            (default: 30).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure containing the plot(s).
        evaluation_kwargs : dict
            Mapping from identifiers to numerical values used for fixed variables.

        """
        assert (
            isinstance(independent_variables, (list, tuple))
            and len(independent_variables) == 2
        ), "independent_variables must be a list or tuple of exactly 2 variable names"
        assert all(var in self.variable_names for var in independent_variables), (
            "Both independent_variables must be function variable names"
        )
        assert len(set(independent_variables)) == 2, (
            "independent_variables must be distinct"
        )
        assert isinstance(variable_settings, dict), "variable_settings must be a dict"
        assert all(var not in variable_settings for var in independent_variables), (
            "independent_variables must not be in variable_settings"
        )
        assert all(
            var_name in self.variable_names for var_name in variable_settings.keys()
        ), "All keys of variable_settings must be function variable names"
        assert isinstance(resolution, int) and resolution > 0, (
            "resolution must be a positive integer"
        )

        evaluation_kwargs = variable_settings.copy()

        # randomize in domain the variable settings that are not specified
        for var_name in self.variable_names:
            if (
                var_name not in variable_settings
                and var_name not in independent_variables
            ):
                evaluation_kwargs[var_name] = np.random.uniform(
                    low=self.variable_domains[var_name][0],
                    high=self.variable_domains[var_name][1],
                )

        # Create meshgrid for the two independent variables
        x_vals = np.linspace(
            self.variable_domains[independent_variables[0]][0],
            self.variable_domains[independent_variables[0]][1],
            resolution,
        )
        y_vals = np.linspace(
            self.variable_domains[independent_variables[1]][0],
            self.variable_domains[independent_variables[1]][1],
            resolution,
        )
        X, Y = np.meshgrid(x_vals, y_vals)

        # Flatten for evaluation
        evaluation_kwargs[independent_variables[0]] = X.flatten()
        evaluation_kwargs[independent_variables[1]] = Y.flatten()

        # Evaluate function and reshape to grid
        z_vals = self.evaluate_numerical(evaluation_kwargs)
        Z = z_vals.reshape(X.shape)

        # complex, plot real and imaginary parts separately
        if np.iscomplexobj(Z):
            fig = plt.figure(figsize=(6, 3))

            # Real part
            ax_re = fig.add_subplot(121, projection="3d")
            ax_re.plot_surface(X, Y, Z.real, cmap="viridis", alpha=0.8)
            ax_re.set_title(f"Re({self.name})")
            ax_re.set_xlabel(independent_variables[0])
            ax_re.set_ylabel(independent_variables[1])

            # Imaginary part
            ax_im = fig.add_subplot(122, projection="3d")
            ax_im.plot_surface(X, Y, Z.imag, cmap="plasma", alpha=0.8)
            ax_im.set_title(f"Im({self.name})")
            ax_im.set_xlabel(independent_variables[0])
            ax_im.set_ylabel(independent_variables[1])

            plt.tight_layout()
        # real, plot function values
        else:
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
            ax.set_xlabel(independent_variables[0])
            ax.set_ylabel(independent_variables[1])
            ax.set_title(
                f"{self.name}({independent_variables[0]}, {independent_variables[1]})"
            )

        # Close the figure to avoid displaying it immediately in Jupyter notebooks
        plt.close(fig)

        # Clean up the kwargs so we can use them in return
        del evaluation_kwargs[independent_variables[0]]
        del evaluation_kwargs[independent_variables[1]]

        return fig, evaluation_kwargs

    def find_maximum_magnitude(self, n_restarts: int = 5) -> dict[str, tuple]:
        """
        Find variable values that (approximately) maximize the function.

        Used to determine a good initial pivot for TCI.

        Parameters
        ----------
        n_restarts : int, optional
            Number of random restarts to attempt to avoid local maxima (default: 5).

        Returns
        -------
        dict[str, tuple]
            Mapping from variable identifiers to the optimal values found.

        """
        # Extract domains and create bounds for the optimizer
        bounds = [self.variable_domains[var] for var in self.variable_names]

        # Function to minimize (negative squared magnitude)
        def objective(x: NDArray[np.float64]) -> float:

            # Map flat array back to dictionary of variable values
            var_dict = dict(zip(self.variable_names, x))
            result = self.evaluate_numerical(var_dict)

            # Negative because we're minimizing, abs^2 for stability
            return -(np.abs(result) ** 2)

        # Try multiple random starting points to avoid local minima
        best_x = None
        best_val = -np.inf

        for _ in range(n_restarts):
            # Random starting point within domains
            x0 = [np.random.uniform(low, high) for low, high in bounds]

            # Run optimization
            result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            if result.success and -result.fun > best_val:
                best_val = -result.fun
                best_x = result.x

        if best_x is None:
            msg = "Optimization failed to find a valid maximum"
            raise RuntimeError(msg)

        # Format result as dictionary
        return dict(zip(self.variable_names, best_x))
