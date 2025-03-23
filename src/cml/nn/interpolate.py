"""Use a torch Module to interpolate values between sequence elements"""
import numpy as np
import torch


def _default_start_func(deltas):
    return np.floor(min(deltas))

def _default_until_func(deltas):
    return np.ceil(max(deltas))

def _default_point_func(start, until, num):
    return np.linspace(start, until, num=num, endpoint=True)

def _default_output_convert_func(tensor, output_dtype):
    return tensor.cpu().detach().to(output_dtype).numpy().squeeze()

def _default_tensor_convert_func(ndarr, model_dtype):
    return torch.from_numpy(ndarr).to(model_dtype)


def interpolate_meanvar_sequence(data, model, n_points=1000, from_prev=False,
                                 start_func=_default_start_func,
                                 until_func=_default_until_func,
                                 point_func=_default_point_func,
                                 model_dtype=torch.float32,
                                 output_dtype=torch.float64,
                                 output_convert=_default_output_convert_func,
                                 tensor_convert=_default_tensor_convert_func):
    """
    Arguments
    ---------
    data: list[(float, float, sequence)]
        An iterable of triplets of sequence position, target value, and
        prediction instance template. The triplets are assumed sorted by
        sequence position.
    model: torch.nn.Module
        A model which produces mean and variance estimates. The input to the
        model is assumed to be the concatenation of the prediction instance
        template with a float target (or "cuing") time.
    points: int = 1000
        The total number of points to interpolate from sequence start to
        sequence finish.
    from_prev: bool = False
        Given an interpolation target time of _t_, should we use the preceding
        or the following sequence element as the basis for interpolation?
    """
    deltas, values, bases = zip(*data)
    n_features = len(bases[0]) + 1

    # Generate the interpolations and their predicted variances
    start = start_func(deltas)
    until = until_func(deltas)
    points = point_func(start, until, num=n_points)
    instances = np.zeros((n_points, n_features))
    for i, dt in enumerate(points):
        if dt in deltas:
            ix = deltas.index(dt)
        elif from_prev:
            ix = np.searchsorted(deltas, dt)
            ix = max(ix-1, 0)
        else:
            ix = np.searchsorted(deltas, dt)
            ix = min(ix, len(deltas)-1)
            ix = max(ix, 0)
        instances[i, :-1] = bases[ix]
        instances[i, -1] = dt

    with torch.no_grad():
        M, V = model.eval()(tensor_convert(instances, model_dtype))
        interpolations = output_convert(M, output_dtype)
        interpolation_variances = output_convert(V, output_dtype)

    # Generate the estimates for the actual sequence
    instances = np.zeros((len(data), n_features))
    for i, (dt, _, base) in enumerate(data):
        instances[i, :-1] = base
        instances[i, -1] = dt

    with torch.no_grad():
        M, V = model.eval()(tensor_convert(instances, model_dtype))
        predictions = output_convert(M, output_dtype)
        prediction_variances = output_convert(V, output_dtype)

    return (deltas, values, bases, points,
            interpolations, interpolation_variances,
            predictions, prediction_variances)
