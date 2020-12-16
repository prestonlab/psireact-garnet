"""Reponse time modeling for the Garnet study."""

import numpy as np
import pandas as pd
import theano.tensor as tt
from psireact import lba


def read_data(csv_file):
    """Read response time data."""
    data = pd.read_csv(csv_file)
    data['subj_idx'] = data['subject'] - 1
    data['test'] = data['phase'].map({1: 'BC', 2: 'AC'}).astype('category')
    data['response'] = data['accuracy']
    data.test.cat.reorder_categories(['BC', 'AC'], ordered=True, inplace=True)
    return data


def tpdf31(t, i, n, A, b, v1, v2, s):
    """Probability distribution function for 3afc tests."""
    # probability that all accumulators are negative
    all_neg = (
        lba.normcdf(-v1 / s) * lba.normcdf(-v2 / s) * lba.normcdf(-v2 / s)
    )

    # PDF for each accumulator
    p1 = lba.tpdf(t, A, b, v1, s)
    p2 = lba.tpdf(t, A, b, v2, s)

    # probability of having not hit threshold by now
    n1 = 1 - lba.tcdf(t, A, b, v1, s)
    n2 = 1 - lba.tcdf(t, A, b, v2, s)

    # conditional probability of each accumulator hitting threshold now
    c1 = p1 * n2 * n2
    c2 = p2 * n1 * n2
    c3 = p2 * n1 * n2

    # calculate probability of this response and rt,
    # conditional on a valid response
    pdf = tt.switch(
        tt.eq(i, 1), c1 / (1 - all_neg), (c2 + c3) / (1 - all_neg)
    )
    pdf_cond = tt.switch(tt.gt(t, 0), pdf, 0)
    return pdf_cond


def tpdf31_rvs(n, A, b, v1, v2, s, tau, size=1):
    """Random generator for 3afc tests."""

    # finish times of accumulators
    rt, resp = lba.sample_response(A, b, [v1, v2, v2], s, tau, size)

    # accumulator 1 indicates correct response
    correct = np.zeros(size)
    correct[resp == 0] = 1
    return rt, correct


def summarize_trace_stats(stats):
    """Summarize trace statistics like ess and rhat."""
    all_var = []
    all_min = []
    all_max = []
    all_med = []
    for key, val in stats.items():
        all_var.append(key)
        if val.size == 1:
            all_min.append(val.values)
            all_max.append(val.values)
            all_med.append(val.values)
        else:
            all_min.append(val.min().values)
            all_max.append(val.max().values)
            all_med.append(val.median().values)
    df = pd.DataFrame(
        {'var': all_var, 'min': all_min, 'med': all_med, 'max': all_max}
    )
    return df
