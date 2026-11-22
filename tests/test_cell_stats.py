"""CellStats.from_results gives each signal's stats as its docstring defines them."""

from types import SimpleNamespace

import numpy as np
import torch

from masknmf.demixing.cell_stats import CellStats


def test_from_results_matches_the_definitions_computed_in_float64():
    rng = np.random.default_rng(0)
    num_frames, num_signals = 500, 40
    c = rng.gamma(2.0, size=(num_frames, num_signals)).astype(np.float32)
    pmd = (c.T + rng.normal(scale=0.5, size=(num_signals, num_frames))).astype(np.float32)
    resid = rng.normal(scale=0.3, size=(num_signals, num_frames)).astype(np.float32)
    bkgd = rng.normal(scale=0.2, size=(num_signals, num_frames)).astype(np.float32)
    results = SimpleNamespace(
        temporal_demixed=torch.as_tensor(c),
        compression_array_roi_averages=torch.as_tensor(pmd),
        residual_roi_averages=torch.as_tensor(resid),
        fluctuating_background_roi_averages=torch.as_tensor(bkgd),
    )

    stats = CellStats.from_results(results)

    c, pmd, resid, bkgd = (a.astype(np.float64) for a in (c, pmd, resid, bkgd))
    std = c.std(0)
    expected = {
        "mean": c.mean(0),
        "std": std,
        "snr": c.max(0) / std,
        "skew": ((c - c.mean(0)) ** 3).mean(0) / std**3,
        "fit": np.array([np.corrcoef(c[:, i], pmd[i])[0, 1] for i in range(num_signals)]),
        "resid": resid.std(1) / pmd.std(1),
        "bkgd": bkgd.std(1) / pmd.std(1),
    }
    assert stats.names == tuple(expected)
    for index, (name, values) in enumerate(expected.items()):
        np.testing.assert_allclose(stats.values[:, index], values, rtol=1e-4, err_msg=name)
