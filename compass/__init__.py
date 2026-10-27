"""COMPASS: resolution theory, sequence-resolved fine-mapping and conformal credible sets."""
from . import theory, srfm, conformal, metrics  # noqa: F401
from .theory import *  # noqa: F401,F403
from .srfm import sequence_score, fit_lambda, srfm_posterior, standardize  # noqa: F401
from .conformal import conformal_threshold, conformal_sets, evaluate_cross_fit  # noqa: F401
from .metrics import within_set_auc, top1_accuracy, credible_set, ece  # noqa: F401

__version__ = "1.0.0"
