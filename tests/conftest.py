"""Shared pytest setup: use the Agg backend so plots render off-screen."""

import matplotlib

matplotlib.use("Agg")
