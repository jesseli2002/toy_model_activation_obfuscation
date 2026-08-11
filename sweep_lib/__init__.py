"""Shared building blocks for the sweep*.py reporting entry points.

Each sweep gets its own entry point, since the question a sweep asks shapes
its plots and its per-run bookkeeping. What they share lives here: reading
training histories, recomputing and caching per-run metrics, sorting runs
into outcome bands, and drawing the handful of recurring figures.
"""
