"""Compatibility boundary for the proven project pipeline.

New integrations should use :mod:`sp_patch_toolbox.pipeline` and profiles;
this module keeps exact behaviour of the validated legacy implementation while
the migration is completed in small, testable steps.
"""
