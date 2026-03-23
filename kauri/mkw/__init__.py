"""
The ``kauri.mkw`` sub-package implements the Munthe-Kaas--Wright (MKW) ordered-tree
Hopf algebra operations for truncated EES verification.
"""

from .mkw import (counit, coproduct_terms, CoproductTerm,
                  sign_for_tree, planar_convolution, verify_mkw_ees)
