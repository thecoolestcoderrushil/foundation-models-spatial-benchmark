"""
Registration methods under test.

Each method exposes a uniform interface:

    m = Method()
    ok, detail = m.available()          # honest install/import probe
    out = m.register(ref, mov)          # mov.obsm['spatial'] = DAMAGED coords
    # out = {"pred_xy": (n_mov,2), "runtime_s": float, "failed": bool, "reason": str}

`register` predicts, for every moving spot, its coordinate in the reference
frame. Ground truth (array bridge, in datasets.py) is used only for scoring and
is never passed to a method.

APIs are taken from the actually-installed packages (PASTE `pairwise_align`,
PASTE2 `partial_pairwise_align`, GPSA `VariationalGPSA` as used in ARCA's
verified `run_gpsa_tear.py`). Methods that will not install in this environment
(STalign on Python 3.14; GPSA unless locally installed) report the exact error
and are marked absent - never stubbed with a fabricated call.
"""
from __future__ import annotations

import time
import traceback

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def barycentric(pi, ref_xy):
    """Map each moving spot into the reference frame via a transport plan.
    pi has shape (n_ref, n_mov). Returns pred_xy (n_mov,2); zero-mass columns NaN."""
    pi = np.asarray(pi, float)
    ref_xy = np.asarray(ref_xy, float)
    col = pi.sum(0)
    safe = col > 0
    pred = np.full((pi.shape[1], 2), np.nan)
    pred[safe] = (pi[:, safe].T @ ref_xy) / col[safe, None]
    return pred


def _lognorm_pcs(A, B, k=15):
    """Shared PCA features on common genes (for the geometry-only GPSA feature)."""
    from scipy.sparse import issparse
    from sklearn.decomposition import TruncatedSVD
    genes = np.intersect1d(np.asarray(A.var_names), np.asarray(B.var_names))
    def ln(ad_):
        X = ad_[:, genes].X
        X = np.asarray(X.todense() if issparse(X) else X, float)
        c = X.sum(1, keepdims=True); c[c == 0] = 1
        return np.log1p(X * 1e4 / c)
    Xa, Xb = ln(A), ln(B)
    Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(np.vstack([Xa, Xb]))
    return Z[:A.n_obs], Z[A.n_obs:]


class Method:
    name = "base"
    def available(self):
        return True, ""
    def register(self, ref, mov):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. rigid/affine floor - parameter-free, expression-blind
# --------------------------------------------------------------------------- #
class RigidAffine(Method):
    """Similarity transform (rotation + isotropic scale + translation) estimated
    from the two point clouds' moments alone - no expression, no correspondence.
    The floor a real method must beat; it cannot model tears/folds at all."""
    name = "rigid"

    def register(self, ref, mov):
        t0 = time.time()
        r = np.asarray(ref.obsm["spatial"], float)
        m = np.asarray(mov.obsm["spatial"], float)
        try:
            rc, mc = r.mean(0), m.mean(0)
            Rr, Mm = r - rc, m - mc
            # principal axes
            Ur = np.linalg.svd(Rr, full_matrices=False)[2]
            Um = np.linalg.svd(Mm, full_matrices=False)[2]
            R = Ur.T @ Um                      # rotate mov axes onto ref axes
            if np.linalg.det(R) < 0:           # forbid reflection
                Ur2 = Ur.copy(); Ur2[-1] *= -1; R = Ur2.T @ Um
            scale = (np.linalg.norm(Rr) / (np.linalg.norm(Mm) + 1e-9))
            pred = (m - mc) @ R.T * scale + rc
            failed = not np.isfinite(pred).all()
            return dict(pred_xy=pred, runtime_s=time.time() - t0,
                        failed=failed, reason="non-finite" if failed else "")
        except Exception as e:
            return dict(pred_xy=np.full_like(m, np.nan), runtime_s=time.time() - t0,
                        failed=True, reason=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 2. PASTE (paste-bio)
# --------------------------------------------------------------------------- #
class Paste(Method):
    name = "paste"

    def available(self):
        try:
            from paste import pairwise_align  # noqa: F401
            return True, "paste-bio importable"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def register(self, ref, mov, alpha=0.1):
        import paste
        t0 = time.time()
        try:
            pi = paste.pairwise_align(ref, mov, alpha=alpha)
            pred = barycentric(pi, np.asarray(ref.obsm["spatial"], float))
            failed = not np.isfinite(pred).any()
            return dict(pred_xy=pred, runtime_s=time.time() - t0,
                        failed=failed, reason="all-NaN plan" if failed else "")
        except Exception as e:
            return dict(pred_xy=np.full((mov.n_obs, 2), np.nan),
                        runtime_s=time.time() - t0, failed=True,
                        reason=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 3. PASTE2 (partial OT - built for partial overlap / tissue loss)
# --------------------------------------------------------------------------- #
class Paste2(Method):
    name = "paste2"

    def available(self):
        try:
            from paste2.PASTE2 import partial_pairwise_align  # noqa: F401
            return True, "paste2 importable"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def register(self, ref, mov, s=None):
        from paste2.PASTE2 import partial_pairwise_align
        t0 = time.time()
        try:
            # overlap fraction s: fraction of mov spots that still overlap ref.
            if s is None:
                s = float(min(1.0, mov.n_obs / max(ref.n_obs, 1)))
            s = float(np.clip(s, 0.1, 1.0))
            pi = partial_pairwise_align(ref, mov, s=s)
            pred = barycentric(pi, np.asarray(ref.obsm["spatial"], float))
            failed = not np.isfinite(pred).any()
            return dict(pred_xy=pred, runtime_s=time.time() - t0,
                        failed=failed, reason="all-NaN plan" if failed else "")
        except Exception as e:
            return dict(pred_xy=np.full((mov.n_obs, 2), np.nan),
                        runtime_s=time.time() - t0, failed=True,
                        reason=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 4. STalign  (LDDMM image registration)
# --------------------------------------------------------------------------- #
class STalign(Method):
    name = "stalign"

    def available(self):
        try:
            import STalign  # noqa: F401
            return True, "STalign importable"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def register(self, ref, mov):
        # Verified absent in this environment (STalign fails to build its pinned
        # numpy on Python 3.14). We refuse to fabricate its LDDMM API; the harness
        # records this method as absent from availability() above.
        raise RuntimeError("STalign unavailable in this environment")


# --------------------------------------------------------------------------- #
# 5. GPSA (Gaussian-Process Spatial Alignment)
# --------------------------------------------------------------------------- #
class GPSA(Method):
    name = "gpsa"

    def available(self):
        try:
            import gpsa  # noqa: F401
            import torch  # noqa: F401
            return True, "gpsa importable"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def register(self, ref, mov, m_inducing=100, epochs=400, lr=1e-2):
        # API mirrors ARCA's verified baselines/GPSA/run_gpsa_tear.py.
        import torch
        from gpsa import VariationalGPSA, rbf_kernel
        t0 = time.time()
        try:
            A_xy = np.asarray(ref.obsm["spatial"], float)
            B_xy = np.asarray(mov.obsm["spatial"], float)
            A_f, B_f = _lognorm_pcs(ref, mov, k=10)
            X = np.vstack([A_xy, B_xy]).astype(np.float32)
            Y = np.vstack([A_f, B_f]).astype(np.float32)
            n_views = 2
            view_idx = [np.arange(A_xy.shape[0]),
                        np.arange(A_xy.shape[0], A_xy.shape[0] + B_xy.shape[0])]
            n_samples_list = [A_xy.shape[0], B_xy.shape[0]]
            data_dict = {"expression": {
                "spatial_coords": X, "outputs": Y,
                "n_samples_list": n_samples_list}}
            model = VariationalGPSA(
                data_dict, n_spatial_dims=2, m_X_per_view=m_inducing,
                m_G=m_inducing, data_init=True, minmax_init=False,
                grid_init=False, n_latent_gps={"expression": None},
                mean_function="identity_fixed", kernel_func_warp=rbf_kernel,
                kernel_func_data=rbf_kernel, fixed_view_idx=0)
            view_idx_d, Ns, _, _ = model.create_view_idx_dict(data_dict)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            x = torch.from_numpy(X)
            for _ in range(epochs):
                opt.zero_grad()
                G_means, G_samples, F_latent_samples, F_samples = model.forward(
                    {"expression": x}, view_idx=view_idx_d, Ns=Ns, S=3)
                loss = model.loss_fn({"expression": x}, F_samples,
                                     {"expression": torch.from_numpy(Y)})
                loss.backward(); opt.step()
            with torch.no_grad():
                G_means, *_ = model.forward({"expression": x}, view_idx=view_idx_d, Ns=Ns)
            pred = G_means["expression"].detach().numpy()[view_idx[1]]
            failed = not np.isfinite(pred).all()
            return dict(pred_xy=pred, runtime_s=time.time() - t0,
                        failed=failed, reason="non-finite" if failed else "")
        except Exception as e:
            return dict(pred_xy=np.full((mov.n_obs, 2), np.nan),
                        runtime_s=time.time() - t0, failed=True,
                        reason=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}")


ALL_METHODS = [RigidAffine, Paste, Paste2, STalign, GPSA]


def probe_all():
    """Availability of every method (honest install/import check)."""
    out = []
    for cls in ALL_METHODS:
        m = cls()
        ok, detail = m.available()
        out.append(dict(method=m.name, available=ok, detail=detail))
    return out


if __name__ == "__main__":
    from pathlib import Path
    rows = probe_all()
    lines = ["# Registration methods - environment availability\n",
             "_Host: Windows, CPU-only, Python 3.14 main env._\n",
             "| method | available | detail |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['method']}` | {r['available']} | {r['detail']} |")
    lines.append("\n- `rigid` is implemented in-repo (no dependency), the "
                 "parameter-free floor.\n- Absent methods are recorded here and "
                 "skipped by the harness; their APIs are never stubbed.")
    p = Path(__file__).resolve().parent.parent / "results" / "methods_env.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    for r in rows:
        print(f"{r['method']:10s} available={r['available']}  {r['detail'][:80]}")
    print("wrote", p)
