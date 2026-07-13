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
    """Rigid ICP (iterative closest point) - geometry only, no expression, no
    given correspondence. Iterates: match each moving spot to its nearest
    reference spot, solve the optimal rigid transform (Procrustes), repeat. The
    honest floor a real method must beat; it recovers a global rigid offset but
    cannot model tears/folds/tissue loss (no partial handling, no non-rigid)."""
    name = "rigid"

    def register(self, ref, mov, iters=40):
        from scipy.spatial import cKDTree
        t0 = time.time()
        r = np.asarray(ref.obsm["spatial"], float)
        cur = np.asarray(mov.obsm["spatial"], float).copy()
        try:
            tree = cKDTree(r)
            prev = np.inf
            for _ in range(iters):
                d, nn = tree.query(cur)
                tgt = r[nn]
                mc, mt = cur.mean(0), tgt.mean(0)
                H = (cur - mc).T @ (tgt - mt)
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt = Vt.copy(); Vt[-1] *= -1; R = Vt.T @ U.T
                cur = (cur - mc) @ R.T + mt
                err = float(d.mean())
                if abs(prev - err) < 1e-4 * (r.max() - r.min()):
                    break
                prev = err
            failed = not np.isfinite(cur).all()
            return dict(pred_xy=cur, runtime_s=time.time() - t0,
                        failed=failed, reason="non-finite" if failed else "")
        except Exception as e:
            return dict(pred_xy=np.full_like(cur, np.nan), runtime_s=time.time() - t0,
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
        import ot.optim
        t0 = time.time()
        try:
            # Use precomputed PCA features (obsm['X_pca']) + euclidean cost so the
            # OT is CPU-tractable at ~2000 spots (glmpca over all genes is ~18x
            # slower and crashes on this POT version).
            kw = {}
            if "X_pca" in ref.obsm and "X_pca" in mov.obsm:
                kw = dict(dissimilarity="euclidean", use_rep="X_pca")
            # POT>=0.9 compat: ot.optim.cg now calls line_search with a 6th
            # positional arg (df_G, the gradient at G) that paste-bio's 5-arg
            # closures don't accept -> "takes 5 positional arguments but 6 were
            # given". Wrap cg for the duration of this call to drop df_G (paste's
            # armijo/closed-form line searches don't use it). Scoped + restored so
            # paste2/gpsa/stalign are unaffected.
            _orig_cg = ot.optim.cg

            def _cg_compat(*a, **kk):
                a = list(a)
                if len(a) >= 8:
                    _ls = a[7]
                    a[7] = lambda cost, G, dG, Mi, cG, df_G=None, *r, **k: _ls(cost, G, dG, Mi, cG, **k)
                elif kk.get("line_search") is not None:
                    _ls = kk["line_search"]
                    kk["line_search"] = lambda cost, G, dG, Mi, cG, df_G=None, *r, **k: _ls(cost, G, dG, Mi, cG, **k)
                return _orig_cg(*a, **kk)

            ot.optim.cg = _cg_compat
            try:
                pi = paste.pairwise_align(ref, mov, alpha=alpha, **kw)
            finally:
                ot.optim.cg = _orig_cg
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
            kw = {}
            if "X_pca" in ref.obsm and "X_pca" in mov.obsm:
                kw = dict(dissimilarity="euclidean", use_rep="X_pca")
            pi = partial_pairwise_align(ref, mov, s=s, **kw)
            pred = barycentric(pi, np.asarray(ref.obsm["spatial"], float))
            failed = not np.isfinite(pred).any()
            return dict(pred_xy=pred, runtime_s=time.time() - t0,
                        failed=failed, reason="all-NaN plan" if failed else "",
                        s_used=s)
        except Exception as e:
            return dict(pred_xy=np.full((mov.n_obs, 2), np.nan),
                        runtime_s=time.time() - t0, failed=True,
                        reason=f"{type(e).__name__}: {e}", s_used=float(s))


class Paste2S1(Paste2):
    """PASTE2 with overlap forced to s=1 (no abstention): the no-abstention
    control reported alongside the heuristic-s PASTE2, so a reader can see the
    partial-overlap parameterisation is not hiding the result (see the
    abstention-bias analysis)."""
    name = "paste2_s1"

    def register(self, ref, mov, s=None):
        return super().register(ref, mov, s=1.0)


# --------------------------------------------------------------------------- #
# 4. STalign  (LDDMM image registration)
# --------------------------------------------------------------------------- #
class STalign(Method):
    """LDDMM diffeomorphic registration (STalign).

    EXPERIMENTAL / not in the default leaderboard. STalign installs and imports
    only on Python 3.11 with numpy<2 (its functional submodule uses the removed
    `numpy.bool8`); on Python 3.14 it does not build. The adapter below uses
    STalign's REAL API (rasterize -> LDDMM -> transform_points), verified against
    the installed package - it is NOT a fabricated call. However, across the
    hyperparameter/direction settings tried on this DLPFC data it did not
    converge to a registration that beats the no-op baseline on control
    transforms, and it is slow on CPU. Rather than publish mis-tuned numbers as a
    fair comparison (a wrong call), it is excluded from the leaderboard and this
    status is reported honestly. Kept runnable for future tuning.
    """
    name = "stalign"

    def available(self):
        try:
            from STalign import STalign as _S  # functional submodule (needs numpy<2)
            for fn in ("rasterize", "LDDMM", "transform_points_source_to_target"):
                if not hasattr(_S, fn):
                    return False, f"STalign.STalign missing {fn}"
            return True, "STalign importable (experimental; needs py3.11 + numpy<2)"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def register(self, ref, mov, niter=150, dx_pitch=1.3, a_pitch=2.0):
        import torch
        from STalign import STalign as _S
        t0 = time.time()
        try:
            r = np.asarray(ref.obsm["spatial"], float)
            m = np.asarray(mov.obsm["spatial"], float)
            from scipy.spatial import cKDTree
            pitch = float(np.median(cKDTree(r).query(r, k=2)[0][:, 1]))
            rn, mn = r / pitch, m / pitch                   # normalise to pitch units

            def rast(p):
                X0, X1, MM, _ = _S.rasterize(p[:, 0], p[:, 1], dx=dx_pitch)
                MM = np.asarray(MM)
                return [np.asarray(X1), np.asarray(X0)], MM / (MM.mean() + 1e-9)

            xI, I = rast(rn)                                # target = reference
            xJ, J = rast(mn)                                # source = moving
            out = _S.LDDMM(xI, I, xJ, J, niter=niter, device="cpu", a=a_pitch,
                           epV=5.0, epT=1e-2, epL=5e-5, sigmaM=0.3, sigmaR=5.0)
            pts = np.ascontiguousarray(mn[:, [1, 0]])       # (row,col)=(y,x)
            phi = _S.transform_points_source_to_target(
                out["xv"], out["v"], out["A"], torch.tensor(pts, dtype=torch.float64))
            pred = np.asarray(phi.detach())[:, [1, 0]] * pitch
            failed = not np.isfinite(pred).all()
            return dict(pred_xy=pred, runtime_s=time.time() - t0,
                        failed=failed, reason="non-finite" if failed else "experimental")
        except Exception as e:
            return dict(pred_xy=np.full((mov.n_obs, 2), np.nan),
                        runtime_s=time.time() - t0, failed=True,
                        reason=f"{type(e).__name__}: {e}")


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

    def register(self, ref, mov, m_inducing=100, epochs=200, lr=1e-2):
        # API mirrors ARCA's verified baselines/GPSA/run_gpsa_tear.py exactly
        # (torch-tensor data_dict; loss_fn(data_dict, F_samples)).
        import torch
        from gpsa import VariationalGPSA, rbf_kernel
        t0 = time.time()
        try:
            A_xy = np.asarray(ref.obsm["spatial"], float)
            B_xy = np.asarray(mov.obsm["spatial"], float)
            if "X_pca" in ref.obsm and "X_pca" in mov.obsm:
                A_f = np.asarray(ref.obsm["X_pca"])[:, :10]
                B_f = np.asarray(mov.obsm["X_pca"])[:, :10]
            else:
                A_f, B_f = _lognorm_pcs(ref, mov, k=10)
            nA, nB = A_xy.shape[0], B_xy.shape[0]
            x = torch.from_numpy(np.vstack([A_xy, B_xy]).astype(np.float32))
            y = torch.from_numpy(np.vstack([A_f, B_f]).astype(np.float32))
            data_dict = {"expression": {"spatial_coords": x, "outputs": y,
                                        "n_samples_list": [nA, nB]}}
            model = VariationalGPSA(
                data_dict, n_spatial_dims=2, m_X_per_view=m_inducing, m_G=m_inducing,
                data_init=True, minmax_init=False, grid_init=False,
                n_latent_gps={"expression": None}, mean_function="identity_fixed",
                kernel_func_warp=rbf_kernel, kernel_func_data=rbf_kernel, fixed_view_idx=0)
            view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(epochs):
                model.train()
                G_means, G_samples, F_latent_samples, F_samples = model.forward(
                    {"expression": x}, view_idx=view_idx, Ns=Ns, S=3)
                loss = model.loss_fn(data_dict, F_samples)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                G_means, _, _, _ = model.forward({"expression": x}, view_idx=view_idx, Ns=Ns)
            G = G_means["expression"] if isinstance(G_means, dict) else G_means
            pred = G.detach().cpu().numpy()[nA:]
            failed = not np.isfinite(pred).all()
            return dict(pred_xy=pred, runtime_s=time.time() - t0,
                        failed=failed, reason="non-finite" if failed else "")
        except Exception as e:
            return dict(pred_xy=np.full((mov.n_obs, 2), np.nan),
                        runtime_s=time.time() - t0, failed=True,
                        reason=f"{type(e).__name__}: {e}")


ALL_METHODS = [RigidAffine, Paste, Paste2, Paste2S1, STalign, GPSA]


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
