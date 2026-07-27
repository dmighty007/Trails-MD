# Overall assessment

Trails-MD is structurally well suited for adding committor analysis, but its **exploration trajectories must not be treated directly as an unbiased kinetic ensemble**. The repository itself makes this distinction: exploration mode redraws velocities and adaptively selects restart points, whereas kinetics mode preserves velocities and weighted-ensemble probabilities. ([GitHub][1])

The strongest route is therefore:

1. Use Trails-MD to discover basins, transition tubes, and candidate configurations.
2. Launch separate, unbiased local shooting simulations from selected configurations.
3. Estimate local transition probabilities or direct endpoint commitments.
4. Assemble these local estimates into a global discrete committor.
5. Validate the assembled committor against brute-force shooting on a two-dimensional toy potential.

I would call this a **local-shooting, globally stitched committor protocol**, rather than simply “piecewise committor”.

---

# 1. Strength of the current codebase

## Strong points

### A. Clean separation of sampling components

The central controller separates:

- simulation engines,
- feature/CV extraction,
- spawning,
- adaptive latent-space fitting,
- MSM estimation,
- path reconstruction,
- checkpoints,
- execution backends.

This is the right architecture for committor work because the analysis can be introduced without rewriting the adaptive-sampling controller. The engine, spawner and space abstractions are already constructed through factories and registries.

### B. Per-walker trajectory segmentation is retained

The MSM estimator accepts a **list of individual continuous trajectories**, rather than concatenating all frames into one artificial trajectory. It then performs clustering, sliding-window counting, restriction to the largest connected component and MSM estimation.

That is essential. An adaptive campaign contains discontinuities between iterations, and those discontinuities must never be interpreted as dynamical transitions.

### C. Lineage metadata is unusually useful

Every saved frame has:

- iteration,
- walker,
- trajectory,
- frame number,
- CV coordinates,
- parent frame.

The path module reconstructs branches by tracing ancestry to a common root.

This is a substantial strength for committor analysis because it provides candidate transition tubes and allows you to select shooting structures along a connected history rather than merely selecting geometrically close endpoints.

### D. Existing MSM infrastructure supplies most of the required machinery

The current estimator already produces:

- transition matrix,
- stationary distribution,
- count matrix,
- active-state symbols,
- cluster centres,
- slow eigenvectors,
- PCCA metastable assignments,
- implied timescales,
- Bayesian timescale uncertainty.

A discrete committor solver can therefore be added with relatively little code.

### E. Attention to kinetic correctness in weighted ensemble

The code explicitly handles:

- velocity inheritance,
- walker failure mappings,
- source-to-sink recycling,
- persistent source structures,
- inconsistency detection after restart,
- endpoint-state persistence.

The comments around recycling show that the authors understand how an apparently minor restart error can bias $1/\text{flux}$ kinetics.

### F. Test coverage appears broad

The repository contains tests for:

- transition matrices,
- MSM lag guards,
- weighted-ensemble weights,
- recycling,
- failed walkers,
- exactness of WE rates,
- checkpoint deltas,
- reproducibility,
- persistent workers,
- path reconstruction.

That is stronger than many research repositories. ([GitHub][2])

---

# 2. Main limitations for committor calculation

## 2.1 No committor implementation currently exists

The analysis package presently contains data and plotting utilities but no dedicated committor module. ([GitHub][3])

You will need at least:

```text
trails_md/
    analysis/
        committor.py
        shooting.py
        tpt.py
```

## 2.2 Exploration data are selection-biased

In exploration mode, frames are preferentially selected because they are:

- under-sampled,
- far from previous samples,
- local outliers,
- near a target,
- favoured by an MSM uncertainty score.

The MSM spawner itself deliberately targets states with high uncertainty and leverage.

Therefore:

> Occupancy in the exploration dataset is not an equilibrium population, and raw transition counts across the whole adaptive campaign are not automatically an unbiased kinetic estimate.

This does not make the data useless. It makes the data primarily useful for:

- identifying state-space support,
- selecting local shooting configurations,
- learning an initial discretisation,
- finding candidate pathways.

## 2.3 Fresh velocities change the object being estimated

For underdamped Langevin or ordinary molecular dynamics, the exact committor is formally a function of phase space,

$$
q(\mathbf{x},\mathbf{v})
=
\Pr\!\left[
\tau_B<\tau_A
\mid
\mathbf{x}_0=\mathbf{x},
\mathbf{v}_0=\mathbf{v}
\right].
$$

A configuration-only committor is a velocity-averaged quantity,

$$
\bar q(\mathbf{x})
=
\int q(\mathbf{x},\mathbf{v})
\,\rho(\mathbf{v}\mid\mathbf{x})\,d\mathbf{v}.
$$

Drawing new Maxwell–Boltzmann velocities is appropriate when estimating $\bar q(\mathbf{x})$, but it is not equivalent to continuing the original trajectory.

The analysis API should explicitly distinguish:

- `phase_space_committor`,
- `configuration_committor`,
- `overdamped_committor`.

## 2.4 Clustering changes between adaptive iterations

The estimator can initialise K-means from previous centres when `stable_clustering=True`, which is useful.

However, if the CV model itself is retrained, the coordinate system may change. A committor assembled across changing latent spaces becomes difficult to interpret.

For committor production:

- freeze the feature map,
- freeze the scaler,
- freeze the clustering,
- then collect local kinetic data.

## 2.5 Largest-connected-set restriction can discard important frontier states

The MSM estimator calls `submodel_largest()`.

That is often reasonable for a production MSM, but potentially problematic during transition discovery. A poorly sampled bridge state may be excluded even though it is mechanistically central.

For committor analysis, save both:

- the complete count graph,
- the largest connected subgraph.

Report excluded states and their counts.

## 2.6 Current uncertainty is not committor uncertainty

Bayesian MSM support currently provides posterior uncertainty mainly for implied timescales.

You need uncertainty propagation for $q$, preferably by:

1. sampling transition matrices from a row-wise Dirichlet posterior or Bayesian MSM posterior;
2. solving the committor for each sample;
3. reporting posterior mean and credible interval for every state.

---

# 3. What committor should Trails-MD compute?

Let $A$ and $B$ be disjoint metastable sets. The forward committor is

$$
q^+(\mathbf{x})
=
\Pr_{\mathbf{x}}\!\left(\tau_B<\tau_A\right).
$$

It satisfies the backward-Kolmogorov boundary-value problem,

$$
\mathcal{L}q^+(\mathbf{x})=0,
\qquad
q^+|_A=0,
\qquad
q^+|_B=1.
$$

For a discrete Markov model with transition matrix $P$,

$$
q_i = \sum_j P_{ij}q_j,
\qquad i\notin A\cup B,
$$

with (q_i=0) in $A$ and (q_i=1) in $B$. This discrete TPT construction is standard for Markov chains. ([SIAM][4])

Partitioning states into interior $I$, reactant $A$, and product $B$,

$$
\left(I-P_{II}\right)\mathbf{q}_I
=
P_{IB}\mathbf{1}.
$$

That is the main linear system Trails-MD needs to solve.

---

# 4. Recommended hierarchy of committor estimators

Use three estimators, not one.

## Level 1: Direct shooting committor

For configuration $\mathbf{x}_k$, launch $N_k$ independent unbiased trajectories with independently sampled thermal noise or velocities.

Stop each trajectory on first entry into $A$ or $B$:

$$
Y_{kn}
=
\begin{cases}
1, & B\text{ reached first},\\
0, & A\text{ reached first}.
\end{cases}
$$

Then

$$
\hat q(\mathbf{x}_k)
=
\frac{1}{N_k}\sum_{n=1}^{N_k}Y_{kn}.
$$

A useful Bayesian estimate is

$$
q_k\mid\{Y_{kn}\}
\sim
\operatorname{Beta}
\left(n_B+\frac{1}{2},n_A+\frac{1}{2}\right),
$$

using Jeffreys’ prior.

This is the reference calculation and should be treated as ground truth for the toy potential.

## Level 2: MSM/TPT committor

Construct a transition matrix from unbiased trajectory segments and solve the discrete Dirichlet problem.

This gives a committor for every microstate rather than only selected shooting structures.

## Level 3: Locally stitched committor

For sparse or expensive systems, estimate only local transition probabilities between neighbouring cells or milestones.

For state $i$, launch short trajectories until they hit one of its neighbouring interfaces. Estimate

$$
K_{ij}
=
\Pr\!\left(\text{next interface/state is }j\mid i\right).
$$

Then solve

$$
q_i = \sum_j K_{ij}q_j
$$

with the same boundary values.

This is the principled version of your “piecewise committor from local transition” idea.

It closely resembles a discrete harmonic extension on a locally estimated transition graph. Point-cloud and Markov-chain formulations of TPT use this same principle: construct a local Markov process and solve for the discrete committor. ([SIAM][5])

---

# 5. Important condition for local stitching

The local state must be approximately Markovian.

If all configurations assigned to state $i$ have different hidden variables or velocity histories, a single number $K_{ij}$ is not sufficient.

Test this explicitly:

1. Divide configurations within each state into independent subsets.
2. Estimate local transition rows $K_{i\cdot}^{(1)}$ and $K_{i\cdot}^{(2)}$.
3. Compare them using total-variation distance or Jensen–Shannon divergence.
4. Repeat after adding more CVs or reducing state size.
5. Reject the local-state representation if transition probabilities remain history-dependent.

For underdamped systems, possible remedies are:

- include velocity projected along the pathway,
- use longer local equilibration constrained within the cell,
- use a history-augmented state label,
- adopt core-set or milestoning definitions,
- increase the MSM lag time.

---

# 6. Proposed Trails-MD workflow

## Phase A: Exploratory campaign

Run Trails-MD in exploration mode.

For the toy potential, use physical coordinates:

$$
\boldsymbol{\xi}(\mathbf{x})=$x,y$.
$$

Outputs:

- sampled point cloud,
- connected lineage paths,
- candidate basins,
- candidate transition channels,
- frozen clustering.

Do not calculate final kinetics from this stage.

### Suggested spawning sequence

1. Density or FPS for broad coverage.
2. Voronoi or LOF to fill holes.
3. MSM-guided spawning only after a connected transition region appears.
4. Freeze the discretisation before shooting.

The existing MSM spawner is useful for identifying statistically weak but dynamically influential states.

---

## Phase B: Define $A$ and $B$

Use core sets rather than broad geometrical halves.

For a two-dimensional double-well or Müller–Brown system:

$$
A
=
\left\{
\mathbf{x}:\lVert\mathbf{x}-\mathbf{x}_A\rVert<r_A
\right\},
$$

$$
B
=
\left\{
\mathbf{x}:\lVert\mathbf{x}-\mathbf{x}_B\rVert<r_B
\right\}.
$$

Choose radii so that trajectories entering a core rapidly relax and have a low probability of immediately exiting.

Perform a sensitivity study with at least three radii.

---

## Phase C: Frozen state-space model

Freeze:

- CV transformation,
- scaler,
- cluster centres,
- basin definitions,
- neighbour graph.

Use either:

- regular $x,y$ grid,
- Voronoi cells around K-means centres,
- adaptive cells refined around $q\approx0.5$.

For the first implementation, use a regular grid. It produces a clean numerical reference.

---

## Phase D: Select local shooting structures

For each cell $i$:

- draw configurations from multiple walkers and iterations;
- avoid using many nearly consecutive frames from one trajectory;
- use spatially separated representatives;
- record their lineage and origin.

Initial allocation:

$$
N_i^{(0)}=20\text{--}50
$$

shooting trajectories per cell.

For direct shooting, continue each trajectory until $A$ or $B$.

For local stitching, stop at the first hit of:

- a neighbouring cell core,
- $A$,
- $B$,
- a maximum duration.

---

## Phase E: Adaptive allocation near the transition state

The Bernoulli uncertainty of direct shooting is largest near $q=0.5$:

$$
\operatorname{SE}(\hat q_i)
=
\sqrt{\frac{\hat q_i(1-\hat q_i)}{N_i}}.
$$

Therefore allocate more trajectories to cells with:

- posterior mean near 0.5,
- wide credible interval,
- high reactive flux,
- disagreement between local and direct estimators,
- strong within-cell heterogeneity.

A reasonable toy-system target is:

- basin-like cells: 20–50 shots,
- intermediate cells: 100–200 shots,
- candidate transition-state cells: 500–1000 shots.

---

# 7. Piecewise/local committor construction

## Option A: Local cell-to-cell transition matrix

For every interior state $i$, estimate counts

$$
C_{ij}
=
\text{number of local trajectories starting in }i
\text{ whose first registered destination is }j.
$$

Use a Dirichlet-smoothed estimate,

$$
K_{ij}
=
\frac{C_{ij}+\alpha_{ij}}
{\sum_k\left(C_{ik}+\alpha_{ik}\right)}.
$$

With $\alpha_{ij}=1/2$ for allowed neighbours, solve

$$
(I-K_{II})q_I=K_{IB}\mathbf{1}.
$$

This yields the stitched committor.

### Advantages

- local trajectories can be short;
- each region can be sampled independently;
- embarrassingly parallel;
- adaptive refinement is straightforward;
- compatible with Trails-MD walker execution.

### Risks

- missed non-local jumps;
- non-Markovian cells;
- insufficient velocity treatment;
- artificial dependence on cell boundaries;
- disconnected graph regions.

---

## Option B: Milestone-to-milestone committor

Construct interfaces

$$
\Sigma_0,\Sigma_1,\ldots,\Sigma_M
$$

across the transition tube.

From configurations equilibrated on milestone $\Sigma_i$, estimate the probability of hitting each neighbouring milestone first:

$$
K_{ij}
=
\Pr\!\left(\Sigma_j\text{ is hit next}\mid\Sigma_i\right).
$$

Set:

$$
q_0=0,\qquad q_M=1.
$$

and solve the interior harmonic equations.

This is more efficient when the transition is approximately one-dimensional, but it can fail when there are multiple pathways or strong recrossing. In a multi-channel landscape, use a network of milestones rather than a single ordered chain.

---

## Option C: Committor-stratified interfaces

After obtaining an approximate $q^{(0)}$, define interfaces by its level sets:

$$
\Sigma_k
=
\left\{
\mathbf{x}:q^{(0)}(\mathbf{x})\approx \frac{k}{M}
\right\}.
$$

Then resample locally and recompute the transition graph.

This produces an iterative protocol:

$$
\text{exploration}
\rightarrow
q^{(0)}
\rightarrow
\text{isocommittor-like strata}
\rightarrow
\text{local resampling}
\rightarrow
q^{(1)}
\rightarrow\cdots
$$

Recent committor-guided pathway methods similarly alternate between estimating the committor and refining pathways or transition tubes. ([ACS Publications][6])

---

# 8. Recommended toy potential

Use the Müller–Brown potential first:

$$
V$x,y$
=
\sum_{k=1}^{4}
A_k
\exp\left[
a_k(x-x_k)^2
+b_k(x-x_k)(y-y_k)
+c_k(y-y_k)^2
\right].
$$

Standard parameters:

$$
A=(-200,-100,-170,15),
$$

$$
a=(-1,-1,-6.5,0.7),
$$

$$
b=(0,0,11,0.6),
$$

$$
c=(-10,-10,-6.5,0.7),
$$

$$
x_k=(1,0,-0.5,-1),
\qquad
y_k=(0,0.5,1.5,1).
$$

Use a shifted or scaled potential if necessary to produce convenient transition times.

## Dynamics

Start with overdamped Langevin dynamics:

$$
d\mathbf{X}_t
=
-D\beta\nabla V(\mathbf{X}_t)\,dt
+
\sqrt{2D}\,d\mathbf{W}_t.
$$

Euler–Maruyama:

$$
\mathbf{X}_{n+1}
=
\mathbf{X}_n
-
D\beta\nabla V(\mathbf{X}_n)\Delta t
+
\sqrt{2D\Delta t}\,\boldsymbol{\eta}_n.
$$

Suggested initial parameters:

```yaml
temperature_scale: 1.0
diffusion: 1.0
dt: 1.0e-4
segment_steps: 1000
stride: 10
walkers: 32
iterations: 100
```

The exact barrier scale should be checked numerically; do not commit to production parameters before measuring transition frequencies and integration stability.

---

# 9. Full toy-potential validation workflow

## Experiment 1: Reference PDE committor

On a fine rectangular grid, solve

$$
-\nabla V\cdot\nabla q
+
\beta^{-1}\Delta q
=
0.
$$

with

$$
q|_A=0,\qquad q|_B=1.
$$

Use finite differences or finite elements.

This produces (q\_{\mathrm{PDE}}$x,y$), the numerical reference.

## Experiment 2: Direct shooting

Select grid points or sampled configurations and run endpoint shooting.

Compute:

$$
q_{\mathrm{shoot}}(x_i,y_i)
=
\frac{n_B}{n_A+n_B}.
$$

Compare against the PDE reference.

## Experiment 3: Unbiased long-trajectory MSM

Generate several independent equilibrium or long unbiased trajectories.

Discretise them using the same grid/Voronoi model and solve:

$$
q_{\mathrm{MSM}}.
$$

Perform lag-time and grid-resolution studies.

## Experiment 4: Trails-MD exploration plus direct shooting

Use Trails-MD only for selecting candidate configurations.

Run independent unbiased endpoint shots from those configurations.

This tests whether adaptive exploration efficiently locates the $q\approx0.5$ region.

## Experiment 5: Local stitched committor

Run only neighbour-hitting simulations, construct $K$, and solve:

$$
q_{\mathrm{local}}.
$$

Compare:

$$
q_{\mathrm{local}}
\quad \text{vs.} \quad
q_{\mathrm{shoot}}
\quad \text{vs.} \quad
q_{\mathrm{PDE}}.
$$

## Experiment 6: Adaptive refinement

Refine cells where:

$$
0.2 < q < 0.8,
$$

or where posterior uncertainty is large.

Repeat local shooting and reconstruction.

---

# 10. Required validation metrics

Use more than visual agreement.

## Global error

$$
\operatorname{RMSE}_q
=
\sqrt{
\frac{1}{N}
\sum_i
\left(
q_i^{\mathrm{estimate}}
-
q_i^{\mathrm{PDE}}
\right)^2
}.
$$

Also report:

$$
\operatorname{MAE}_q
=
\frac{1}{N}
\sum_i
\left|q_i^{\mathrm{estimate}}-q_i^{\mathrm{PDE}}\right|.
$$

## Transition-region error

Calculate error only for:

$$
0.1 < q_{\mathrm{PDE}} < 0.9.
$$

This is more meaningful than error dominated by basin points.

## Calibration

Group predicted committors into bins. For configurations with predicted $q\approx0.3$, approximately 30% of direct shots should reach $B$ first.

## Harmonic residual

For every interior state:

$$
r_i = q_i - \sum_j K_{ij}q_j.
$$

Report:

- maximum $|r_i|$,
- weighted RMS residual.

## Transition-state consistency

For configurations with estimated $q\in[0.45,0.55]$, perform independent shooting.

Their outcome distribution should be approximately balanced.

Committor shooting is the standard direct test for identifying or validating a transition-state ensemble. ([ACS Publications][7])

## Resolution convergence

Repeat for:

- $25\times25$,
- $50\times50$,
- $100\times100$ grids,

or equivalent Voronoi resolutions.

## Locality convergence

Increase maximum neighbour range and confirm that the reconstructed committor does not change materially.

---

# 11. Proposed implementation inside Trails-MD

## Data structures

```python
@dataclass
class BasinDefinition:
    name: str
    predicate: Callable[[np.ndarray], np.ndarray]


@dataclass
class ShootingResult:
    start_key: str
    start_state: int
    destination_state: int | None
    committed_to: str | None
    first_passage_steps: int
    censored: bool
    seed: int


@dataclass
class CommittorResult:
    committor: np.ndarray
    state_ids: np.ndarray
    basin_a_states: np.ndarray
    basin_b_states: np.ndarray
    transition_matrix: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    residual: np.ndarray
```

## Core discrete solver

```python
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve


def solve_forward_committor(
    transition_matrix: np.ndarray,
    basin_a: np.ndarray,
    basin_b: np.ndarray,
) -> np.ndarray:
    """Solve q = Pq with q$A$=0 and q$B$=1."""
    p = np.asarray(transition_matrix, dtype=float)

    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("transition_matrix must be square.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Rows of transition_matrix must sum to one.")

    n_states = p.shape[0]
    a = np.unique(np.asarray(basin_a, dtype=int))
    b = np.unique(np.asarray(basin_b, dtype=int))

    if np.intersect1d(a, b).size:
        raise ValueError("A and B must be disjoint.")

    boundary = np.union1d(a, b)
    interior = np.setdiff1d(np.arange(n_states), boundary)

    q = np.zeros(n_states, dtype=float)
    q[b] = 1.0

    if interior.size:
        p_ii = csr_matrix(p[np.ix_(interior, interior)])
        rhs = p[np.ix_(interior, b)].sum(axis=1)
        q[interior] = spsolve(eye(interior.size) - p_ii, rhs)

    return np.clip(q, 0.0, 1.0)
```

## Bayesian uncertainty

```python
def sample_row_dirichlet(
    counts: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    samples = np.zeros_like(counts)

    for i, row in enumerate(counts):
        allowed = row > 0

        if not np.any(allowed):
            samples[i, i] = 1.0
            continue

        concentration = row[allowed] + alpha
        samples[i, allowed] = rng.dirichlet(concentration)

    return samples
```

Solve $q$ for hundreds or thousands of sampled matrices and calculate quantiles.

---

# 12. New CLI proposal

```bash
trails-md-committor prepare \
    --run-dir runs/muller_brown \
    --config config.yaml \
    --basin-a basin_a.yaml \
    --basin-b basin_b.yaml \
    --states grid \
    --grid 50 50 \
    --output committor_campaign
```

```bash
trails-md-committor shoot \
    --campaign committor_campaign \
    --mode endpoint \
    --shots-per-state 50 \
    --max-steps 100000
```

```bash
trails-md-committor solve \
    --campaign committor_campaign \
    --estimator direct
```

or:

```bash
trails-md-committor shoot \
    --campaign committor_campaign \
    --mode local-first-exit \
    --shots-per-state 100
```

```bash
trails-md-committor solve \
    --campaign committor_campaign \
    --estimator local-markov \
    --posterior-samples 1000
```

Outputs should include:

```text
committor.npz
state_table.csv
shooting_results.parquet
transition_counts.npz
transition_matrix.npz
committor_map.png
committor_uncertainty.png
harmonic_residual.png
validation_report.json
```

---

# 13. Recommended staged development

## Stage 1: Minimal toy implementation

Implement:

- overdamped two-dimensional toy engine,
- rectangular grid,
- direct endpoint shooting,
- discrete committor solver,
- PDE reference,
- comparison plots.

## Stage 2: Local stitching

Add:

- first-exit local trajectories,
- neighbour graph,
- Dirichlet transition estimates,
- posterior committor intervals,
- adaptive sampling around $q=0.5$.

## Stage 3: Integrate with existing MSM

Allow:

```python
result = core.last_msm_result
q = solve_forward_committor(
    result.transition_matrix,
    basin_a_states,
    basin_b_states,
)
```

Carefully map full cluster IDs to active-state IDs using `state_symbols`, because the MSM is restricted to its largest connected set.

## Stage 4: Molecular systems

Move to alanine dipeptide:

- $A$: C7eq core,
- $B$: C7ax or another clearly defined core,
- CVs: periodic $(\phi,\psi)$,
- independent Maxwell–Boltzmann velocity draws,
- direct shooting validation at $q\approx0.5$.

Only after this should the method be applied to protein folding or ligand unbinding.

---

# 14. Final recommended protocol

For Trails-MD, the recommended scientific protocol is structured as follows:

1. **Adaptive exploration**
2. **Freeze CVs and discretisation**
3. **Define strict $A/B$ core sets**
4. **Select diverse structures per local state**
5. **Run unbiased local or endpoint shootings**
6. **Estimate $K_{ij}$ with uncertainty**
7. $$ (I - K*{II})q_I = K*{IB}\mathbf{1} $$

8. **Refine states near $q = 0.5$**
9. **Validate against direct shooting and PDE**

The key methodological claim would be:

> Trails-MD is used to discover and stratify the transition region; the committor itself is reconstructed from independently launched, unbiased local first-passage simulations and validated by direct endpoint shooting.

That claim is substantially safer than calculating a committor directly from adaptively selected exploration trajectories. The latter may produce a useful heuristic coordinate, but without additional correction and Markov validation it should not be described as the physical committor.

[1]: https://github.com/TeamSuman/Trails-MD "GitHub - TeamSuman/Trails-MD: Lightweight, engine-agnostic framework for lineage-aware adaptive molecular dynamics sampling across fixed or machine-learned collective-variable spaces · GitHub"
[2]: https://github.com/TeamSuman/Trails-MD/tree/main/tests "Trails-MD/tests at main · TeamSuman/Trails-MD · GitHub"
[3]: https://github.com/TeamSuman/Trails-MD/tree/main/trails_md/analysis "Trails-MD/trails_md/analysis at main · TeamSuman/Trails-MD · GitHub"
[4]: https://epubs.siam.org/doi/10.1137/070699500?utm_source=chatgpt.com "Transition Path Theory for Markov Jump Processes"
[5]: https://epubs.siam.org/doi/10.1137/21M1437883?utm_source=chatgpt.com "Transition Path Theory for Langevin Dynamics on Manifolds"
[6]: https://pubs.acs.org/doi/10.1021/acs.jctc.6c00007?utm_source=chatgpt.com "Following the Committor Flow: A Data-Driven Discovery of ..."
[7]: https://pubs.acs.org/doi/10.1021/acs.jctc.2c00543?utm_source=chatgpt.com "ATESA: An Automated Aimless Shooting Workflow"
