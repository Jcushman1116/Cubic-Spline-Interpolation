# Interpolation Methods

A Python implementation and empirical analysis of five numerical interpolation techniques, comparing accuracy, smoothness, and derivative behavior across smooth test functions and real discrete data.

---

## Methods Implemented

### Global Barycentric-1 Polynomial
Constructs a single global interpolating polynomial using the Barycentric-1 form with Chebyshev points of the first kind and Leja ordering. Chebyshev nodes cluster near the endpoints to suppress oscillation and mitigate Runge's phenomenon. Weights are precomputed in O(n²) and reused for O(n) evaluation per point.

### Piecewise Newton Interpolant
Partitions the interval into m subintervals and constructs a local degree-d Newton polynomial on each using divided differences evaluated via Horner's rule. Supports both uniform and Chebyshev node distributions. Does not guarantee derivative continuity across subinterval boundaries.

### Piecewise Cubic Hermite Interpolant
Extends the Newton approach by incorporating both function values and first derivatives at each subinterval endpoint using a modified divided difference table with repeated nodes. Guarantees C¹ continuity across the global interval.

### Interpolatory Cubic Spline
Constructs a piecewise cubic function satisfying C² continuity at all interior nodes by solving a tridiagonal linear system derived from smoothness conditions. Supports two boundary condition types:
- **S1** — prescribed first derivatives at endpoints
- **S2** — prescribed second derivatives (natural spline when set to zero)

Evaluated using the cubic Hermite basis with local coordinate mapping.

### Interpolatory Cubic B-Spline
Parameterizes the cubic spline in terms of the B-spline basis, where each basis function has compact support on four subintervals. Coefficients are determined by solving a banded (n+3) × (n+3) linear system. Offers improved local control and numerical stability — perturbing one coefficient affects only its local support region.

---

## Test Functions

| Function | Interval | Purpose |
|----------|----------|---------|
| `x³ + x² + x` | [-1, 1] | Polynomial reproduction test |
| `sin(x)` | [-1, 1] | Smooth periodic function |
| `1 / (1 + 25x²)` | [-1, 1] | Runge's function — tests oscillation |
| `exp(-x²)` | [-2, 2] | Hermite validation (known derivative) |
| `x³ - 2x² + x + 1` | [0, 5] | Spline reproduction test |

---

## Error Metrics

All methods are evaluated using four metrics:

- **Infinity Norm** `‖f - pₙ‖∞` — maximum pointwise error
- **MSE** — mean squared error
- **Max Relative Error** — scale-independent maximum error
- **MAE** — mean absolute error

---

## Discrete Data Recovery (Task 2)

The methods are applied to recover three interrelated functions from 8 nonuniformly spaced measurements of y(tᵢ):

| tᵢ | 0.5 | 1.0 | 2.0 | 4.0 | 5.0 | 10.0 | 15.0 | 20.0 |
|----|-----|-----|-----|-----|-----|------|------|------|
| y(tᵢ) | 0.0552 | 0.0600 | 0.0682 | 0.0801 | 0.0843 | 0.0931 | 0.0912 | 0.0857 |

The target quantities are:

```
f(t) = y(t) + t·y′(t)
D(t) = exp(-t · y(t))
```

Both require `y′(t)`, which is not directly available. The **natural cubic spline** (S2, zero second derivatives at endpoints) recovers a smooth C² interpolant whose derivative is computed analytically from the Hermite basis. The piecewise Newton interpolant uses `np.gradient` for numerical differentiation, which introduces discontinuities at subinterval boundaries that propagate into `f(t)` and `D(t)`.

---

## Key Results

- The **cubic spline (S1)** reproduced `x³ - 2x² + x + 1` to near machine precision (infinity norm: `2.84e-14`)
- The **Barycentric-1** method achieved infinity norm `4.78e-11` on `sin(x)` with 10 nodes, but showed significant error (`2.69e-01`) on Runge's function — expected behavior
- The **natural cubic spline** consistently outperformed piecewise Newton for the discrete data task, producing smooth derivatives essential for computing `f(t)` and `D(t)`

---

## Requirements

```
numpy
matplotlib
```

Install with:

```bash
pip install numpy matplotlib
```

---

## Usage

```bash
python CubicSpline.py
```

Prints error metrics for all methods and displays interpolation and comparison plots.

---

## Project Structure

```
├── CubicSpline.py            # All implementations and task runners
└── Cubic-Spline-Report.pdf   # Full write-up with derivations and results
```
