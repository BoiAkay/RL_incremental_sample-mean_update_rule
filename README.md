# ⚡ Incremental Mean Update — RL Bandit Visualizer

> An interactive visualization of the **Incremental Sample-Mean Update Rule** from Reinforcement Learning — derived from Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 2.

---

## 🔢 The Formula

$$Q_{n+1} = Q_n + \frac{1}{n}\left[R_n - Q_n\right]$$

| Symbol | Meaning |
|---|---|
| $Q_{n+1}$ | New estimate after n rewards |
| $Q_n$ | Current (old) estimate |
| $R_n$ | Latest reward received |
| $\frac{1}{n}$ | Step-size α — shrinks with every step |
| $[R_n - Q_n]$ | Prediction error δ |

This is **mathematically equivalent** to computing the exact sample mean — but uses **O(1) memory** and a single arithmetic update per step instead of storing all past rewards.

---

## 🚀 Live Demo

👉 **[Open the visualizer](./index.html)** — or deploy to Netlify / GitHub Pages (see [Deployment](#deployment))

---

## ✨ Features

- **Step-by-step derivation** — all 6 algebraic steps from the full sum to the incremental form, with hover cards
- **Syntax-highlighted Python code** — with one-click copy
- **Live interactive demo**
  - Feed rewards manually or randomly
  - ⚡ Burst mode — simulate hundreds of steps instantly
  - Real-time chart — Q estimate vs true running mean vs raw rewards
  - Scrollable step log — n, Rₙ, error δ, α, and Qₙ₊₁ at every step
- **Zero dependencies** — pure HTML + CSS + JS, single file

---

## 📐 Step-by-Step Derivation

```
Step 1:  Q_{n+1} = (1/n) Σ R_i                     ← exact mean definition

Step 2:  = (1/n)(R_n + Σ_{i=1}^{n-1} R_i)          ← peel off latest reward

Step 3:  = (1/n)(R_n + (n-1) · Q_n)                ← recognise Q_n in the sum

Step 4:  = (1/n)(R_n + nQ_n − Q_n)                 ← expand (n−1)

Step 5:  = Q_n + (1/n)[R_n − Q_n]                  ← final incremental form ✓

Step 6:  = Q_n + α[R_n − Q_n]  (α fixed)           ← non-stationary variant
```

The **circled substitution** (Step 3) is the key insight: the sum of (n−1) past rewards divided by (n−1) is exactly the previous estimate $Q_n$.

---

## 🐍 Python Implementation

```python
class IncrementalMean:
    """Sample-average action value estimator — O(1) memory."""
    def __init__(self, Q0=0.0):
        self.Q = Q0   # current estimate
        self.n = 0    # step counter

    def update(self, R):
        """Feed one reward; returns (new_Q, error, alpha)."""
        self.n += 1
        alpha  = 1.0 / self.n        # step-size α = 1/n
        error  = R - self.Q          # prediction error δ
        self.Q += alpha * error      # Q ← Q + α·δ
        return self.Q, error, alpha


class IncrementalMeanConstantAlpha:
    """Non-stationary variant: fixed α weights recent rewards more."""
    def __init__(self, alpha=0.1, Q0=0.0):
        self.Q     = Q0
        self.alpha = alpha
        self.n     = 0

    def update(self, R):
        self.n += 1
        error  = R - self.Q
        self.Q += self.alpha * error
        return self.Q, error, self.alpha
```

### Sample Output

```
Step   Reward      Error    Alpha      Q_est
------------------------------------------------
   1     4.00    +4.0000   1.0000     4.000000
   2     2.00    -2.0000   0.5000     3.000000
   3     7.00    +4.0000   0.3333     4.333333
   4     1.00    -3.3333   0.2500     3.500000
   5     5.00    +1.5000   0.2000     3.800000
```

---

## ⚠️ Drawbacks of the 1/n Rule

| Issue | Sample-Mean (α = 1/n) | Constant-α |
|---|---|---|
| Stationary environments | ✅ Optimal | ⚠️ Small residual bias |
| Non-stationary environments | ❌ Blind to drift | ✅ Tracks changes |
| Memory | ✅ O(1) | ✅ O(1) |
| Convergence guarantee | ✅ Yes | ❌ Fluctuates forever |
| Recency weighting | ❌ Uniform | ✅ Exponential decay |

**Key problem:** After 1000 steps, α = 0.001 — new rewards have almost no influence. If the true reward distribution shifts, the estimator cannot track it.

---

## 🗂️ Project Structure

```
.
├── index.html          # Entire app — single self-contained file
└── README.md           # This file
```

No build step, no npm, no frameworks. Open `index.html` directly in any browser.

---

## 🌐 Deployment

### Netlify Drop *(30 seconds)*
1. Go to [netlify.com/drop](https://app.netlify.com/drop)
2. Drag and drop `index.html`
3. Get a live `*.netlify.app` URL instantly

### GitHub Pages
```bash
git init
git add index.html README.md
git commit -m "feat: incremental mean update RL visualizer"
gh repo create rl-incremental-mean --public --push --source=.
# Settings → Pages → Deploy from branch → main / root
```

### Vercel CLI
```bash
npm i -g vercel
vercel deploy index.html
```

---

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), §2.4 — Incremental Implementation. MIT Press.
- Robbins, H., & Monro, S. (1951). A Stochastic Approximation Method. *Annals of Mathematical Statistics*, 22(3), 400–407.

---

## 🎓 Context

Built as a study companion for:

**BITS Pilani WILP — Deep Reinforcement Learning (AIMLZG512)**

The incremental update rule is foundational to understanding how agents maintain value estimates efficiently — from simple bandits all the way up to Q-learning and actor-critic methods.

---

## 📄 License

MIT — free to use, modify, and share.
