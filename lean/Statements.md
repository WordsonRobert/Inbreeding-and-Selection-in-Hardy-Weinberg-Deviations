# Formalised statements

Each statement is written in LaTeX with exactly the hypotheses proved in Lean, followed by
its Lean names so that it can be traced to `BastionTheory/`.  The statements are numbered
P1–P13 here; the manuscript merges them into Propositions 1–12 (see the table in
`README.md`).  Every Lean name below also appears in
`BastionTheory/Checks.lean`, which audits the axiom dependencies of all 81 of them, and in
the table in `README.md`.

## Notation

A genotype vector is $x=(x_1,x_2,x_3)\in\mathbb{R}^3$ (frequencies of $AA$, $Aa$, $aa$),
with

$$p(x)=x_1+\tfrac{x_2}{2},\qquad q(x)=1-p(x),\qquad D(x)=x_1-p(x)^2 .$$

$x$ lies in the *genotype simplex* $\Delta$ when $x_i\ge 0$ and $x_1+x_2+x_3=1$.  Write

$$\mathrm{vec}(p,D)=\bigl(p^2+D,\;2(pq-D),\;(1-p)^2+D\bigr),\qquad
\mathrm{zyg}(p,F)=\mathrm{vec}(p,Fpq).$$

For fitnesses $w=(w_1,w_2,w_3)$ put $\bar w(x)=w_1x_1+w_2x_2+w_3x_3$ and

$$(\mathrm{sel}_w x)=\frac{1}{\bar w(x)}\,(w_1x_1,\;w_2x_2,\;w_3x_3),
\qquad \kappa(w)=w_1w_3-w_2^2 .$$

The BASTION fitnesses are $w=(1,\,1-hs,\,1-s)$.  Genotype-level mutation with rates
$A\to a$ equal to $u$ and $a\to A$ equal to $v$ is

$$M_{u,v}(x)=(n_1,\,1-n_1-n_3,\,n_3),\quad
\begin{aligned}
n_1&=x_1(1-u)^2+x_2(1-u)v+x_3v^2,\\
n_3&=x_1u^2+x_2u(1-v)+x_3(1-v)^2,
\end{aligned}$$

and the remaining operators are $Y_0(x)=\mathrm{vec}(p(x),0)$,
$I_F(x)=\mathrm{vec}\bigl(p(x),\,D(x)+Fp(x)q(x)\bigr)$,
$\mathrm{Dr}_\delta(x)=\mathrm{vec}\bigl(p(x)+\delta,\;
\tfrac{D(x)}{p(x)q(x)}(p(x)+\delta)(1-p(x)-\delta)\bigr)$ and $S=\mathrm{sel}_w$.

---

## Proposition 1 — the two forms of the Hardy–Weinberg deviation

**Proposition.** If $x_1+x_2+x_3=1$ then $D(x)=x_1x_3-\dfrac{x_2^2}{4}$.

*Lean:* `Geno.D_eq_of_total` (and `Geno.D_eq` on $\Delta$).

## Proposition 2 — coordinates and admissibility

**Proposition.** For all $p,D\in\mathbb{R}$,
$p(\mathrm{vec}(p,D))=p$ and $D(\mathrm{vec}(p,D))=D$; and if $x_1+x_2+x_3=1$ then
$\mathrm{vec}(p(x),D(x))=x$.  For $0\le p\le 1$,

$$\mathrm{vec}(p,D)\in\Delta\iff -\min(p,1-p)^2\;\le\;D\;\le\;p(1-p),$$

and for $0<p<1$,

$$\mathrm{zyg}(p,F)\in\Delta\iff \frac{-\min(p,1-p)}{\max(p,1-p)}\;\le\;F\;\le\;1 .$$

*Lean:* `p_vec`, `D_vec`, `vec_p_D_self`, `inSimplex_vec_iff`, `inSimplex_zyg_iff`.
*(The first two hold with no hypothesis at all.)*

## Proposition 3 — closed form for the post-selection deviation

**Proposition.** If $\bar w(x)\neq 0$ then

$$D(\mathrm{sel}_w x)=\frac{w_1w_3\,x_1x_3-w_2^2\,x_2^2/4}{\bar w(x)^2}.$$

*Lean:* `D_sel_closed`.  *(Neither $x\in\Delta$ nor $w>0$ is needed.)*

## Proposition 4 — the $\kappa$ decomposition

**Proposition.** If $x_1+x_2+x_3=1$ and $\bar w(x)\neq 0$ then

$$D(\mathrm{sel}_w x)=\frac{w_1w_3\,D(x)+\kappa(w)\,(x_2/2)^2}{\bar w(x)^2},$$

and for the BASTION fitnesses $\kappa=s(2h-1)-h^2s^2$.

*Lean:* `D_sel_kappa`, `kappa_bastion`.

## Proposition 5 — sign rules

**Proposition (a).** If $x_1+x_2+x_3=1$, $\bar w(x)>0$, $x_2>0$ and $D(x)=0$, then

$$D(\mathrm{sel}_w x)=\kappa(w)\cdot\frac{(x_2/2)^2}{\bar w(x)^2},$$

so $D(\mathrm{sel}_w x)>0\iff\kappa>0$, $=0\iff\kappa=0$, and $<0\iff\kappa<0$.

**Proposition (b).** If $w>0$ and $w_1<w_2$, $w_3<w_2$ (overdominance) then $\kappa<0$;
hence for $0<p<1$, $D(\mathrm{sel}_w\,\mathrm{vec}(p,0))<0$, i.e. selection alone creates a
heterozygote excess out of Hardy–Weinberg zygotes.

**Proposition (c).** If $w>0$ and $w_2<w_1$, $w_2<w_3$ (underdominance) then $\kappa>0$ and
$D(\mathrm{sel}_w\,\mathrm{vec}(p,0))>0$ for $0<p<1$.

**Proposition (d).** If $w_2^2=w_1w_3$ (multiplicative fitness) then $\kappa=0$, and for
$w>0$, $0<p<1$, $D(\mathrm{sel}_w\,\mathrm{vec}(p,0))=0$.

**Proposition (e).** If $w>0$, $x\in\Delta$, $D(x)\ge 0$ and $\kappa\ge 0$ then
$D(\mathrm{sel}_w x)\ge 0$; and with all inequalities reversed, $D(\mathrm{sel}_w x)\le 0$.

**Proposition (f).**  For every $x$, $x_2=2\bigl(p(x)(1-p(x))-D(x)\bigr)$; so $D<0$ is
exactly a heterozygote excess relative to Hardy–Weinberg.

*Lean:* `D_sel_of_D_eq_zero`, `D_sel_pos_iff`, `D_sel_eq_zero_iff`, `D_sel_neg_iff`,
`x2_eq`, `kappa_neg_of_overdominance`, `D_sel_vec_zero_neg`,
`kappa_pos_of_underdominance`, `D_sel_vec_zero_pos`, `kappa_eq_zero_of_multiplicative`,
`D_sel_vec_zero_eq_zero`, `D_sel_nonneg`, `D_sel_nonpos`.

## Proposition 6 — mutation

**Proposition.** If $0\le u\le 1$, $0\le v\le 1$ and $x\in\Delta$ then
$M_{u,v}(x)\in\Delta$.  If $x_1+x_2+x_3=1$ then

$$p\bigl(M_{u,v}x\bigr)=(1-u)\,p(x)+v\,\bigl(1-p(x)\bigr),
\qquad
D\bigl(M_{u,v}x\bigr)=(1-u-v)^2\,D(x).$$

*Lean:* `inSimplex_mutate`, `p_mutate`, `D_mutate`.

## Proposition 7 — the operators

**Proposition.** For every $x$: $D(Y_0x)=0$ and $p(Y_0x)=p(x)$;
$p(I_Fx)=p(x)$ and $D(I_Fx)=D(x)+F\,p(x)\,(1-p(x))$;
$p(\mathrm{Dr}_\delta x)=p(x)+\delta$ and
$D(\mathrm{Dr}_\delta x)=\tfrac{D(x)}{p(x)(1-p(x))}\,(p(x)+\delta)(1-p(x)-\delta)$.

Null operators are the identity: if $x_1+x_2+x_3=1$ then $M_{0,0}x=x$, $I_0x=x$ and
$\mathrm{sel}_{(1,1,1)}x=x$ (so $\mathrm{sel}_{w(s=0,h)}x=x$); and if in addition
$0<p(x)<1$ then $\mathrm{Dr}_0x=x$.

*Lean:* `p_reset`, `D_reset`, `p_inbreed`, `D_inbreed`, `p_drift`, `D_drift`,
`mutate_zero`, `inbreed_zero`, `sel_one`, `sel_bastion_zero`, `drift_zero`.

## Proposition 8 — life-cycle consistency, and agreement with the implementation

**Proposition (composition).** Let $a$ satisfy $a_1+a_2+a_3=1$ and put
$m=(1-u)\,p(a)+v\,(1-p(a))$ and $p'=m+\delta$.  Then
$Y_0(M_{u,v}a)=\mathrm{vec}(m,0)$ and
$I_F(Y_0(M_{u,v}a))=\mathrm{zyg}(m,F)$; and if $0<m<1$,

$$\mathrm{Dr}_\delta\bigl(I_F(Y_0(M_{u,v}a))\bigr)=\mathrm{zyg}(p',F),
\qquad
S\Bigl(\mathrm{Dr}_\delta\bigl(I_F(Y_0(M_{u,v}a))\bigr)\Bigr)
  =\mathrm{sel}_w\bigl(\mathrm{zyg}(p',F)\bigr).$$

**Proposition (agreement with the implementation).** Assume in addition $a\in\Delta$,
$0\le u\le 1$, $0\le v\le 1$, $w>0$, $0<p'<1$, and that $F$ is admissible — in the sense
of Proposition 2 — at **both** $m$ and $p'$, i.e.

$$\frac{-\min(m,1-m)}{\max(m,1-m)}\le F\le 1
\quad\text{and}\quad
\frac{-\min(p',1-p')}{\max(p',1-p')}\le F\le 1 .$$

Then every vector of the generation lies in $\Delta$:

$$M_{u,v}a,\quad \mathrm{vec}(m,0),\quad \mathrm{zyg}(m,F),\quad
\mathrm{zyg}(p',F),\quad \mathrm{sel}_w\bigl(\mathrm{zyg}(p',F)\bigr)\;\in\;\Delta .$$

By Proposition 2, membership of $\Delta$ is exactly the statement that the clip inside the
implementation's `_vec_with` is inactive, so under these hypotheses the implementation
computes the maps above at every step.  If $F$ is inadmissible at $m$ the implementation
clips at the inbreeding step and the drift step propagates the clipped $F_g$.

*Lean:* `reset_mutate`, `inbreed_reset_mutate`, `drift_inbreed_reset_mutate`, `lifecycle`,
`inSimplex_lifecycle`, `lifecycle_python_agreement`.

## Proposition 9 — Shapley attribution

For a game $v:\mathcal{P}(\{1,\dots,n\})\to\mathbb{R}$ define

$$\varphi_i(v)=\sum_{S\subseteq N\setminus\{i\}}
  \frac{|S|!\,(n-|S|-1)!}{n!}\,\bigl(v(S\cup\{i\})-v(S)\bigr).$$

**Proposition (efficiency).** $\displaystyle\sum_{i=1}^{n}\varphi_i(v)=v(N)-v(\varnothing)$.

**Proposition (dummy).** If $v(S\cup\{i\})=v(S)$ for every $S\not\ni i$ then
$\varphi_i(v)=0$.

**Proposition (symmetry).** If $i\neq j$ and $v(S\cup\{i\})=v(S\cup\{j\})$ for every $S$
containing neither $i$ nor $j$, then $\varphi_i(v)=\varphi_j(v)$.

**Proposition (telescoping).** For any chain
$\varnothing=S_0\subseteq S_1\subseteq\dots\subseteq S_n=N$,

$$\sum_{k=0}^{n-1}\bigl(v(S_{k+1})-v(S_k)\bigr)=v(N)-v(\varnothing).$$

*Lean:* `ShapleyGame.sum_shapley`, `ShapleyGame.shapley_dummy`,
`ShapleyGame.shapley_symm`, `ShapleyGame.sum_seq_telescope`.

**Applied to BASTION.** With $n=5$ operators in life-cycle order
$(M,Y_0,I,\mathrm{Dr},S)$ and $v(T)=D(\text{operators in }T\text{ applied to }a)-D(a)$:

$$\sum_{i=1}^{5}\varphi_i
  = D\Bigl(\mathrm{sel}_w\bigl(\mathrm{zyg}(p',F)\bigr)\Bigr)-D(a)$$

under the hypotheses of Proposition 8.  Each null operator receives exactly zero, for
$a_1+a_2+a_3=1$: $\varphi_S=0$ when $w=(1,1,1)$; $\varphi_M=0$ when $u=v=0$;
$\varphi_I=0$ when $F=0$; and $\varphi_{\mathrm{Dr}}=0$ when $\delta=0$, provided
additionally $0<p(a)<1$ and $0<m<1$ — the frequency reaching the drift step is $p(a)$ when
the mutation operator is absent from the coalition and $m$ when it is present, and
$\mathrm{Dr}_0$ is the identity only at an interior frequency.

*Lean:* `sum_shapley_game`, `sum_shapley_game_lifecycle`, `shapley_selection_eq_zero`,
`shapley_mutation_eq_zero`, `shapley_inbreeding_eq_zero`, `shapley_drift_eq_zero`.

## Proposition 10 — the two-player level split

With $v_I=D(\mathrm{zyg}(p,F))$, $v_S=D(\mathrm{sel}_w(\mathrm{zyg}(p,0)))$,
$v_{IS}=D(\mathrm{sel}_w(\mathrm{zyg}(p,F)))$ and

$$D^z=\tfrac{1}{2}(v_I+v_{IS}-v_S),\qquad D^S=\tfrac{1}{2}(v_S+v_{IS}-v_I),$$

**Proposition.** $D^z+D^S=v_{IS}$ for all $w,p,F$; $D^z=0$ when $F=0$; and $D^S=0$ when
$w=(1,1,1)$, i.e. when $s=0$.

*Lean:* `Dz_add_DS`, `Dz_eq_zero`, `DS_eq_zero`, `DS_eq_zero_bastion`.

## Proposition 11 — rare-allele selection

Let $q$ be the frequency of the deleterious allele $a$ in the zygotes
$\mathrm{zyg}(1-q,F)$, let $w=(1,1-hs,1-s)$, and let $q'$ be its frequency among the
adults.  Write $\bar w(q)$ for the mean fitness and
$s_{\mathrm{eff}}=s\bigl(F+(1-F)h\bigr)$.

**Proposition (a).** If $\bar w(q)\neq 0$ then, exactly,

$$q'-q=\frac{-s\,q(1-q)\bigl[F+(1-F)h+(1-F)(1-2h)q\bigr]}{\bar w(q)} .$$

*Lean (a):* `qAdult_sub`.

**Proposition (b).** The map $q\mapsto q'$ is differentiable at $q=0$ with

$$\left.\frac{dq'}{dq}\right|_{q=0}=1-s_{\mathrm{eff}} .$$

Consequently two parameter triples $(s,h,F)$ with the same $s_{\mathrm{eff}}$ have the same
linearised dynamics at $q=0$.  Under random mating, $s_{\mathrm{eff}}=hs$.

*Lean (b):* `hasDerivAt_qAdult`, `sEff_determines_linearisation`, `sEff_zero`.

**Proposition (c).** Define the first-order coefficients

$$c_2(s,h,F)=2(1-F)(1-hs),\qquad c_3(s,F)=F(1-s).$$

As $q\to 0$ (through $q\neq 0$),

$$\frac{a_2}{q}\longrightarrow c_2(s,h,F),\qquad
  \frac{a_3}{q}\longrightarrow c_3(s,F).$$

*Random mating.* If $F=0$ and $h_1s_1=h_2s_2$ then
$c_2(s_1,h_1,0)=c_2(s_2,h_2,0)$ and $c_3(s_1,0)=c_3(s_2,0)=0$: the first-order
coefficients depend on $(s,h)$ only through $hs$, and the $aa$ class is absent at first
order.  This is not merely an upper bound on what can be learned — $(s,h)=(0.2,0.5)$ and
$(s',h')=(0.1,1)$ are distinct and give identical first-order coefficients.

*Inbreeding.* If $0<F<1$ and $c_2(s_1,h_1,F)=c_2(s_2,h_2,F)$ and $c_3(s_1,F)=c_3(s_2,F)$,
then $s_1=s_2$ and $h_1s_1=h_2s_2$; if moreover $s_1\neq 0$ then $h_1=h_2$.  Equivalently,
stated through the limits themselves: if two parameter triples at the same $F\in(0,1)$ have
the same limits $L_2$ and $L_3$ for $a_2/q$ and $a_3/q$, then $s_1=s_2$ and
$h_1s_1=h_2s_2$.  **The paper may therefore not claim that only $hs$ is identified once
$F>0$.**

*Lean:* `tendsto_rareAdult_x2_div`, `tendsto_rareAdult_x3_div`, `tendsto_c2`, `tendsto_c3`,
`first_order_hs_only`, `first_order_not_identified`, `first_order_injective`,
`first_order_injective_h`, `first_order_limits_injective`.

## Proposition 12 — cross-sectional identifiability

**Proposition.** Let $J\in\mathbb{R}^{m\times k}$ have columns that each sum to zero, i.e.
$\sum_{r=1}^{m}J_{rc}=0$ for every $c$.  Then $\operatorname{rank}J\le m-1$, and for
*every* $W\in\mathbb{R}^{m\times m}$,

$$\operatorname{rank}\bigl(J^{\mathsf T}WJ\bigr)\;\le\;m-1 .$$

In particular a single cross-sectional genotype sample ($m=3$) carries Fisher information
of rank at most $2$, and a dominant-phenotype sample ($m=2$) of rank at most $1$, whatever
the number $k$ of parameters.

**Proposition (the hypothesis is automatic).** If $f:\mathbb{R}\to\mathbb{R}^m$ satisfies
$\sum_r f_r(t)=1$ for all $t$ and each $f_r$ is differentiable at $t_0$ with derivative
$d_r$, then $\sum_r d_r=0$: the Jacobian of any differentiable family of probability
vectors has zero column sums.

*Lean:* `rank_le_of_col_sum_zero`, `rank_quadratic_le`, `rank_fisher_genotype_le_two`,
`rank_fisher_phenotype_le_one`, `sum_deriv_eq_zero_of_probability_family`.

## Proposition 13 — drift versus the within-population $F_{IS}$

**Proposition (a).** If $F_{t+1}=c+(1-c)F_t$ for all $t$, then
$1-F_t=(1-c)^t\,(1-F_0)$.  With $c=1/(2N)$ this is Wright's recursion, and $F_t\to 1$.

**Proposition (b).** For $n\ge 1$ and any $p$,

$$\sum_{k=0}^{n}\binom{n}{k}p^k(1-p)^{n-k}\,\frac{k}{n}\Bigl(1-\frac{k}{n}\Bigr)
  = p(1-p)\Bigl(1-\frac1n\Bigr),$$

so under Wright–Fisher sampling of $n=2N$ gene copies
$\mathbb{E}[2\hat p\hat q]=2pq\bigl(1-\tfrac{1}{2N}\bigr)$.

**Proposition (c).** For $N\ge 1$ and $0<p<1$, with $\mathbb{E}[H_{\mathrm{obs}}]$ the
expected heterozygote fraction among $N$ individuals formed by random union of gametes
from a pool at frequency $p$ (each heterozygous with probability $2pq$, so
$\mathbb{E}[H_{\mathrm{obs}}]=2pq$),

$$1-\frac{\mathbb{E}[H_{\mathrm{obs}}]}{\mathbb{E}[2\hat p\hat q]}
  = -\frac{1}{2N-1},$$

a constant independent of the generation $t$, whereas Wright's $F_t$ of (a) grows
towards $1$.

**Scope.** This is the ratio of two expectations.  It is *not* a statement about
$\mathbb{E}[F_{IS}]$, which is the expectation of a ratio and does not equal this value.

*Lean:* `Drift.wright_closed_form`, `Drift.binom_moment1`, `Drift.binom_moment2`,
`Drift.binom_var_identity`, `Drift.expected_het_fraction`, `Drift.fis_ratio_full`.
