# =============================================================================
#  uk_olg.jl —  UK two-period OLG model of immigration policy
#  ( Busch et. al.)
# =============================================================================
using Distributions, LinearAlgebra, Printf, Plots

# =============================================================================
# 1. PARAMETERS
# =============================================================================
# The four visa routes. `rt(a,b,c,d)` builds a NamedTuple (work=a, care=b,
# study=c, family=d), so any route parameter is read as e.g. p.δ[:care].
const ROUTES = (:work, :care, :study, :family)
rt(v...) = NamedTuple{ROUTES}(v)

# All structural parameters in one NamedTuple. Change one with
# merge(P, (σ_nf = 1.0,)). Money values are £ per year.
const P = (
    # --- time: two periods of life ---
    Tw = 45.0, Tr = 18.0,              # working years, retirement years
    β = 0.98^30,                       # private discount factor between the two periods
    R = 1.03^30,                       # world gross return per generation (30 years)
    δsoc = 0.98^30,                    # SOCIAL discount factor across generations (welfare only)
    γ = 1.0,                           # relative risk aversion (1 = log utility)
    # --- technology ---
    α = 0.33,                          # capital share
    σ_lh = 2.0,                        # elasticity: graduates vs non-graduates
    σ_nf = 7.0,                        # elasticity: natives vs migrants in the same skill market
    economy = :soe,                    # :soe (small open, K mobile) or :closed
    # --- natives ---
    shH = 0.42, σn = 0.45,             # share with a degree; spread of log ability
    prem = 1.6, ybar = 39_000.0,       # targets: graduate premium, mean native pay (£)
    # --- migrants ---
    migr = 0.22, σm = 0.55,            # migrants per young native; spread of log ability
    share = rt(.30, .15, .35, .20),    # target route mix
    hi    = rt(.80, .35, .75, .35),    # share with a degree, by route
    δ     = rt(.80, .60, .60, .55),    # productivity on ARRIVAL vs a native (downgrading)
    ψ     = rt(.60, .30, .60, .50),    # share of that gap closed over working life (assimilation)
    h     = rt(.90, .15, .50, .35),    # share of a graduate migrant's career in graduate jobs
    part  = rt(.95, .95, .55, .60),    # participation
    stay  = rt(.55, .50, .30, .85),    # probability of staying into retirement
    nrpf  = rt(1.0, 1.0, 1.0, 0.5),    # share with no recourse to public funds
    # --- UK fiscal system ---
    p_hsv = 0.15,                      # income-tax progressivity: tax(y) = y − λ y^(1−p)
    τc = 0.12, τk = 0.20,              # effective VAT, capital-income tax
    θb = 0.31,                         # flat State Pension / mean native pay
    pens_mig = 25 / 35,                # migrants who stay: 25 of 35 NI years ⇒ partial pension
    ucB = 9_000.0, taper = 0.55,       # Universal Credit: maximum award, taper
    gy = 4_000.0, go = 9_000.0,        # public services per worker / per pensioner (NHS etc.)
    Gpure = 3_000.0,                   # pure public goods (defence, debt interest)
    bhome = 2_000.0,                   # old-age income abroad for returners
    nq = 21, T = 12)                   # quadrature nodes; generations in a transition

# Baseline (pre-2024) policy. fees = health surcharge + visa fee, per visa-year.
const POL0 = (thr_work = 26_200.0,     # Skilled Worker salary threshold
    thr_care = 20_960.0,               # care-worker threshold
    degree = false,                    # graduate-level jobs only?
    fees = rt(924.0, 734.0, 570.0, 994.0),
    isc  = rt(1_000.0, 1_000.0, 0.0, 0.0),   # Immigration Skills Charge (employer)
    pool = rt(1.0, 1.0, 1.0, 1.0),     # scale of each route (0 = route closed)
    settle = 5.0,                      # years to settlement (ILR)
    assim = 1.0)                       # multiplier on assimilation speed ψ

# Objects chosen by calibration. CES SHARE parameters a ∈ (0,1).
mutable struct Cal
    gbp::Float64                       # £ per model unit
    aMH::Float64                       # share of migrants in the graduate CES
    aML::Float64                       # share of migrants in the non-graduate CES
    aH::Float64                        # share of graduates in total labour
    β::Float64                         # private discount factor (calibrated if :closed)
    pool::Vector{Float64}              # applicant pool by route
end

# =============================================================================
# 2. ABILITY DISTRIBUTIONS (quantile quadrature)
# =============================================================================
# Gauss–Legendre nodes and weights on [-1,1], computed once (Golub–Welsch:
# eigenvalues of a tridiagonal matrix).
const GL = let n = P.nq
    E = eigen(SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:n-1]))
    (E.values, 2 .* E.vectors[1, :] .^ 2)
end

# Log-normal ability (mean 1) restricted to quantiles (ulo, 1). A salary
# threshold admits only ability above a cut-off, i.e. quantiles above ulo.
# Nodes move smoothly with ulo, so the equilibrium is smooth in the threshold.
function nodes(σ, ulo)
    u = ulo .+ (1 - ulo) .* (GL[1] .+ 1) ./ 2              # map [-1,1] → (ulo,1)
    (exp.(-σ^2 / 2 .+ σ .* quantile.(Normal(), u)),         # ability z at each node
     GL[2] .* (1 - ulo) ./ 2,                               # weights (sum = admitted share)
     u)                                                     # quantile of each node
end

# =============================================================================
# 3. AGENTS
# =============================================================================
# A Group = natives of one skill, or migrants of one route × education.
# Each of its 21 nodes is a "representative person" with weight mass·w[i].
mutable struct Group
    route::Symbol; educ::Symbol        # e.g. :work, :H
    mass::Float64                      # people admitted (per young native = 1)
    z::Vector{Float64}; w::Vector{Float64}; u::Vector{Float64}   # ability nodes, weights, quantiles
    hsh::Float64                       # share of labour supplied to the graduate market
    m::Float64                         # lifetime productivity vs a native of same skill
    part::Float64; stay::Float64       # participation; prob. of staying into retirement
    fees::Float64; isc::Float64        # fees, skills charge (model units, per working year)
    uc::Float64                        # share of working life eligible for Universal Credit
    pf::Float64                        # share of the full State Pension if staying
    y::Vector{Float64}; inc::Vector{Float64}; ucv::Vector{Float64}   # filled by solvers:
    s::Vector{Float64}; c1::Vector{Float64}                          # earnings, income, UC, saving, c1
end
newgroup(a...) = Group(a..., Float64[], Float64[], Float64[], Float64[], Float64[])

# Natives: graduates and non-graduates, ability rescaled to mean exactly 1.
function natives(p)
    z, w, u = nodes(p.σn, 0.0); z = z ./ sum(z .* w)
    [newgroup(:native, e, sh, copy(z), copy(w), copy(u), Float64(e === :H), 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0)
     for (e, sh) in ((:H, p.shH), (:L, 1 - p.shH))]
end

# Migrants. KEY MECHANISM: the threshold applies to the ENTRY wage δ·z·w,
# while the economy receives LIFETIME productivity m·z, with m = δ + (1−δ)ψ.
# A threshold on entry pay can therefore exclude people who would assimilate.
function migrants(p, pol, cal, wHM, wLM)
    G = Group[]
    for (i, r) in enumerate(ROUTES), e in (:H, :L)
        # size of the applicant pool after policy (closures, degree rule)
        pool = cal.pool[i] * pol.pool[r] * (e === :H ? p.hi[r] : 1 - p.hi[r])
        (r === :work && e === :L && pol.degree) && (pool = 0.0)
        # threshold → ability cut-off: δ·z·w_job − ISC ≥ threshold
        thr  = r === :work ? pol.thr_work : r === :care ? pol.thr_care : 0.0
        wjob = (r === :work && e === :H) ? wHM : wLM
        zcut = (thr + pol.isc[r]) / cal.gbp / (p.δ[r] * wjob)
        ulo  = thr > 0 ? cdf(Normal(), (log(zcut) + p.σm^2 / 2) / p.σm) : 0.0
        z, w, u = nodes(p.σm, ulo)
        # lifetime productivity, stay probability (falls if settlement takes longer)
        m    = p.δ[r] + (1 - p.δ[r]) * min(1.0, p.ψ[r] * pol.assim)
        stay = p.stay[r] * max(0.0, 1 - 0.15 * (pol.settle - 5) / 5)
        push!(G, newgroup(r, e, pool, z, w, u, e === :H ? p.h[r] : 0.0, m, p.part[r], stay,
              pol.fees[r] * 5 / p.Tw / cal.gbp,          # 5 visa years spread over working life
              pol.isc[r] * 5 / p.Tw / cal.gbp,
              1 - p.nrpf[r] * min(1.0, pol.settle / p.Tw),   # no UC before settlement
              p.pens_mig))
    end
    G
end

# Efficiency units supplied to the graduate (EH) and non-graduate (EL) markets.
function eff(G)
    EH = EL = 0.0
    for g in G
        x = g.mass * g.part * g.m * sum(g.w .* g.z)
        EH += x * g.hsh; EL += x * (1 - g.hsh)
    end
    EH, EL
end

# =============================================================================
# 4. FIRMS: nested CES in share form (FIX for σ = 1)
# =============================================================================
# X = [(1−a) x1^ρ + a x2^ρ]^(1/ρ), ρ = 1 − 1/σ. As σ → 1 this converges to the
# Cobb–Douglas x1^(1−a) x2^a, which we compute EXPLICITLY (the CES formula
# would divide by ρ = 0). Returns X and its derivatives ∂X/∂x1, ∂X/∂x2.
function ces(x1, x2, a, σ)
    if abs(σ - 1) < 1e-8                                   # Cobb–Douglas branch
        X = x1^(1 - a) * x2^a
        return X, (1 - a) * X / x1, a * X / x2
    end
    ρ = 1 - 1 / σ
    X = ((1 - a) * x1^ρ + a * x2^ρ)^(1 / ρ)
    return X, (1 - a) * (X / x1)^(1 - ρ), a * (X / x2)^(1 - ρ)
end

# Production: Y = K^α L^(1−α); L = CES(non-graduate, graduate; σ_lh);
# each skill = CES(natives, migrants; σ_nf). Wages = marginal products.
function tech(p, cal, K, NH, NL, MH, ML)
    LH, dLH_N, dLH_M = ces(NH, max(MH, 1e-12), cal.aMH, p.σ_nf)   # graduate market
    LL, dLL_N, dLL_M = ces(NL, max(ML, 1e-12), cal.aML, p.σ_nf)   # non-graduate market
    L,  dL_L,  dL_H  = ces(LL, LH, cal.aH, p.σ_lh)                 # total labour
    # small open economy: K adjusts so that the marginal product of capital = world return
    p.economy === :soe && (K = (p.α / p.R)^(1 / (1 - p.α)) * L)
    Y  = K^p.α * L^(1 - p.α)
    wl = (1 - p.α) * Y / L                                          # price of a unit of L
    (Y = Y, K = K, R = p.α * Y / K,
     wHN = wl * dL_H * dLH_N, wHM = wl * dL_H * dLH_M,              # chain rule
     wLN = wl * dL_L * dLL_N, wLM = wl * dL_L * dLL_M)
end

# Within-period equilibrium for given K: wages depend on who is admitted, and
# admission depends on wages (threshold) ⇒ iterate to a fixed point.
function period(p, pol, cal, K, wg)
    nat = natives(p); NH, NL = eff(nat); wHM, wLM = wg; mig = Group[]; t = nothing
    for _ in 1:1000
        mig = migrants(p, pol, cal, wHM, wLM)
        t = tech(p, cal, K, NH, NL, eff(mig)...)
        d = abs(t.wHM - wHM) + abs(t.wLM - wLM)
        wHM, wLM = (wHM + t.wHM) / 2, (wLM + t.wLM) / 2     # damped update (numerical step only)
        d < 1e-12 && break
    end
    G = vcat(nat, mig)
    for g in G                                              # gross earnings at each node
        wpu = g.route === :native ? (g.educ === :H ? t.wHN : t.wLN) : g.hsh * t.wHM + (1 - g.hsh) * t.wLM
        g.y = max.(g.part * g.m .* g.z .* wpu .- g.isc, 1e-6)
    end
    (G = G, t = t, wg = (wHM, wLM), ybar = sum(sum(g.mass .* g.w .* g.y) for g in nat))
end

# =============================================================================
# 5. HOUSEHOLDS (CRRA utility)
# =============================================================================
ω(p) = p.Tr / p.Tw                       # retirement length relative to working life
Rnet(p, R) = R - p.τk * (R - 1)          # after-tax gross return

u(c, γ)  = γ == 1 ? log(c) : (c^(1 - γ) - 1) / (1 - γ)   # period utility
mu(c, γ) = c^(-γ)                                        # marginal utility

# Budgets per year:  young (1+τc)c1 + s = inc
#                    old in UK   ω(1+τc)c2 = R̃ s + ω·pension
#                    old abroad  ω c2      = R̃ s + ω·b_home
# Euler equation: u'(c1)/(1+τc) = βR̃[stay·u'(c2_UK)/(1+τc) + (1−stay)·u'(c2_abroad)].
# Left side rises in s, right side falls ⇒ unique root ⇒ bisection.
function saving(inc, R̃, pens, stay, β, om, bh, τc, γ)
    c1(s)  = (inc - s) / (1 + τc)
    c2s(s) = (R̃ * s + om * pens) / (om * (1 + τc))
    c2l(s) = (R̃ * s + om * bh) / om
    f(s) = mu(c1(s), γ) / (1 + τc) - β * R̃ * (stay * mu(c2s(s), γ) / (1 + τc) + (1 - stay) * mu(c2l(s), γ))
    f(0.0) >= 0 && return 0.0                               # borrowing limit binds
    lo, hi = 0.0, inc * (1 - 1e-12)
    for _ in 1:100
        mid = (lo + hi) / 2; f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end

# Solve every node of every group; store income, UC, saving and c1.
function decide!(p, cal, G, λ, τni, R̃next, bnext)
    B, bh, om = p.ucB / cal.gbp, p.bhome / cal.gbp, ω(p)
    for g in G
        g.ucv = @. g.uc * max(0.0, B - p.taper * g.y)                            # Universal Credit
        g.inc = @. max(λ * g.y^(1 - p.p_hsv) - τni * g.y + g.ucv - g.fees, 1e-6) # after tax, NI, fees
        g.s   = [saving(x, R̃next, bnext * g.pf, g.stay, cal.β, om, bh, p.τc, p.γ) for x in g.inc]
        g.c1  = (g.inc .- g.s) ./ (1 + p.τc)
    end
end

# Per-year old-age consumption: (in the UK, abroad).
function c2(p, cal, g, R̃, b)
    om = ω(p); bh = p.bhome / cal.gbp
    (@. (R̃ * g.s + om * b * g.pf) / (om * (1 + p.τc))), (@. (R̃ * g.s + om * bh) / om)
end

# =============================================================================
# 6. GOVERNMENT: NI/pension budget and general budget, balanced each period
# =============================================================================
nold(g) = g.route === :native ? 1.0 : g.stay   # retirees in the UK per young person

function budget(p, cal, Y, O, R̃, R, b)          # Y = young groups, O = old groups
    om = ω(p); pm(g) = g.mass * sum(g.w)
    S(f) = sum(sum(g.mass .* g.w .* f(g)) for g in Y)                 # sum over young nodes
    Ysum = S(g -> g.y); Yp = S(g -> g.y .^ (1 - p.p_hsv))
    # (i) PAYGO pension: NI rate × earnings = pension × pensioners
    τni  = b * om * sum(pm(g) * nold(g) * g.pf for g in O) / Ysum
    # (ii) spending: public services by age + pure public goods
    popY = sum(pm(g) for g in Y); popO = om * sum(pm(g) * nold(g) for g in O)
    Gtot = (p.gy * popY + p.go * popO + p.Gpure) / cal.gbp
    # revenues other than income tax: VAT, capital tax, migrant fees and skills charge
    Co   = sum(g.mass * nold(g) * om * sum(g.w .* c2(p, cal, g, R̃, b)[1]) for g in O)
    ktax = sum(g.mass * nold(g) * p.τk * (R - 1) * sum(g.w .* g.s) for g in O)
    other = p.τc * (S(g -> g.c1) + Co) + ktax + sum(pm(g) * (g.fees + g.isc) for g in Y)
    # income tax = Ysum − λ·Yp ⇒ λ balances the budget (lower λ = higher taxes)
    λ = (Ysum + other - Gtot - S(g -> g.ucv)) / Yp
    τni, λ, (popY = popY, popO = popO)
end

# =============================================================================
# 7. STEADY STATE, CALIBRATION, TRANSITION
# =============================================================================
# Steady state: the old generation equals the young one.
function steady(p, pol, cal)
    K, λ, τ, wg = 0.15, 0.8, 0.1, (1.0, 0.6); c = nothing; acc = nothing; b = R̃ = 0.0
    for _ in 1:2000
        c = period(p, pol, cal, K, wg); wg = c.wg
        R̃ = Rnet(p, c.t.R); b = p.θb * c.ybar
        decide!(p, cal, c.G, λ, τ, R̃, b)
        τn, λn, acc = budget(p, cal, c.G, c.G, R̃, c.t.R, b)
        Kn = p.economy === :closed ? sum(sum(g.mass .* g.w .* g.s) for g in c.G) : c.t.K
        d = abs(λn - λ) + abs(τn - τ) + abs(Kn - K)
        λ, τ, K = (λ + λn) / 2, (τ + τn) / 2, (K + Kn) / 2   # damped (numerical step only)
        d < 1e-10 && break
    end
    (c = c, λ = λ, τ = τ, b = b, R̃ = R̃, R = c.t.R, K = K, acc = acc)
end

odds(a) = a / (1 - a); fromodds(o) = o / (1 + o)

# Calibration WITHOUT arbitrary exponents. For given labour quantities,
#   w_M/w_N  is proportional to odds(aM)   and   w_H/w_L  to odds(aH),
# so multiplying the odds by (target ratio / current ratio) hits the target
# exactly; admitted migrants are linear in the pool; £ per unit is exact.
# Only the closed-economy β has no closed form: we use the secant method.
function calibrate(p, pol)
    tgt = [p.migr * p.share[r] for r in ROUTES]
    cal = Cal(p.ybar / 0.7, 0.4, 0.4, 0.5, p.β, copy(tgt))
    βold = fold = NaN
    for _ in 1:500
        s = steady(p, pol, cal); t = s.c.t
        adm = [sum(g.mass * sum(g.w) for g in s.c.G if g.route === r) for r in ROUTES]
        fR  = log(s.R / p.R)                                           # closed: R should equal R_world
        err = abs(t.wHN / t.wHM - 1) + abs(t.wLN / t.wLM - 1) + abs(t.wHN / t.wLN - p.prem) +
              abs(s.c.ybar * cal.gbp / p.ybar - 1) + sum(abs.(adm ./ tgt .- 1)) +
              (p.economy === :closed ? abs(fR) : 0.0)
        err < 1e-8 && break
        cal.aMH = fromodds(odds(cal.aMH) * t.wHN / t.wHM)           # migrants paid like natives per unit
        cal.aML = fromodds(odds(cal.aML) * t.wLN / t.wLM)
        cal.aH  = fromodds(odds(cal.aH) * p.prem / (t.wHN / t.wLN))  # graduate premium
        cal.gbp = p.ybar / s.c.ybar                                  # mean native pay = £39k
        cal.pool .*= tgt ./ adm                                      # route mix
        if p.economy === :closed                                     # secant step on β
            βnew = isnan(βold) ? cal.β * 1.05 : cal.β - fR * (cal.β - βold) / (fR - fold)
            βold, fold, cal.β = cal.β, fR, max(βnew, 1e-3)
        end
    end
    cal
end

# Transition after an UNEXPECTED, PERMANENT policy change at t = 1 (perfect
# foresight afterwards). Retirees at t = 1 saved under the old regime (s0).
function transition(p, pol, cal, s0, s1)
    T = p.T; K = fill(s0.K, T)
    λ = collect(range(s0.λ, s1.λ; length = T)); τ = collect(range(s0.τ, s1.τ; length = T))
    p.economy === :closed && (K[2:end] .= range(s0.K, s1.K; length = T)[2:end])
    C = Vector{Any}(undef, T); R̃ = zeros(T + 1); b = zeros(T + 1); wg = s0.c.wg
    for _ in 1:3000
        for t in 1:T                                       # (a) prices, admitted migrants
            C[t] = period(p, pol, cal, K[t], wg); wg = C[t].wg
            R̃[t] = Rnet(p, C[t].t.R); b[t] = p.θb * C[t].ybar
        end
        R̃[T+1], b[T+1] = s1.R̃, s1.b
        for t in 1:T                                       # (b) households, perfect foresight
            decide!(p, cal, C[t].G, λ[t], τ[t], R̃[t+1], b[t+1])
        end
        d = 0.0
        for t in 1:T                                       # (c) budgets and capital
            O = t == 1 ? s0.c.G : C[t-1].G
            τn, λn, _ = budget(p, cal, C[t].G, O, R̃[t], C[t].t.R, b[t])
            d = max(d, abs(τn - τ[t]), abs(λn - λ[t]))
            τ[t] = (τ[t] + τn) / 2; λ[t] = (λ[t] + λn) / 2
            if p.economy === :closed && t < T
                Kn = sum(sum(g.mass .* g.w .* g.s) for g in C[t].G)
                d = max(d, abs(Kn - K[t+1])); K[t+1] = (K[t+1] + Kn) / 2
            end
        end
        d < 1e-9 && break
    end
    (C = C, R̃ = R̃, b = b)
end

# =============================================================================
# 8. WELFARE — computed directly from utility (no closed-form exponents)
# =============================================================================
# Lifetime utility of a native (natives always stay) from consumption (c1, c2);
# g scales consumption in BOTH periods by (1+g).
life(c1, c2, g, βω, γ) = u((1 + g) * c1, γ) + βω * u((1 + g) * c2, γ)

# Root-finder: h(g) is increasing in g, so bisection finds h(g) = 0.
function solve_g(h; lo = -0.95, hi = 5.0)
    for _ in 1:100
        mid = (lo + hi) / 2; h(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end

wavg(v, g) = sum(g.w .* v) / sum(g.w)                         # mean over ability nodes
natcons(p, cal, g, R̃, b) = (g.c1, c2(p, cal, g, R̃, b)[1])    # a native group's (c1, c2) by node

# Pareto weights λ_i on each native node:
#   :utilitarian — every person counts equally
#   :negishi     — λ = 1/u'(c): a £ is worth the same to everyone ⇒ the social
#                  CEV becomes a compensation test (cf. Busch et al.'s net gain ratio)
#   (H = a, L = b) — your own weights by skill, e.g. (H = 0.5, L = 1.5) favours non-graduates
function pareto(p, scheme, base, cb)
    scheme === :utilitarian && return [ones(length(g.w)) for g in base]
    scheme === :negishi     && return [1 ./ mu.(cb[k][1], p.γ) for k in 1:2]
    [fill(Float64(scheme[g.educ]), length(g.w)) for g in base]
end

function welfare(p, cal, s0, s1, tr; scheme = :utilitarian)
    βω, γ, om = cal.β * ω(p), p.γ, ω(p)
    base = s0.c.G[1:2]                                           # natives: 1 graduates, 2 non-graduates
    n    = [g.mass .* g.w for g in base]                         # people at each node
    cb   = [natcons(p, cal, g, s0.R̃, s0.b) for g in base]       # BASELINE consumption
    λw   = pareto(p, scheme, base, cb)                           # Pareto weights
    cnew = [[natcons(p, cal, tr.C[t].G[k], tr.R̃[t+1], tr.b[t+1]) for k in 1:2] for t in 1:p.T]
    cLR  = [natcons(p, cal, s1.c.G[k], s1.R̃, s1.b) for k in 1:2] # long run
    c2o  = [c2(p, cal, g, tr.R̃[1], tr.b[1])[1] for g in base]    # retirees at the shock

    # (a) INDIVIDUAL CEV: g such that U_base((1+g)·c) = U_new, node by node
    function indiv(cn, k)
        Un = life.(cn[k][1], cn[k][2], 0.0, βω, γ)
        gs = [solve_g(g -> life(cb[k][1][j], cb[k][2][j], g, βω, γ) - Un[j]) for j in eachindex(Un)]
        wavg(gs, base[k])
    end

    # (b) SOCIAL CEV of one generation: W = Σ λ_i n_i U_i. Find the UNIFORM g that,
    #     given to everyone in the baseline, yields the same W as the reform.
    Wb(g)  = sum(sum(λw[k] .* n[k] .* life.(cb[k][1], cb[k][2], g, βω, γ)) for k in 1:2)
    Wn(cn) = sum(sum(λw[k] .* n[k] .* life.(cn[k][1], cn[k][2], 0.0, βω, γ)) for k in 1:2)
    social(cn) = (W = Wn(cn); solve_g(g -> Wb(g) - W))

    # (c) INTERGENERATIONAL SWF: retirees at the shock (remaining utility ω·u(c2))
    #     plus generations 1..T discounted by δsoc; one uniform g for all of them.
    Wob(g) = sum(sum(λw[k] .* n[k] .* om .* u.((1 + g) .* cb[k][2], γ)) for k in 1:2)
    Won    = sum(sum(λw[k] .* n[k] .* om .* u.(c2o[k], γ)) for k in 1:2)
    SWFb(g) = Wob(g) + sum(p.δsoc^t * Wb(g) for t in 1:p.T)
    SWFn    = Won + sum(p.δsoc^t * Wn(cnew[t]) for t in 1:p.T)

    (old = [wavg(c2o[k] ./ cb[k][2] .- 1, base[k]) for k in 1:2],   # retirees: CEV = % change in c2
     gen = [indiv(cnew[t], k) for t in 1:p.T, k in 1:2],             # T×2: by generation and skill
     lr  = [indiv(cLR, k) for k in 1:2],
     social_gen = [social(cnew[t]) for t in 1:p.T],
     social_lr  = social(cLR),
     swf = solve_g(g -> SWFb(g) - SWFn))
end

# Theil index of earnings per worker (0 = equality).
function theil(G)
    m = reduce(vcat, [g.mass .* g.w .* g.part for g in G]); y = reduce(vcat, [g.y ./ g.part for g in G])
    k = m .> 0; m, y = m[k], y[k]; ȳ = sum(m .* y) / sum(m)
    sum(m .* (y ./ ȳ) .* log.(y ./ ȳ)) / sum(m)
end

# =============================================================================
# 9. UK POLICY SHOCKS (unexpected, permanent, from the pre-2024 baseline)
# =============================================================================
const IHS_UP = rt(411.0, 411.0, 306.0, 411.0)     # health surcharge £624→£1,035 (students £470→£776)
const ISC_UP = rt(1_320.0, 1_320.0, 0.0, 0.0)     # Skills Charge +32%
scen = [
    ("S1 threshold £41.7k+degree", merge(POL0, (thr_work = 41_700.0, degree = true))),
    ("S2 health surcharge up",     merge(POL0, (fees = map(+, POL0.fees, IHS_UP),))),
    ("S3 care route closed",       merge(POL0, (pool = rt(1.0, 0.0, 1.0, 1.0),))),
    ("S4 student dependants ban",  merge(POL0, (pool = rt(1.0, 1.0, 0.7, 1.0),))),
    ("S5 skills charge +32%",      merge(POL0, (isc = ISC_UP,))),
    ("S6 settlement 10 years",     merge(POL0, (settle = 10.0,))),
    ("S7 package S1-S5",           merge(POL0, (thr_work = 41_700.0, degree = true,
                                     fees = map(+, POL0.fees, IHS_UP), pool = rt(1.0, 0.0, 0.7, 1.0), isc = ISC_UP))),
    ("S8 faster integration",      merge(POL0, (assim = 1.5,))),
]

# =============================================================================
# 10. RUN AND REPORT
# =============================================================================
# Columns: migr% GDPpc% wLN% wHN% (changes), tax/NI (pp), individual CEVs (%):
# retirees at the shock, first generation, long run (H = graduates, L = non-
# graduates); then SOCIAL CEVs with the chosen Pareto weights: first generation,
# long run, and the intergenerational SWF.
function simulate(p, pol0, scen; scheme = :utilitarian)
    cal = calibrate(p, pol0); s0 = steady(p, pol0, cal); t0 = s0.c.t; ȳ = s0.c.ybar
    tax(λ)   = 100 * (1 - λ * ȳ^(-p.p_hsv))                   # avg income tax at baseline mean pay
    nmig(s)  = sum(g.mass * sum(g.w) for g in s.c.G if g.route !== :native)
    gdppc(s) = s.c.t.Y / (s.acc.popY + s.acc.popO)
    @printf("\nBaseline (%s, σ_nf=%.1f, weights=%s): mean pay £%.0f | income tax %.1f%% | NI %.1f%% | pension £%.0f | Theil %.3f\n",
            p.economy, p.σ_nf, string(scheme), ȳ * cal.gbp, tax(s0.λ), 100 * s0.τ, s0.b * cal.gbp, theil(s0.c.G))
    @printf("%-27s %6s %6s %6s %6s %5s %5s | %6s %6s | %6s %6s | %6s %6s | %6s %6s %6s\n", "scenario (CEV in %)",
            "migr", "GDPpc", "wLN", "wHN", "tax", "NI", "oldH", "oldL", "t1 H", "t1 L", "LR H", "LR L", "S t1", "S LR", "SWF")
    out = []
    for (name, pol) in scen
        s1 = steady(p, pol, cal); tr = transition(p, pol, cal, s0, s1)
        w = welfare(p, cal, s0, s1, tr; scheme = scheme); t1 = s1.c.t
        @printf("%-27s %6.2f %6.2f %6.2f %6.2f %5.2f %5.2f | %6.2f %6.2f | %6.2f %6.2f | %6.2f %6.2f | %6.2f %6.2f %6.2f\n",
                name, 100 * (nmig(s1) / nmig(s0) - 1), 100 * (gdppc(s1) / gdppc(s0) - 1),
                100 * (t1.wLN / t0.wLN - 1), 100 * (t1.wHN / t0.wHN - 1), tax(s1.λ) - tax(s0.λ), 100 * (s1.τ - s0.τ),
                100 * w.old[1], 100 * w.old[2], 100 * w.gen[1, 1], 100 * w.gen[1, 2],
                100 * w.lr[1], 100 * w.lr[2], 100 * w.social_gen[1], 100 * w.social_lr, 100 * w.swf)
        push!(out, (name = name, w = w))
    end
    out
end

# --- main specification: small open economy, utilitarian weights ---
res = simulate(P, POL0, scen)

# --- figure: welfare of natives by generation, 2024–25 package ---
w7  = res[7].w
fig = plot(1:P.T, 100 .* w7.gen; label = ["graduates" "non-graduates"], lw = 2,
           xlabel = "generation born after the shock", ylabel = "CEV (%)",
           title = "2024-25 package: welfare of natives by generation")
plot!(fig, 1:P.T, 100 .* w7.social_gen; label = "social (utilitarian)", lw = 2, ls = :dash, color = :black)
hline!(fig, [0.0]; color = :gray, label = "")
savefig(fig, "welfare_by_generation.png"); display(fig)

# --- Pareto weights: how much does the verdict depend on whose welfare counts? ---
simulate(P, POL0, scen[[1, 7]]; scheme = :negishi)              # compensation test
simulate(P, POL0, scen[[1, 7]]; scheme = (H = 0.5, L = 1.5))    # favour non-graduates

# --- Cobb–Douglas check (σ_nf = 1) now runs correctly ---
simulate(merge(P, (σ_nf = 1.0,)), POL0, scen[[1, 7]])

# --- closed economy (capital not internationally mobile) ---
simulate(merge(P, (economy = :closed,)), POL0, scen[[1, 3, 7]])
