# Maximum Likelihood Estimation for Gaussian Process
# Jura Dataset - Zinc Concentration Analysis
# Model: C(x,y) = σ² exp(-||s(x-y)||^α) + τ² δ(x,y)
# where α ∈ (0,2] and s,σ,τ > 0

library(gstat)
library(sp)

# Load jura dataset and extract zinc data
data(jura)
jura <- prediction.dat
zinc_raw <- jura$Zn
coords <- jura[, c("Xloc", "Yloc")]

# Transform data: log-transform and center
zinc_log <- log(zinc_raw)
zinc_transformed <- zinc_log - mean(zinc_log)

cat("Number of observations:", length(zinc_transformed), "\n")
cat("Variance of transformed data:", var(zinc_transformed), "\n")

# Covariance matrix function
covariance_matrix <- function(coords, params) {
  alpha <- params[1]; s <- params[2]; sigma <- params[3]; tau <- params[4]
  dist_matrix <- as.matrix(dist(coords))
  C <- sigma^2 * exp(-(s * dist_matrix)^alpha)
  diag(C) <- sigma^2 + tau^2
  return(C)
}

# Log-likelihood function
log_likelihood <- function(params, data, coords) {
  alpha <- params[1]; s <- params[2]; sigma <- params[3]; tau <- params[4]
  
  # Parameter constraints
  if (alpha <= 0 || alpha > 2 || s <= 0 || sigma <= 0 || tau < 0) return(-Inf)
  
  # Compute covariance matrix
  C <- covariance_matrix(coords, params)
  diag(C) <- diag(C) + 1e-8  # Numerical stability
  
  # Cholesky decomposition for efficiency
  chol_result <- tryCatch(chol(C), error = function(e) NULL)
  if (is.null(chol_result)) return(-Inf)
  
  # Log-likelihood computation
  log_det_C <- 2 * sum(log(diag(chol_result)))
  alpha_vec <- backsolve(chol_result, forwardsolve(t(chol_result), data))
  quadratic_form <- sum(data * alpha_vec)
  
  return(-0.5 * (log_det_C + quadratic_form))
}

# Initial parameter values
empirical_var <- var(zinc_transformed)
coord_range <- apply(coords, 2, function(x) diff(range(x)))
initial_params <- c(alpha = 1.0, 
                   s = 1 / (mean(coord_range) / 4),
                   sigma = sqrt(empirical_var * 0.7),
                   tau = sqrt(empirical_var * 0.3))

# Optimization
opt_result <- optim(
  par = initial_params,
  fn = function(params, data, coords) -log_likelihood(params, data, coords),
  data = zinc_transformed,
  coords = coords,
  method = "L-BFGS-B",
  lower = c(0.1, 1e-6, 1e-6, 0),
  upper = c(2.0, Inf, Inf, Inf),
  control = list(maxit = 1000)
)

# Extract results
mle_params <- opt_result$par
names(mle_params) <- c("alpha", "s", "sigma", "tau")

cat("Convergence:", ifelse(opt_result$convergence == 0, "SUCCESS", "FAILED"), "\n")
cat("Log-likelihood:", -opt_result$value, "\n")
cat("Estimated Parameters:\n")
for (i in 1:length(mle_params)) {
  cat(sprintf("  %s = %.6f\n", names(mle_params)[i], mle_params[i]))
}

# Model validation
fitted_cov <- covariance_matrix(coords, mle_params)
min_eigenval <- min(eigen(fitted_cov, only.values = TRUE)$values)

# Calculate spatial parameters
effective_range <- (log(20))^(1/mle_params["alpha"]) / mle_params["s"]
sill <- mle_params["sigma"]^2
nugget <- mle_params["tau"]^2

cat("\nModel validation:\n")
cat("  Positive definite covariance:", min_eigenval > 0, "\n")
cat("  Sill (σ²):", sill, "\n")
cat("  Nugget (τ²):", nugget, "\n")
cat("  Effective range:", effective_range, "\n")
cat("  AIC:", 2 * length(mle_params) - 2 * (-opt_result$value), "\n")

# Plot theoretical variogram
distances <- seq(0, max(dist(coords)), length.out = 100)
theoretical_variogram <- sill * (1 - exp(-(mle_params["s"] * distances)^mle_params["alpha"]))

plot(distances, theoretical_variogram, type = "l", col = "red", lwd = 2,
     xlab = "Distance", ylab = "Semi-variance", 
     main = "Fitted Variogram Model")
abline(h = sill, col = "blue", lty = 2)
legend("bottomright", legend = c("Fitted variogram", "Sill"), 
       col = c("red", "blue"), lty = c(1, 2))

"""
MAXIMUM-LIKELIHOOD-SCHÄTZUNG ERGEBNISSE:

PARAMETERSCHÄTZUNGEN:
  α (alpha) = 0.898566  [Formparameter, α < 1 → nicht-differenzierbarer Prozess]
  s         = 5.291970  [Skalenparameter, kontrolliert Korrelationsabfall]  
  σ (sigma) = 0.354362  [Standardabweichung des räumlichen Prozesses]
  τ (tau)   = 0.097395  [Nugget-Standardabweichung (Messfehler)]

RÄUMLICHE STRUKTUR:
  - Sill (σ²) = 0.1256     [Räumliche Varianzkomponente]
  - Nugget (τ²) = 0.0095   [Nicht-räumliche Varianz] 
  - Gesamtvarianz = 0.1351 [Nahe der empirischen Varianz: 0.1496]
  - Nugget/Sill-Verhältnis = 7.6% [Sehr geringer Messfehler]
  - Effektive Reichweite = 0.641  [Distanz bei der Korrelation ≈ 0.05]

MODELLQUALITÄT:
  - Log-Likelihood = 210.69
  - AIC = -413.38

INTERPRETATION:
  Das angepasste Modell zeigt starke räumliche Korrelation der 
  Zinkkonzentrationen mit rapidem Abfall (hoher s-Parameter). 
  Der Formparameter deutet auf nicht-differenzierbare räumliche Variation hin. 
  Der geringe Nugget-Effekt lässt auf minimale Messfehler schließen, was bedeutet, 
  dass die meiste Variation räumlich strukturiert und nicht zufälliges Rauschen ist.
"""