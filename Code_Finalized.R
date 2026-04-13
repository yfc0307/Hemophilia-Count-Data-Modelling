# Hong Kong Metropolitan University
# COMP S461F Data Science Project
# 
# Finalized in April 2026
# 
# Code by Chen Yufan


# Step 1: Import all necessary libraries
library(ggplot2)
library(corrplot)
library(dplyr)
library(stats)
library(AER)
library(MASS)
library(pscl)
library(COMPoissonReg)


# Step 2: Import data
data <- read.csv("HEM.csv", sep=";")
data <- data.frame(death = data$deaths, 
                   hiv = data$hiv, 
                   factor = data$factor, 
                   age = data$age, 
                   py = data$py)


# Step 3: Set mid-point value to age
data$age <- data$age * 5 - 3


# Step 4: Remove row 378 where py is 0
zero_indices <- which(data$py <= 0)
if (zero_indices == 378) {
  data <- data[-378, ] # Row 378 contains py = 0 which is unreasonable, identify as problematic data
  zero_indices <- which(data$py <= 0)
}


# Step 5: Spearman Correlation Matrix
cor_matrix <- cor.mtest(data, conf.level = .95)
par(family="serif")
par(cex = 1.5)
par(mar = c(0, 0, 3.5, 0))
corrplot(
  cor(data),
  method = "color",
  type = "upper",
  tl.cex = 0.7,
  tl.col = "black",
  tl.srt = 45,
  p.mat = cor_matrix$p,
  insig = "label_sig",
  sig.level = c(.001, .01, .05),
  pch.cex = 0.8,
  pch.col = "red",
  addCoef.col = "black",
  mar = c(0, 0, 2, 0)
)
title("Spearman Correlation Heatmap", line = 2)


# Step 6: Convert hiv and factor to categorical
data$hiv <- as.factor(ifelse(data$hiv ==1, 'negative', 'positive'))
data$factor <- as.factor(ifelse(data$factor ==1, 'high',
                                ifelse(data$factor ==2, 'moderate', 
                                       ifelse(data$factor ==3, 'low', 
                                              ifelse(data$factor ==4, 'unknown', 
                                                     'none')))))


# Step 7: Poisson Regression
model_poisson_full <- glm(death ~ hiv + factor + age + offset(log(py)), family="poisson", data=data)


# Step 8: Overdispersion Test
dispersiontest(model_poisson_full, trafo=2)


# Step 9: Negative Binomial Regression (NB)
model_nb_full <- glm.nb(death ~ hiv + factor + age + offset(log(py)), data=data)


# Step 10: Conway-Maxwell Poisson Regression (CMP)
model_cmp_full <- glm.cmp(
  death ~ hiv + factor + age + offset(log(py)),
  formula.nu = ~ 1,
  formula.p = NULL,
  data = data,
  control = get.control(
    optim.method = "BFGS",
    hybrid.tol = 1e-4,
    truncate.tol = 1e-4,
    ymax = 200
  )
)


# Step 11: Zero-Inflated Poisson Regression (ZIP)
model_zip_full_0 <- zeroinfl(death ~ hiv + factor + age + offset(log(py)) | hiv + factor + age + offset(log(py)), data = data)
model_zip_1 <- zeroinfl(death ~ hiv + factor + age + offset(log(py)) | hiv + age + offset(log(py)), data = data)


# Step 12: Zero-Inflated Negative Binomial Regression (ZINB)
model_zinb_full_0 <- zeroinfl(death ~ hiv + factor + age + offset(log(py)) | hiv + factor + age + offset(log(py)), data = data, dist = "negbin")
model_zinb_1 <- zeroinfl(death ~ hiv + factor + age + offset(log(py)) | hiv + age + offset(log(py)), data = data, dist = "negbin")


# Step 13: Zero-Inflated Conway-Maxwell Poisson Regression (ZICMP)
model_zicmp_full <- glm.cmp(
  death ~ hiv + factor + age + offset(log(py)),
  formula.nu = ~ 1,
  formula.p = ~ hiv + factor + age + offset(log(py)),
  data = data,
  control = get.control(
    optim.method = "BFGS",
    hybrid.tol = 1e-6,
    truncate.tol = 1e-6,
    ymax = 500
  )
)
model_zicmp_1 <- glm.cmp(
  death ~ hiv + factor + age + offset(log(py)),
  formula.nu = ~ 1,
  formula.p = ~ hiv  + age + offset(log(py)),
  data = data,
  control = get.control(
    optim.method = "BFGS",
    hybrid.tol = 1e-6,
    truncate.tol = 1e-6,
    ymax = 500
  )
)


# Step 14: Model Summaries
summary(model_poisson_full)
summary(model_nb_full)
summary(model_cmp_full)
summary(model_zip_full_0)
summary(model_zinb_full_0)
summary(model_zicmp_full)
summary(model_zip_1)
summary(model_zinb_1)
summary(model_zicmp_1)


# Step 15: Log-likelihood
models <- list(
  model_poisson_full, model_nb_full, model_cmp_full,
  model_zip_full_0, model_zinb_full_0, model_zicmp_full,
  model_zip_1, model_zinb_1, model_zicmp_1
)
model_names <- c("poisson_full", "nb_full", "cmp_full",
                 "zip_full", "zinb_full", "zicmp_full",
                 "zip_1", "zinb_1", "zicmp_1")
loglik_values <- sapply(models, logLik)
loglik_table <- data.frame(
  Model = model_names,
  loglik = loglik_values
)
print(loglik_table[order(-loglik_table$loglik),])


# Step 16: AIC
aic_values <- sapply(models, AIC)
aic_table <- data.frame(
  Model = model_names,
  AIC = aic_values
)
print(aic_table[order(aic_table$AIC),])


# Step 17: BIC
bic_values <- sapply(models, BIC)
bic_table <- data.frame(
  Model = model_names,
  BIC = bic_values
)
print(bic_table[order(bic_table$BIC),])


# Step 18: Visualizing Prediction of P, NB, ZIP-1, ZINB-1
means_p <- predict(model_poisson_full, type = "response")
means_nb <- predict(model_nb_full, type = "response")
means_zip <- predict(model_zip_1, type = "response")
means_zinb <- predict(model_zinb_1, type = "response")
zero_prob_zip <- predict(model_zip_1, type = "zero")
zero_prob_zinb <- predict(model_zinb_1, type = "zero")
count_mean_zip <- predict(model_zip_1, type = "count")
count_mean_zinb <- predict(model_zinb_1, type = "count")
dzip_correct <- function(x, lambda, pi) {
  # x: count value (scalar)
  # lambda: Poisson mean (vector)
  # pi: zero-inflation probability (vector)
  result <- numeric(length(lambda))
  for(i in 1:length(lambda)) {
    if(x == 0) {
      result[i] <- pi[i] + (1 - pi[i]) * dpois(x, lambda[i])
    } else {
      result[i] <- (1 - pi[i]) * dpois(x, lambda[i])
    }
  }
  return(result)
}
dzinb_correct <- function(x, mu, theta, pi) {
  # x: count value (scalar)
  # mu: NB mean (vector)
  # theta: dispersion parameter (scalar from model)
  # pi: zero-inflation probability (vector)
  result <- numeric(length(mu))
  for(i in 1:length(mu)) {
    if(x == 0) {
      result[i] <- pi[i] + (1 - pi[i]) * dnbinom(x, mu = mu[i], size = theta)
    } else {
      result[i] <- (1 - pi[i]) * dnbinom(x, mu = mu[i], size = theta)
    }
  }
  return(result)
}
pred_p <- colSums(sapply(0:6, function(x) dpois(x, means_p)))
pred_nb <- colSums(sapply(0:6, function(x) dnbinom(x, mu = means_nb, size = model_nb_full$theta)))
pred_zip <- sapply(0:6, function(y) {
  sum(dzip_correct(y, lambda = count_mean_zip, pi = zero_prob_zip))
})
pred_zinb <- sapply(0:6, function(y) {
  sum(dzinb_correct(y, mu = count_mean_zinb, theta = model_zinb_1$theta, 
                    pi = zero_prob_zinb))
})
pred_data <- data.frame(
  death = 0:6,
  observed = c(1832, 212, 62, 28, 6, 2, 1),
  pred_p = round(pred_p),
  pred_nb = round(pred_nb),
  pred_zip = round(pred_zip),
  pred_zinb = round(pred_zinb)
)
print(pred_data)


# Utilities
summary(data)
table(data$death)
table(data$hiv)
table(data$factor)
table(data$age)
table(data$py)