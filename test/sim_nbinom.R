# -*- coding: utf-8 -*-

size <- 5
mu <- 8

se.size.sum <- 0
se.mu.sum <- 0
n.sim <- 1000
n.rnd <- 10000
for (i in 1:n.sim) {
  set.seed(i)
  cat("iteration:")
  print(i)
  
  prob <- size / (size + mu)
  rnd <- rnbinom(size=size,prob=prob,n=n.rnd)
  
  library(MASS)
  res <- fitdistr(rnd, "Negative Binomial")
  se.size <- res$sd["size"] / sqrt(n.rnd)
  se.mu <- res$sd["mu"] / sqrt(n.rnd)
  cat("se.size:")
  print(se.size)
  cat("se.mu")
  print(se.mu)
  
  se.size.sum <- se.size.sum + se.size
  se.mu.sum <- se.mu.sum + se.mu
}

se.size.mean <- se.size.sum / n.sim
se.mu.mean <- se.mu.sum / n.sim

print(se.size.mean)
print(se.mu.mean)

# > print(se.size.mean)
#        size 
# 0.001178097 
# > print(se.mu.mean)
#           mu 
# 0.0004559864 
