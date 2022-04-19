# une fonction à appliquer à chaque ligne d'une matrice P pour
# que tous les coeffs restent positifs
constr.mat <- function(x) {
  if(all(x >= 0)) return(x)
  m <- sum( x[x < 0] )
  w <- which(x <= 0)
  x[w] <- 0
  x[-w] <- x[-w] + m / (length(x) - length(w))
  return(constr.mat(x))
}
