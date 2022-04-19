fit.modele <- function(scrutin.a, scrutin.b, P, eps.lik = 1e-3, eps.P = 1e-4, K = 5, n.threads = 4, lsx = 10, verbose = TRUE) {  
  # on passe le A en proportions
  A <- as.matrix(scrutin.a)
  A <- A/rowSums(A)

  B <- as.matrix(scrutin.b)

  sA <- colSums(A)
  sB <- colSums(B)

  if(missing(P)) {
    # P <- matrix( sB / sum(sB), byrow = TRUE, nrow = ncol(A), ncol = ncol(B), dimnames = list(colnames(A),colnames(B)) )
    P <- matrix(rexp(ncol(A)*ncol(B)),  nrow = ncol(A), ncol = ncol(B), dimnames = list(colnames(A),colnames(B)) )
    P <- P/rowSums(P)
  } else {
    P <- P/rowSums(P)
    P <- t(apply(P, 1, constr.mat));
  }

  # if(!exists("PP")) PP <<- list()
  # PP <<- c(PP, list(P))

  # deux fonctions pour la recherche linéaire
  # Gc = la direction de recherche
  lin.search <- function(alpha, Gc) {
    P1 <- P + alpha * Gc;
    P1 <- t(apply(P1, 1, constr.mat));
    likelihood2(P1, A, B, n.threads)
  }
  # idem mais on ne modifie que la ligne pointée par 'index'
  lin.search.2 <- function(alpha, index, Gc) {
    P1 <- P
    P1[index,] <- constr.mat(P1[index,] + alpha * Gc[index,])
    likelihood2(P1, A, B, n.threads)
  }

  Li <- likelihood2(P, A, B, n.threads)
  Li.prev <- Li - eps.lik - 1
  P.prev <- 2 + eps.P
  k <- 1

  ### itérations ###
  while(TRUE) {
    P.prev <- P
    Li.prev <- Li

    if(verbose) {
      if(k %% K != 0) 
        cat(sprintf("likelihood = %f    ", Li), "\n")
      else
        cat(sprintf("likelihood = %f [*]", Li), "\n")
    
      print(round(P*100,1))
      cat(paste0(rep("\033[F",nrow(P)+2), collapse = "")) # go up
    }

    G <- gradient2(P, A, B, n.threads)
    # rescaling du gradient (c'est un peu heuristique)
    Gc <- G/sA
    Gc <- (G - rowMeans(G))

    if(k %% K != 0) { 
      # linear search . m = distance max [on bouge pas de plus de 10% en une étape]
      m <- 0.1/max(abs(Gc))
      op <- optimise(lin.search, c(0, m), maximum = TRUE, Gc = Gc, tol = m/10)
      if( lin.search(0, Gc) <= op$objective ) {
        P <- P + Gc * op$maximum
         P <- t(apply(P, 1, constr.mat))
        Li <- likelihood2(P, A, B, n.threads)
      }
    } else { 
      # toutes les K étapes on fait ligne à ligne
      for(i in 1:nrow(P)) {
        m <- 0.1/max(abs(Gc[i,]))
        op <- optimise(lin.search.2, c(0, m), maximum = TRUE, index = i, Gc = Gc, tol = m/10)
        if( lin.search.2(0, i, Gc) <= op$objective ) {
          P[i,] <- constr.mat(P[i,] + Gc[i,] * op$maximum)
        }
      }
      Li <- likelihood2(P, A, B, n.threads)
      if( (Li - Li.prev < eps.lik) & max(abs(P - P.prev)) < eps.P ) break;
    }
    # PP <<- c(PP, list(P))
    k <- k+1
  }
  if(verbose) cat(paste0(rep("\n", nrow(P)+2), collapse = ""));
  return( list(P = P, likelihood = Li) );
}

