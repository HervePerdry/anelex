// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;
typedef Eigen::Map<Eigen::VectorXd> VEC;
typedef Eigen::Map<Eigen::MatrixXd> MAT;

// [[Rcpp::export]]
double likelihood(NumericMatrix P_, NumericMatrix scrutinA, NumericMatrix scrutinB, int nThreads = 2) {
  MAT A( as<MAT>(scrutinA) ); // S x n
  MAT B( as<MAT>(scrutinB) ); // S x m
  MAT P( as<MAT>(P_) );       // n x m
  unsigned int n = A.cols();
  unsigned int m = B.cols();
  if(P.rows() != n || P.cols() != m)
    stop("in 'likelihood', dimensions mismatch");

  Eigen::MatrixXd Q = A*P;    // S x m
  double L = 0;
#pragma omp parallel num_threads(nThreads)
#pragma omp for reduction(+:L)
  for(unsigned int s = 0; s < A.rows(); s++) {
    for(unsigned int j = 0; j < m; j++) {
      if(B(s,j) > 0) // si B(s,j) = 0 on peut avoir Q(s,j) = 0 ; 0 log(0) = 0...
        L += B(s,j) * log(Q(s,j));
    }
  }
  return L; 
}

// Calcul de Qsj au vol ; plus rapide sur le portable
// [[Rcpp::export]]
double likelihood2(NumericMatrix P_, NumericMatrix scrutinA, NumericMatrix scrutinB, int nThreads = 2) {
  MAT A( as<MAT>(scrutinA) ); // S x n
  MAT B( as<MAT>(scrutinB) ); // S x m
  MAT P( as<MAT>(P_) );       // n x m
  unsigned int n = A.cols();
  unsigned int m = B.cols();
  if(P.rows() != n || P.cols() != m)
    stop("in 'likelihood', dimensions mismatch");

  // Eigen::MatrixXd Q = A*P;    // S x m
  double L = 0;
#pragma omp parallel num_threads(nThreads)
#pragma omp for reduction(+:L)
  for(unsigned int s = 0; s < A.rows(); s++) {
    for(unsigned int j = 0; j < m; j++) {
      if(B(s,j) > 0) { // si B(s,j) = 0 on peut avoir Q(s,j) = 0 ; 0 log(0) = 0...
        double Qsj = 0;
        for(unsigned int k = 0; k < n; k++) 
          Qsj += A(s,k)*P(k,j);
        L += B(s,j) * log(Qsj);
      }
    }
  }
  return L; 
}

