// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;
typedef Eigen::Map<Eigen::VectorXd> VEC;
typedef Eigen::Map<Eigen::MatrixXd> MAT;

// [[Rcpp::export]]
NumericMatrix gradient0(NumericMatrix P_, NumericMatrix scrutinA, NumericMatrix scrutinB) {
  MAT A( as<MAT>(scrutinA) );
  MAT B( as<MAT>(scrutinB) );
  MAT P( as<MAT>(P_) );
  unsigned int n = A.cols();
  unsigned int m = B.cols();
  if(P.rows() != n || P.cols() != m)
    stop("in 'gradient', dimensions mismatch");

  Eigen::MatrixXd Q = A*P;
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n,m);
  for(unsigned int s = 0; s < A.rows(); s++) {
    for(unsigned int j = 0; j < m; j++) {
      for(unsigned int i = 0; i < n; i++) {
        if(B(s,j) > 0) // pour éviter le 0/0 -> NaN [ces termes n'apparaissent pas dans la vraisemblance!]
          G(i,j) += A(s,i) * B(s,j) / Q(s,j); 
      }
    }
  }
  return wrap(G);
}


// [[Rcpp::export]]
NumericMatrix gradient(NumericMatrix P_, NumericMatrix scrutinA, NumericMatrix scrutinB, int nThreads = 2) {
  MAT A( as<MAT>(scrutinA) );
  MAT B( as<MAT>(scrutinB) );
  MAT P( as<MAT>(P_) );
  unsigned int n = A.cols();
  unsigned int m = B.cols();
  if(P.rows() != n || P.cols() != m)
    stop("in 'gradient', dimensions mismatch");

  Eigen::MatrixXd Q = A*P;
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n,m); 
#pragma omp parallel num_threads(nThreads)
  {
    Eigen::MatrixXd G0 = Eigen::MatrixXd::Zero(n,m); // private to each thread
#pragma omp for
     for(unsigned int s = 0; s < A.rows(); s++) {
      for(unsigned int j = 0; j < m; j++) {
        if(B(s,j) > 0) { // pour éviter le 0/0 -> NaN [ces termes n'apparaissent pas dans la vraisemblance!]
          double co = B(s,j) / Q(s,j);
          for(unsigned int i = 0; i < n; i++) {
            G0(i,j) += A(s,i) * co; 
          }
        }
      }
    }
#pragma omp critical
    {
      for(unsigned int j = 0; j < m; j++) {
        for(unsigned int i = 0; i < n; i++) {
          G(i,j) += G0(i,j);
        }
      }
    }
  }

  return wrap(G);
}

// Calcul de Qsj au vol ; un plus rapide sur le portable
// [[Rcpp::export]]
NumericMatrix gradient2(NumericMatrix P_, NumericMatrix scrutinA, NumericMatrix scrutinB, int nThreads = 2) {
  MAT A( as<MAT>(scrutinA) );
  MAT B( as<MAT>(scrutinB) );
  MAT P( as<MAT>(P_) );
  unsigned int n = A.cols();
  unsigned int m = B.cols();
  if(P.rows() != n || P.cols() != m)
    stop("in 'gradient', dimensions mismatch");

  // Eigen::MatrixXd Q = A*P;
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n,m); 
#pragma omp parallel num_threads(nThreads)
  {
    Eigen::MatrixXd G0 = Eigen::MatrixXd::Zero(n,m); // private to each thread
#pragma omp for
    for(unsigned int s = 0; s < A.rows(); s++) {
      for(unsigned int j = 0; j < m; j++) {
        if(B(s,j) > 0) { // pour éviter le 0/0 -> NaN [ces termes n'apparaissent pas dans la vraisemblance!]
          double Qsj = 0;
          for(unsigned int k = 0; k < n; k++)
            Qsj += A(s,k)*P(k,j);
          double co = B(s,j) / Qsj;
          for(unsigned int i = 0; i < n; i++) {
            G0(i,j) += A(s,i) * co; 
          }
        }
      }
    }
#pragma omp critical
    {
      for(unsigned int j = 0; j < m; j++) {
        for(unsigned int i = 0; i < n; i++) {
          G(i,j) += G0(i,j);
        }
      }
    }
  }

  return wrap(G);
}

