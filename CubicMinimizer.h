#pragma once

#include "pch.h"

using namespace Eigen;

// Parameters and values useful for neighbours of a vertex
// the number of neighbours will be called V in the following comments
struct Neighbours {
	// list of the neighbours (size V)
	std::vector<Index> neighbour_indices;

	// cotangent weight of the neighbours
	// has V values (one for each neighbour)
	VectorXd cotangent_weights;

	// original distance from the point to each of its neighbour
	// 3xV matrix
	MatrixXd original_dist;
	// original distance from the point to each of its neighbour
	// 3xV matrix
	MatrixXd curr_dist;
};

class CubicMinimizer {
public:
	CubicMinimizer(MatrixXd* V, MatrixXi* F, double lambda);

	// return a 3xV (V is the number of neighbours) distance matrix
	MatrixXd compute_neighbour_dist(int vertex);

	// Run one step (local + global) of the algorithm
	void single_step();

	// Run the local step on the vertex 
	// find the rotation that minimizes the energy when points are fixed
	void local_step(int vertex);

	// Run the global step on the vertex
	// find the set of points that minimizes the energy given the rotations
	// it is described in 
	// O.Sorkine and M.Alexa. As-rigid-as-possible surface modeling.
	void global_step();

	// Return the energy of the mesh, the energy is the sum of the ARAP energy 
	// and weighed l_1 norm of rotated vectors
	double get_energy();

	MatrixXd* getV();
	VectorXd* getLambdas();

	// if set to true, does the polyhedral generalization
	bool set_polyhedral = false;

private:
	// number of vertices
	int n;
	// rotation matrix for each vertex
	std::vector<MatrixXd> rotations;
	// original vertices
	MatrixXd* oldV;
	// centroid of the mesh
	Vector3d centroid;
	// faces of the mesh
	MatrixXi* F;
	// newly computed (cubic) vertices
	MatrixXd currV;
	// neighbours of each vertex
	std::vector<Neighbours> neighbours;
	// saved rho value from the previous iteration
	std::vector<double> saved_rho;
	// saved dual value from the previous iteration
	std::vector<Vector3d> saved_u;
	// normals of each vertex
	MatrixXd normals;
	// cubeness factor, can be different for each vertex
	VectorXd lambda;
	// dual cell area of each vertex
	MatrixXd barycentric_areas;
	// laplacian-beltrami matrix
	SparseMatrix<double> laplacian;
	// Solver for the global step
	// using a cholesky decomposition for sparse positive semi-definite matrix
	// as recommended by the ARAP article
	SimplicialLDLT<SparseMatrix<double>> solver;
};