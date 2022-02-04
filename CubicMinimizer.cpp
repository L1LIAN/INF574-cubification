#include "CubicMinimizer.h"

#include "eigen-qp.h"

using std::vector;

CubicMinimizer::CubicMinimizer(MatrixXd* V, MatrixXi* F, double lambda) {
	this->n = V->rows();
	this->oldV = V;
	this->F = F;
	this->lambda = lambda * VectorXd::Ones(n);
	this->centroid = V->colwise().mean();
	rotations.resize(n);
	// initialize all rotations matrix to identity
	for (int i = 0; i < n; i++) {
		rotations[i] = MatrixXd::Identity(3, 3);
	}
	// initialize all points to their original self
	currV = *V;

	igl::per_vertex_normals(currV, *F, normals);

	igl::cotmatrix(currV, *F, laplacian);
	solver.compute(-laplacian);

	// retrieve the neighbours of each vertex
	vector<vector<Index>> vertex_neighbours;
	igl::adjacency_list(*F, vertex_neighbours);

	saved_rho.resize(n, 1e-4);
	saved_u.resize(n, Vector3d::Zero());
	
	neighbours.resize(n);
	for (int i = 0; i < n; i++) {
		// for each vertex, compute its dist matrix and cotangent weights
		int nb_neighbours = vertex_neighbours[i].size();
		neighbours[i].neighbour_indices = vertex_neighbours[i];
		neighbours[i].original_dist = compute_neighbour_dist(i);
		//neighbours[i].curr_dist = neighbours[i].original_dist;
		neighbours[i].cotangent_weights = VectorXd::Zero(nb_neighbours);

		for (int j = 0; j < nb_neighbours; j++) {
			neighbours[i].cotangent_weights(j) = laplacian.coeff(i, vertex_neighbours[i][j]);
		}
	}

	SparseMatrix<double> mass_matrix;
	igl::massmatrix(currV, *F, igl::MassMatrixType::MASSMATRIX_TYPE_BARYCENTRIC, mass_matrix);
	barycentric_areas = mass_matrix.diagonal();
}

MatrixXd CubicMinimizer::compute_neighbour_dist(int vertex)
{
	int nb_neighbours = neighbours[vertex].neighbour_indices.size();
	MatrixXd dists = MatrixXd::Zero(3, nb_neighbours);
	for (int i = 0; i < nb_neighbours; i++) {
		int neighbour = neighbours[vertex].neighbour_indices[i];
		dists.col(i) = (currV.row(neighbour) - currV.row(vertex)).transpose();
	}
	return std::move(dists);
}

void CubicMinimizer::single_step()
{
	/*
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		local_step(i);
	}*/
	igl::parallel_for(n, [this](int vertex) {
		local_step(vertex);
		});

	global_step();
}

void CubicMinimizer::local_step(int vertex)
{
	constexpr double eps_abs = 1e-5;
	constexpr double eps_rel = 1e-3;
	constexpr double mu = 10;
	constexpr double tau_incr = 2;
	constexpr double tau_decr = 2;


	Neighbours &neighbour = neighbours[vertex];

	// compute the dist matrix
	neighbour.curr_dist = compute_neighbour_dist(vertex);

	MatrixXd rot = rotations[vertex];
	Vector3d z = rot * normals.row(vertex).transpose();
	Vector3d u = saved_u[vertex];
	double rho = saved_rho[vertex];

	int nb_neighbours = neighbour.neighbour_indices.size();
	// the part of M that is fixed at every iteration
	Matrix3d M_fixed = neighbour.original_dist * neighbour.cotangent_weights.asDiagonal() * neighbour.curr_dist.transpose();
	for (int it = 0; it < 100; it++) {
		// orthogonal procrustes, eq 4
		Matrix3d M = M_fixed + normals.row(vertex).transpose() * rho * (z - u).transpose();
		Matrix3d U, V;
		Vector3d S;
		igl::svd3x3(M, U, S, V);

		// no need to check the determinant, svd3x3 gives only matrices with positive determinant
		rot = V * U.transpose();

		Vector3d newz;

		if(this->set_polyhedral){
			// hard define the matrix B
			Matrix<double, 4, 3> B;
			double ir3 = 1 / sqrt(3);
			B << ir3, ir3, ir3,
				ir3, ir3, -ir3,
				ir3, -ir3, ir3,
				-ir3, ir3, ir3;
			// define Q, c, A, b like defined in the paper
			Matrix<double, 3+4, 3+4> Q = Matrix<double, 3 + 4, 3 + 4>::Zero();
			Q.block(0, 0, 3, 3) = rho * Matrix3d::Identity();
			Matrix<double, 3+4, 1> c = Matrix<double, 3+4, 1>::Zero();
			c.block(0, 0, 3, 1) = -rho*(rotations[vertex] * normals.row(vertex).transpose() + u);
			c.block(3, 0, 4, 1) = lambda(vertex) * barycentric_areas(vertex) * Vector4d::Ones();
			Matrix<double, 2 * 4, 3 + 4> A = Matrix<double, 2 * 4, 3 + 4>::Zero();
			A.block(0, 0, 4, 3) = B;
			A.block(4, 0, 4, 3) = -B;
			A.block(3, 0, 4, 4) = -Matrix4d::Identity();
			A.block(3, 4, 4, 4) = -Matrix4d::Identity();
			Matrix<double, 4 * 2, 1> b = Matrix<double, 4 * 2, 1>::Zero();
			Matrix<double, 3 + 4, 1> x;
			// we can call the QP solver!
			EigenQP::quadprog<double, 7, 8>(Q, c, A, b, x);
			newz = x.block(0, 0, 3, 1);
		}
		else {
			// lasso problem, eq 5
			double k = lambda(vertex) * barycentric_areas(vertex) / rho;
			newz = rot * normals.row(vertex).transpose() + u;
			Vector3d coeff_factor = Vector3d::Ones() - k * newz.cwiseAbs().cwiseInverse();
			coeff_factor = coeff_factor.cwiseMax(0);
			newz = newz.cwiseProduct(coeff_factor);
		}

		// eq 6, update dual
		u += rot * normals.row(vertex).transpose() - newz;

		// this part is according to Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers
		// by S.Boyd, N.Parikh, E.Chu, B.Peleato, J.Eckstein
		// however, it describes the algorithm working on x in R^n, but here x = R_i in SO3
		double primal_residual = (newz - rot * normals.row(vertex).transpose()).norm();
		// I am not sure about this part, according to the article there should be a ni somewhere
		double dual_residual = rho * (newz - z).norm();

		double factor = 1.0;
		if (primal_residual > mu * dual_residual) {
			factor = tau_incr;
		}
		else if (dual_residual > mu * primal_residual) {
			factor = 1 / tau_decr;
		}
		rho *= factor;

		double primal_eps = sqrt(3) * eps_abs + eps_rel * std::max((rot * normals.row(vertex).transpose()).norm(), newz.norm());
		// same as before, it is kind of hard to adapt what was written in the paper to this case
		double dual_eps = sqrt(3) * eps_abs + eps_rel * rho * u.norm();
		if (primal_residual < primal_eps && dual_residual < dual_eps) {
			// we can stop
			break;
		}

		z = newz;
	}
	rotations[vertex] = rot;
	saved_rho[vertex] = rho;
	saved_u[vertex] = u;
}

void CubicMinimizer::global_step()
{
	// We need to solve Lp' = b
	// where L is the laplacian-beltrami operator 
	// and b is a vector dependant of the rotations
	// this needs to be done 3 times for each coordinate
	MatrixXd b = MatrixXd::Zero(n, 3);
	for (int i = 0; i < n; i++) {
		// for each neighbour
		for (int k = 0; k < neighbours[i].neighbour_indices.size(); k++) {
			// j is the neighbour
			int j = neighbours[i].neighbour_indices[k];
			b.row(i) += laplacian.coeff(i, j) / 2
				* ((rotations[i] + rotations[j])
				* -neighbours[i].original_dist.col(k)).transpose();
		}
	}
	// iterate 3 times (3 dimensions)
	for (int k = 0; k < currV.cols(); k++) {
		currV.col(k) = solver.solve(b.col(k));
	}
	// we need to center the result
	Vector3d new_centroid = currV.colwise().mean();
	Vector3d translation = centroid - new_centroid;
	currV.rowwise() += translation.transpose();
}

double CubicMinimizer::get_energy()
{
	double energy = 0.0;

	for (int v = 0; v < n; v++) {
		Neighbours& neighbour = neighbours[v];
		// l_1 norm
		energy += lambda(v) * barycentric_areas(v) * (rotations[v] * normals.row(v).transpose()).cwiseAbs().sum();
		// ARAP energy
		// this is repetitive
		neighbour.curr_dist = compute_neighbour_dist(v);
		MatrixXd dist_diff = rotations[v] * neighbour.original_dist - neighbour.curr_dist;
		energy += 1.0 / 2.0 * (dist_diff * neighbour.cotangent_weights.asDiagonal() * dist_diff.transpose()).trace();
	}

	return energy;
}

MatrixXd* CubicMinimizer::getV()
{
	return &currV;
}

VectorXd* CubicMinimizer::getLambdas()
{
	return &lambda;
}
