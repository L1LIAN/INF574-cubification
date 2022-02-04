#include "pch.h"

#include "CubicMinimizer.h"

// set to 0 if you don't want the energy to be output
#define OUTPUT_ENERGY 0

using namespace Eigen; // to use the classes provided by Eigen library
using namespace std;

MatrixXd V;
MatrixXi F;

CubicMinimizer* cubicMinimizer;

// current lambda value
double current_lambda = 0.25;

// define a point in the mesh where its lambda value is fixed by the user
struct LambdaPoint {
	int vertex;
	double lambda;
	VectorXd geodesic;
	RowVector3d color;
};
// points where the cubeness factor is defined
vector<LambdaPoint> lambda_points;

FILE* output_file;


// draw the lambda points and the mesh
void draw(igl::opengl::glfw::Viewer& viewer) {


	viewer.data().clear();
	viewer.data().set_mesh(*cubicMinimizer->getV(), F);
	if (!lambda_points.empty()) {
		// we need to update it every time a cubic step is done
		MatrixXd lambda_draw = MatrixXd::Zero(lambda_points.size(), 3);
		MatrixXd lambda_color = MatrixXd::Zero(lambda_points.size(), 3);

		for (int i = 0; i < lambda_points.size(); i++) {
			lambda_draw.row(i) = cubicMinimizer->getV()->row(lambda_points[i].vertex);
			lambda_color.row(i) = lambda_points[i].color;
		}

		viewer.data().set_points(lambda_draw, lambda_color);
	}
}

// compute the lambda value for all the vertices
// the computation is done using barycentric coordinates
// where the coefficient is the inverse of the distance to the point
// this could be done in a better way using geodesics
void update_lambdas() {

	if (lambda_points.size() == 0) {
		// default value for all vertices
		for (int i = 0; i < V.rows(); i++) {
			(*cubicMinimizer->getLambdas())(i) = current_lambda;
		}
	}
	else {
		// this can be done in parallel, so let's do it
		igl::parallel_for(V.rows(), [](int v) {
			// sum the log instead of the values
			double sum_coef = 0.0;
			double sum_lambda = 0.0;
			RowVector3d vertex = V.row(v);
			for (auto& point : lambda_points) {
				if (point.vertex == v) {
					// this is the exact point, return (and prevent some division by 0)
					return point.lambda;
				}

				double coef = 1 / (vertex - V.row(point.vertex)).norm();//point.geodesic(v);
				sum_coef += coef;
				sum_lambda += coef * log(point.lambda);
			}
			(*cubicMinimizer->getLambdas())(v) = exp(sum_lambda / sum_coef);
			});
	}


	if (OUTPUT_ENERGY) {
		double energy = cubicMinimizer->get_energy();
		printf("New energy after lambda change is %lf\n", energy);
		fprintf(output_file, "%lf\n", energy);
	}
}


// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier)
{
	if (key == '1') {
		if (lambda_points.empty() && (*cubicMinimizer->getLambdas())(0) != current_lambda) {
			// this is in case the cubeness value is change without points on the figure
			update_lambdas();
		}

		cubicMinimizer->single_step();
		if (OUTPUT_ENERGY) {
			double energy = cubicMinimizer->get_energy();
			printf("New energy is %lf\n", energy);
			fprintf(output_file, "%lf\n", energy);
		}
		draw(viewer);
		return true;
	}
	
	return false;
}

// this function is called to add specific cubeness factors
// https://github.com/libigl/libigl/issues/750
bool mouse_down(igl::opengl::glfw::Viewer& viewer, int, int) {
	// the face that was clicked
	int fid;
	// barycentric coordinates of the triangle 
	Vector3f bc;
	// look at the code in the github issue
	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;
	if (igl::unproject_onto_mesh(Vector2f(x, y), viewer.core().view, viewer.core().proj,
		viewer.core().viewport, *cubicMinimizer->getV(), F, fid, bc)) {
		int face_vert;
		// we get the closest vertex using the barycentric coordinates
		bc.maxCoeff(&face_vert);

		// vertex that was touched
		int vertex = F(fid, face_vert);

		int pos = lambda_points.size();
		lambda_points.push_back({ vertex, current_lambda });

		// color of the point
		const RowVector3d red = RowVector3d(1.0, 0.0, 0.0);
		const RowVector3d blue = RowVector3d(0.0, 0.0, 1.0);

		// show the color on a log scale
		double cubeness_trimed = min(max(current_lambda, 1e-4), 10.0);
		double coeff_red = (log(10.0) - log(cubeness_trimed)) / (log(10.0) - log(1e-4));
		lambda_points[pos].color = coeff_red * red + (1 - coeff_red) * blue;

		// compute geodesic (libigl.github.io/tutorial/)
		VectorXi VS, FS, VT, FT;
		VS.resize(1);
		// distance from only one vertex
		VS << vertex;

		// to all vertex
		VT.setLinSpaced(V.rows(), 0, V.rows() - 1);

		lambda_points[pos].geodesic = VectorXd();
		igl::exact_geodesic(V, F, VS, FS, VT, FT, lambda_points[pos].geodesic);

		// update lambda
		update_lambdas();
		draw(viewer);
		return true;
	}

	return false;
}

// ------------ main program ----------------
int main(int argc, char *argv[])
{
	igl::opengl::glfw::Viewer viewer; // create the 3d viewer
	// add a menu
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);
	igl::readOBJ("../../../data/bob.obj", V, F);

	

	cubicMinimizer = new CubicMinimizer(&V, &F, current_lambda);
	if (OUTPUT_ENERGY) {
		output_file = fopen("energy_output.txt", "w");
		double energy = cubicMinimizer->get_energy();
		printf("Original energy is %lf\n", energy);
		fprintf(output_file, "%lf\n", energy);
	}

	viewer.callback_key_down = &key_down;
	viewer.callback_mouse_down = &mouse_down;

	menu.callback_draw_viewer_menu = [&]()
	{
		// this is from the libigl imgui tutorial
		ImGui::InputDouble("Cubeness factor", &current_lambda);

		// remove all points
		if (ImGui::Button("Reset points", ImVec2(-1, 0))) {
			lambda_points.clear();
			update_lambdas();
			draw(viewer);
		}

		// if we apply the polyhedral generalization
		ImGui::Checkbox("Polyhedral generalization (not working)", &cubicMinimizer->set_polyhedral);

	};

	//cubicMinimizer->single_step();

	viewer.data().set_mesh(*cubicMinimizer->getV(), F);
	
	viewer.launch(); // run the viewer
}
