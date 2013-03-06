#ifndef _ITASC_HQP_SOLVER_HPP
#define _ITASC_HQP_SOLVER_HPP

#include <rtt/TaskContext.hpp>
#include <algorithm>
#include <itasc_core/Solver.hpp>
#include <itasc_core/choleski_semidefinite.hpp>
#include <soth/HCOD.hpp>
#include <Eigen/LU>

#include <vector>
#include <string>
#include <sstream>

namespace iTaSC {

class HQPSolver: public Solver {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	HQPSolver(const std::string& name);
	virtual ~HQPSolver();//destructor

	virtual bool configureHook();
	virtual bool startHook();
	virtual void updateHook();
	virtual void stopHook();
	virtual void cleanupHook();
	virtual bool solve();

	void resizeSolver( bool has_wq );
	bool checkSolverSize( bool has_wq );

private:
	// the solver
	typedef boost::shared_ptr<soth::HCOD> hcod_ptr_t;
	hcod_ptr_t hsolver;

	// Jacobian of the tasks
	std::vector< Eigen::MatrixXd > Ctasks;

	// Upper/ Lower bounds for each tasks
	std::vector< soth::VectorBound > btasks;

	// Solution found by the solver. Size: size(dq) or size(dq+6)
	Eigen::VectorXd solution;

	// ?
	Eigen::VectorXd qdot;


	// attribute: number of priorities involved (default = 1)
	unsigned int priorityNo;

	// property: precision of matrix comparison
	double precision;


	std::vector<int> nc_priorities;
	RTT::InputPort<std::vector<int> > nc_priorities_port;

	// Port for the damping value.
	RTT::InputPort< double > damping_port;

	// Memory allocation
	Eigen::MatrixXd Wq;

	// Memory allocation matrix concerning the decomposition of Wq
	//  Wq corresponds to the weighting of the joint contribution
	Eigen::MatrixXd Lq;     //

	RTT::InputPort<Eigen::VectorXd> Wq_diag_port;
	Eigen::VectorXd Wq_diag;

	//std::vector priorities contains structs of the "Priority" type
	struct Priority
	{
	public:
		// nq is the number of joints
		// nc is the size of the error
		Priority(unsigned nc, unsigned nq);

	public:
		//// PORTS
		// port for the jacobian
		RTT::InputPort<Eigen::MatrixXd> A_port;

		// port for error desired
		RTT::InputPort<Eigen::MatrixXd> Wy_port;

		// port for the value of the error and the bound min.
		RTT::InputPort<Eigen::VectorXd> ydot_port;

		// port for the max bound.
		RTT::InputPort<Eigen::VectorXd> ydot_max_port;

		// port for the max bound.
		RTT::InputPort< std::vector<unsigned> > inequalities_port;


	public:
		//// PARAMETERS
		// error size
		unsigned int nc_priority;

		//generalized jacobian for a subtask with a certain priority
		Eigen::MatrixXd A_priority;

		//weight in the task space for the generalized jacobian = Wy = Ly^T Ly
		Eigen::MatrixXd Wy_priority;

		//task space coordinates
		Eigen::VectorXd ydot_priority;

		//Upper bound desired task value (optional)
		Eigen::VectorXd ydot_priority_max;

		// vector indicating the type of constraint considered (optional)
		// 0: equality,  1: lower inequality, 2: upper inequality,
		// 3: double bound inequality
		std::vector<unsigned> inequalities;

		// Memory allocation matrix concerning the decomposition of Wy
		Eigen::MatrixXd Ly;     //
	};

	std::vector<Priority*> priorities;

// temporary data only used to avoid time consuption at the creation.
private:
	std::string externalName;
	std::stringstream ssName;
};//end class definition

}//end namespace
#endif
